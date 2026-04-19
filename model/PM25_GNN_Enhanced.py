import torch
from torch import nn
from model.cells import GRUCell
from torch.nn import Sequential, Linear, Sigmoid
import numpy as np
from torch_scatter import scatter_add
from torch.nn import functional as F
from torch.nn import Parameter


class DynamicFeatureSelection(nn.Module):
    """
    Learnable gating mechanism that adaptively weights input features
    at each timestep. Uses a small MLP to produce per-feature importance
    scores via sigmoid gating, so the model can suppress irrelevant
    meteorological variables depending on current conditions.
    """
    def __init__(self, in_dim, hidden_dim=32):
        super(DynamicFeatureSelection, self).__init__()
        self.gate_net = Sequential(
            Linear(in_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, in_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, city_num, in_dim)
        gate = self.gate_net(x)       # (batch, city_num, in_dim)
        return x * gate, gate         # element-wise gating


class AttentionGraphGNN(nn.Module):
    """
    Enhanced GNN with multi-head attention for neighbor message weighting.

    Differences from the original GraphGNN:
    1. Multi-head attention: instead of simple scatter_add aggregation,
       edge messages are weighted by learned attention scores.
    2. The attention is computed from both the edge MLP output and the
       advection coefficient, combining learned and domain-knowledge signals.
    """
    def __init__(self, device, edge_index, edge_attr, in_dim, out_dim,
                 wind_mean, wind_std, num_heads=4):
        super(AttentionGraphGNN, self).__init__()
        self.device = device
        self.num_heads = num_heads
        self.edge_index = torch.LongTensor(edge_index).to(self.device)
        self.edge_attr = torch.Tensor(np.float32(edge_attr))
        self.edge_attr_norm = (self.edge_attr - self.edge_attr.mean(dim=0)) / self.edge_attr.std(dim=0)
        self.w = Parameter(torch.rand([1]))
        self.b = Parameter(torch.rand([1]))
        self.wind_mean = torch.Tensor(np.float32(wind_mean)).to(self.device)
        self.wind_std = torch.Tensor(np.float32(wind_std)).to(self.device)

        e_h = 32
        e_out = 30
        n_out = out_dim

        # Edge MLP: same input structure as original
        self.edge_mlp = Sequential(
            Linear(in_dim * 2 + 2 + 1, e_h),
            Sigmoid(),
            Linear(e_h, e_out),
            Sigmoid(),
        )

        # Multi-head attention: compute attention score per head from edge representation
        self.attn_heads = nn.ModuleList([
            Sequential(
                Linear(e_out + 1, 16),  # e_out (edge msg) + 1 (advection coeff)
                nn.Tanh(),
                Linear(16, 1),
                nn.LeakyReLU(0.2),
            )
            for _ in range(num_heads)
        ])

        # Combine multi-head outputs
        self.head_combine = Linear(e_out * num_heads, e_out)

        # Node MLP: same as original
        self.node_mlp = Sequential(
            Linear(e_out, n_out),
            Sigmoid(),
        )

    def forward(self, x):
        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)

        edge_src, edge_target = self.edge_index
        node_src = x[:, edge_src]       # (batch, num_edges, in_dim)
        node_target = x[:, edge_target]

        # --- Domain-knowledge advection coefficient (same as original) ---
        src_wind = node_src[:, :, -2:] * self.wind_std[None, None, :] + self.wind_mean[None, None, :]
        src_wind_speed = src_wind[:, :, 0]
        src_wind_direc = src_wind[:, :, 1]
        self.edge_attr_ = self.edge_attr[None, :, :].repeat(node_src.size(0), 1, 1)
        city_dist = self.edge_attr_[:, :, 0]
        city_direc = self.edge_attr_[:, :, 1]

        theta = torch.abs(city_direc - src_wind_direc)
        edge_weight = F.relu(3 * src_wind_speed * torch.cos(theta) / city_dist)
        edge_weight = edge_weight.to(self.device)

        # --- Edge messages ---
        edge_attr_norm = self.edge_attr_norm[None, :, :].repeat(node_src.size(0), 1, 1).to(self.device)
        edge_input = torch.cat([node_src, node_target, edge_attr_norm, edge_weight[:, :, None]], dim=-1)
        edge_msg = self.edge_mlp(edge_input)  # (batch, num_edges, e_out)

        # --- Multi-head attention ---
        # Attention input: edge message + advection coefficient
        attn_input = torch.cat([edge_msg, edge_weight[:, :, None]], dim=-1)

        head_outputs = []
        for head in self.attn_heads:
            # Compute raw attention scores
            attn_score = head(attn_input).squeeze(-1)  # (batch, num_edges)

            # Softmax over incoming edges for each target node
            # We use scatter-based softmax: exp(score) / sum(exp(score)) per target
            attn_exp = torch.exp(attn_score - attn_score.max())  # numerical stability
            attn_sum = scatter_add(attn_exp, edge_target, dim=1, dim_size=x.size(1))
            attn_norm = attn_exp / (attn_sum[:, edge_target] + 1e-8)

            # Weighted messages (import)
            weighted_import = edge_msg * attn_norm[:, :, None]
            head_agg_import = scatter_add(weighted_import, edge_target, dim=1, dim_size=x.size(1))

            # Export component (same attention applied in reverse direction)
            weighted_export = edge_msg * attn_norm[:, :, None]
            head_agg_export = scatter_add(weighted_export.neg(), edge_src, dim=1, dim_size=x.size(1))

            head_out = head_agg_import + head_agg_export  # (batch, city_num, e_out)
            head_outputs.append(head_out)

        # Concatenate heads and combine
        multi_head = torch.cat(head_outputs, dim=-1)       # (batch, city_num, e_out * num_heads)
        out = self.head_combine(multi_head)                 # (batch, city_num, e_out)
        out = self.node_mlp(out)                            # (batch, city_num, n_out)

        return out


class PM25_GNN_Enhanced(nn.Module):
    """
    Enhanced PM2.5-GNN with:
    1. Dynamic Feature Selection - learnable gating on input features
    2. Attention-enhanced GNN - multi-head attention on neighbor messages
    3. Same GRU temporal module as original for fair comparison

    The forward pass:
        input features -> dynamic feature selection -> attention GNN -> GRU -> prediction
    """
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size,
                 device, edge_index, edge_attr, wind_mean, wind_std,
                 num_heads=4, dropout=0.1):
        super(PM25_GNN_Enhanced, self).__init__()

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size

        self.in_dim = in_dim
        self.hid_dim = 64
        self.out_dim = 1
        self.gnn_out = 13
        self.dropout = dropout

        # Dynamic feature selection gate
        self.feature_gate = DynamicFeatureSelection(self.in_dim)

        # Attention-enhanced GNN
        self.graph_gnn = AttentionGraphGNN(
            self.device, edge_index, edge_attr,
            self.in_dim, self.gnn_out, wind_mean, wind_std,
            num_heads=num_heads
        )

        # Temporal GRU (same as original)
        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.gru_cell = GRUCell(self.in_dim + self.gnn_out, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

        # Dropout for regularization
        self.drop = nn.Dropout(dropout)

    def forward(self, pm25_hist, feature, return_gates=False):
        pm25_pred = []
        gate_list = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        xn = pm25_hist[:, -1]

        for i in range(self.pred_len):
            x = torch.cat((xn, feature[:, self.hist_len + i]), dim=-1)

            # Dynamic feature selection
            x_gated, gate_weights = self.feature_gate(x)
            if return_gates:
                gate_list.append(gate_weights.detach())

            # Attention-enhanced GNN for spatial transport
            xn_gnn = x_gated.contiguous()
            xn_gnn = self.graph_gnn(xn_gnn)

            # Combine GNN output with gated features
            x_combined = torch.cat([xn_gnn, x_gated], dim=-1)
            x_combined = self.drop(x_combined)

            # Temporal GRU
            hn = self.gru_cell(x_combined, hn)
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)

        pm25_pred = torch.stack(pm25_pred, dim=1)

        if return_gates:
            return pm25_pred, torch.stack(gate_list, dim=1)
        return pm25_pred

    def predict_with_uncertainty(self, pm25_hist, feature, n_samples=20):
        """
        MC Dropout: run forward pass multiple times with dropout ON
        to get mean prediction + uncertainty estimate.
        """
        self.train()  # keep dropout active
        preds = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(pm25_hist, feature)
                preds.append(pred)
        preds = torch.stack(preds, dim=0)
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0)
        self.eval()
        return mean_pred, std_pred
