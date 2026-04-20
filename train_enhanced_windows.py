import os
import sys
import platform
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)

# Load enhanced config before importing anything else
import yaml
conf_fp = os.path.join(proj_dir, 'config_enhanced.yaml')
with open(conf_fp) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Resolve file paths with hostname fallback
nodename = platform.node()
if nodename in config['filepath']:
    file_dir = config['filepath'][nodename]
else:
    file_dir = {
        'knowair_fp': os.path.join(proj_dir, 'data/KnowAir.npy'),
        'results_dir': os.path.join(proj_dir, 'results'),
    }

# Patch util module so graph.py and dataset.py can import from it
import util_windows_fixed as util
util.config = config
util.file_dir = file_dir

from graph import Graph
from dataset_windows import HazeData

from model.MLP import MLP
from model.LSTM import LSTM
from model.GRU import GRU
from model.GC_LSTM import GC_LSTM
from model.nodesFC_GRU import nodesFC_GRU
from model.PM25_GNN import PM25_GNN
from model.PM25_GNN_nosub import PM25_GNN_nosub
from model.PM25_GNN_Enhanced import PM25_GNN_Enhanced
from losses import FrequencyWeightedLoss, CombinedLoss

import arrow
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pickle
import glob
import shutil

torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

graph = Graph()
city_num = graph.node_num

batch_size = config['train']['batch_size']
epochs = config['train']['epochs']
hist_len = config['train']['hist_len']
pred_len = config['train']['pred_len']
weight_decay = config['train']['weight_decay']
early_stop = config['train']['early_stop']
lr = config['train']['lr']
results_dir = file_dir['results_dir']
dataset_num = config['experiments']['dataset_num']
exp_model = config['experiments']['model']
exp_repeat = config['train']['exp_repeat']
save_npy = config['experiments']['save_npy']

# Loss configuration
loss_type = config['experiments'].get('loss', 'MSE')
fmae_beta = config['experiments'].get('fmae_beta', 0.8)
fmae_alpha = config['experiments'].get('fmae_alpha', 0.5)

# Enhanced model configuration
num_heads = config['experiments'].get('num_heads', 4)
dropout = config['experiments'].get('dropout', 0.1)

train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train')
val_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val')
test_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test')
in_dim = train_data.feature.shape[-1] + train_data.pm25.shape[-1]
wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
pm25_mean, pm25_std = test_data.pm25_mean, test_data.pm25_std


def get_criterion():
    """Select loss function based on config."""
    if loss_type == 'MSE':
        print('[Loss] Using MSE loss')
        return nn.MSELoss()
    elif loss_type == 'fMAE':
        print('[Loss] Using Frequency-weighted MAE loss (beta=%.2f)' % fmae_beta)
        fmae = FrequencyWeightedLoss(
            beta=fmae_beta,
            pm25_mean=float(pm25_mean),
            pm25_std=float(pm25_std)
        )
        # Initialize bins from raw training PM2.5 data
        raw_pm25 = train_data.pm25 * pm25_std + pm25_mean
        fmae.initialize_bins(raw_pm25)
        return fmae
    elif loss_type == 'Combined':
        print('[Loss] Using Combined loss (alpha=%.2f, beta=%.2f)' % (fmae_alpha, fmae_beta))
        combined = CombinedLoss(
            alpha=fmae_alpha,
            beta=fmae_beta,
            pm25_mean=float(pm25_mean),
            pm25_std=float(pm25_std)
        )
        raw_pm25 = train_data.pm25 * pm25_std + pm25_mean
        combined.initialize_bins(raw_pm25)
        return combined
    else:
        raise Exception('Wrong loss type! Choose from: MSE, fMAE, Combined')


criterion = get_criterion()


def get_metric(predict_epoch, label_epoch):
    haze_threshold = 75
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold
    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
    csi = hit / (hit + falsealarm + miss)
    pod = hit / (hit + miss)
    far = falsealarm / (hit + falsealarm)
    predict = predict_epoch[:,:,:,0].transpose((0,2,1))
    label = label_epoch[:,:,:,0].transpose((0,2,1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    mae = np.mean(np.mean(np.abs(predict - label), axis=1))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    # R² score
    ss_res = np.sum(np.square(predict - label))
    ss_tot = np.sum(np.square(label - np.mean(label)))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return rmse, mae, csi, pod, far, r2


def get_exp_info():
    exp_info =  '============== Train Info ==============\n' + \
                'Dataset number: %s\n' % dataset_num + \
                'Model: %s\n' % exp_model + \
                'Loss: %s\n' % loss_type + \
                'Train: %s --> %s\n' % (train_data.start_time, train_data.end_time) + \
                'Val: %s --> %s\n' % (val_data.start_time, val_data.end_time) + \
                'Test: %s --> %s\n' % (test_data.start_time, test_data.end_time) + \
                'City number: %s\n' % city_num + \
                'Use metero: %s\n' % config['experiments']['metero_use'] + \
                'batch_size: %s\n' % batch_size + \
                'epochs: %s\n' % epochs + \
                'hist_len: %s\n' % hist_len + \
                'pred_len: %s\n' % pred_len + \
                'weight_decay: %s\n' % weight_decay + \
                'early_stop: %s\n' % early_stop + \
                'lr: %s\n' % lr
    if exp_model == 'PM25_GNN_Enhanced':
        exp_info += 'num_heads: %s\n' % num_heads + \
                    'dropout: %s\n' % dropout
    if loss_type in ['fMAE', 'Combined']:
        exp_info += 'fmae_beta: %s\n' % fmae_beta
    if loss_type == 'Combined':
        exp_info += 'fmae_alpha: %s\n' % fmae_alpha
    exp_info += '========================================\n'
    return exp_info


def get_model():
    if exp_model == 'MLP':
        return MLP(hist_len, pred_len, in_dim)
    elif exp_model == 'LSTM':
        return LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GRU':
        return GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'nodesFC_GRU':
        return nodesFC_GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GC_LSTM':
        return GC_LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index)
    elif exp_model == 'PM25_GNN':
        return PM25_GNN(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'PM25_GNN_nosub':
        return PM25_GNN_nosub(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'PM25_GNN_Enhanced':
        return PM25_GNN_Enhanced(hist_len, pred_len, in_dim, city_num, batch_size, device,
                                 graph.edge_index, graph.edge_attr, wind_mean, wind_std,
                                 num_heads=num_heads, dropout=dropout)
    else:
        raise Exception('Wrong model name!')


def train(train_loader, model, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, data in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= batch_idx + 1
    return train_loss


def val(val_loader, model):
    model.eval()
    val_loss = 0
    for batch_idx, data in tqdm(enumerate(val_loader)):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        val_loss += loss.item()

    val_loss /= batch_idx + 1
    return val_loss


def test(test_loader, model):
    model.eval()
    predict_list = []
    label_list = []
    time_list = []
    test_loss = 0
    for batch_idx, data in enumerate(test_loader):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        test_loss += loss.item()

        pm25_pred_val = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1) * pm25_std + pm25_mean
        pm25_label_val = pm25.cpu().detach().numpy() * pm25_std + pm25_mean
        predict_list.append(pm25_pred_val)
        label_list.append(pm25_label_val)
        time_list.append(time_arr.cpu().detach().numpy())

    test_loss /= batch_idx + 1

    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    time_epoch = np.concatenate(time_list, axis=0)
    predict_epoch[predict_epoch < 0] = 0

    return test_loss, predict_epoch, label_epoch, time_epoch


def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std()


def main():
    exp_info = get_exp_info()
    print(exp_info)

    exp_time = arrow.now().format('YYYYMMDDHHmmss')

    train_loss_list, val_loss_list, test_loss_list, rmse_list, mae_list, csi_list, pod_list, far_list, r2_list = [], [], [], [], [], [], [], [], []

    for exp_idx in range(exp_repeat):
        print('\nNo.%2d experiment ~~~' % exp_idx)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

        model = get_model()
        model = model.to(device)
        model_name = type(model).__name__

        print(str(model))

        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

        exp_model_dir = os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), model_name, str(exp_time), '%02d' % exp_idx)
        if not os.path.exists(exp_model_dir):
            os.makedirs(exp_model_dir)
        model_fp = os.path.join(exp_model_dir, 'model.pth')

        val_loss_min = 100000
        best_epoch = 0

        train_loss_, val_loss_ = 0, 0

        for epoch in range(epochs):
            print('\nTrain epoch %s:' % (epoch))

            train_loss = train(train_loader, model, optimizer)
            val_loss = val(val_loader, model)

            print('train_loss: %.4f' % train_loss)
            print('val_loss: %.4f' % val_loss)

            if epoch - best_epoch > early_stop:
                break

            if val_loss < val_loss_min:
                val_loss_min = val_loss
                best_epoch = epoch
                print('Minimum val loss!!!')
                torch.save(model.state_dict(), model_fp)
                print('Save model: %s' % model_fp)

                test_loss, predict_epoch, label_epoch, time_epoch = test(test_loader, model)
                train_loss_, val_loss_ = train_loss, val_loss
                rmse, mae, csi, pod, far, r2 = get_metric(predict_epoch, label_epoch)
                print('Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f, R2: %0.4f' % (train_loss_, val_loss_, test_loss, rmse, mae, csi, pod, far, r2))

                if save_npy:
                    np.save(os.path.join(exp_model_dir, 'predict.npy'), predict_epoch)
                    np.save(os.path.join(exp_model_dir, 'label.npy'), label_epoch)
                    np.save(os.path.join(exp_model_dir, 'time.npy'), time_epoch)

        # --- Feature importance (explainability) ---
        if exp_model == 'PM25_GNN_Enhanced':
            print('Extracting feature importance from gate weights...')
            model.load_state_dict(torch.load(model_fp))
            model.eval()
            gate_all = []
            for batch_idx, data in enumerate(test_loader):
                pm25, feature, time_arr = data
                pm25 = pm25.to(device)
                feature = feature.to(device)
                pm25_hist = pm25[:, :hist_len]
                _, gates = model(pm25_hist, feature, return_gates=True)
                gate_all.append(gates.cpu().numpy())
            gate_all = np.concatenate(gate_all, axis=0)
            # Average gate weights across batch, time, and cities -> per-feature importance
            feature_importance = gate_all.mean(axis=(0, 1, 2))
            np.save(os.path.join(exp_model_dir, 'feature_importance.npy'), feature_importance)
            np.save(os.path.join(exp_model_dir, 'gate_weights.npy'), gate_all)
            metero_use = config['experiments']['metero_use']
            feat_names = metero_use + ['hour', 'weekday', 'wind_speed', 'wind_direc', 'pm25']
            print('Feature importance:')
            for fname, fimp in zip(feat_names, feature_importance):
                print('  %s: %.4f' % (fname, fimp))

            # --- Uncertainty estimation (MC Dropout) ---
            print('Running MC Dropout uncertainty estimation (20 samples)...')
            uncertainty_list = []
            for batch_idx, data in enumerate(test_loader):
                pm25, feature, time_arr = data
                pm25 = pm25.to(device)
                feature = feature.to(device)
                pm25_hist = pm25[:, :hist_len]
                _, std_pred = model.predict_with_uncertainty(pm25_hist, feature, n_samples=20)
                uncertainty_list.append(std_pred.cpu().numpy())
            uncertainty_epoch = np.concatenate(uncertainty_list, axis=0)
            # Denormalize uncertainty
            uncertainty_epoch = uncertainty_epoch * pm25_std
            np.save(os.path.join(exp_model_dir, 'uncertainty.npy'), uncertainty_epoch)
            print('Mean uncertainty (std): %.2f µg/m³' % np.mean(uncertainty_epoch))

        train_loss_list.append(train_loss_)
        val_loss_list.append(val_loss_)
        test_loss_list.append(test_loss)
        rmse_list.append(rmse)
        mae_list.append(mae)
        csi_list.append(csi)
        pod_list.append(pod)
        far_list.append(far)
        r2_list.append(r2)

        print('\nNo.%2d experiment results:' % exp_idx)
        print(
            'Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f, RMSE: %0.2f, MAE: %0.2f, CSI: %0.4f, POD: %0.4f, FAR: %0.4f, R2: %0.4f' % (
            train_loss_, val_loss_, test_loss, rmse, mae, csi, pod, far, r2))

    exp_metric_str = '---------------------------------------\n' + \
                     'train_loss | mean: %0.4f std: %0.4f\n' % (get_mean_std(train_loss_list)) + \
                     'val_loss   | mean: %0.4f std: %0.4f\n' % (get_mean_std(val_loss_list)) + \
                     'test_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(test_loss_list)) + \
                     'RMSE       | mean: %0.4f std: %0.4f\n' % (get_mean_std(rmse_list)) + \
                     'MAE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(mae_list)) + \
                     'R2         | mean: %0.4f std: %0.4f\n' % (get_mean_std(r2_list)) + \
                     'CSI        | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_list)) + \
                     'POD        | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_list)) + \
                     'FAR        | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_list))

    metric_fp = os.path.join(os.path.dirname(exp_model_dir), 'metric.txt')
    with open(metric_fp, 'w') as f:
        f.write(exp_info)
        f.write(str(model))
        f.write(exp_metric_str)

    print('=========================\n')
    print(exp_info)
    print(exp_metric_str)
    print(str(model))
    print(metric_fp)


if __name__ == '__main__':
    main()
