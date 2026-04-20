import yaml
import sys
import os
import platform
import numpy as np

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
conf_fp = os.path.join(proj_dir, 'config.yaml')
with open(conf_fp) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

nodename = os.uname().nodename if hasattr(os, "uname") else platform.node()

if nodename in config['filepath']:
    file_dir = config['filepath'][nodename]
else:
    file_dir = {
        'knowair_fp': os.path.join(proj_dir, 'data', 'KnowAir.npy'),
        'results_dir': os.path.join(proj_dir, 'results'),
    }

def main():
    pass

if __name__ == '__main__':
    main()
