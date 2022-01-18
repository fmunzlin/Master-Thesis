import argparse
import torch
import numpy as np
import random
import os
import os.path as osp
import pickle

from data_aug.dataset import Content_dataset, Style_dataset
from scripts.trainer import Training
from model.model_config import Config
from model.model import Network

parser = argparse.ArgumentParser()

#general settings
parser.add_argument('--d_id', type=int, default=0, help='')
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--exp', type=str, default='', help='')

# training
parser.add_argument('--G_step', type=int, default=1, help='')
parser.add_argument('--D_step', type=int, default=1, help='')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--crop_size', type=int, default=256, help='')
parser.add_argument('--D_noise', type=float, default=-1.0, help='x')
parser.add_argument('--max_steps', type=int, default=150000, help='')
parser.add_argument('--scheduler_stepsize', type=int, default=-1, help='x')

# Model
parser.add_argument('--config_file', type=str, default='512_c4_s5.yaml', help='x')
parser.add_argument('--vgg', help='x', action='store_true')
parser.add_argument('--coco_funit', help='x', action='store_true')
parser.add_argument('--model_avg', help='x', action='store_true')
parser.add_argument('--spectral_norm_G', help='x', action='store_true')
parser.add_argument('--init_G', type=str, default='none', help='x')
parser.add_argument('--init_D', type=str, default='none', help='x')
parser.add_argument('--load_checkpoint', help='', action='store_true')

# optimizer
parser.add_argument('--lr_G', type=float, default=0.0001, help='x')
parser.add_argument('--lr_D', type=float, default=0.0001, help='x')
parser.add_argument('--beta1_G', type=float, default=0.5, help='x')
parser.add_argument('--beta1_D', type=float, default=0.05, help='x')
parser.add_argument('--beta2_G', type=float, default=0.999, help='x')
parser.add_argument('--beta2_D', type=float, default=0.999, help='x')
parser.add_argument('--eps_G', type=float, default=1e-5, help='x')
parser.add_argument('--eps_D', type=float, default=10e-5, help='x')

# dataset
parser.add_argument('--artist_id', type=int, default=-1, help='')
parser.add_argument('--val_size', type=int, default=100, help='')
parser.add_argument('--val_step', type=int, default=2000, help='')
parser.add_argument('--resize', type=int, default=270, help='')
parser.add_argument('--scale_aug', type=float, default=0.1, help='')
parser.add_argument('--dif_style', help='x', action='store_true')
parser.add_argument('--train_multi_style', help='x', action='store_true')

# loss
parser.add_argument('--lrp', type=float, default=1.0, help='x')
parser.add_argument('--lfm', type=float, default=1.0, help='')
parser.add_argument('--lg', type=float, default=1.0, help='')
parser.add_argument('--gan_loss_method', type=str, default='hinge', help='x')

# plots
parser.add_argument('--sifid_images', type=int, default=50, help='')
parser.add_argument('--moving_avg', type=int, default=20, help='')
parser.add_argument('--save_img', type=int, default=1000, help='')

def create_folder(exp):
    exp_path = osp.join("train", exp)
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(osp.join(exp_path, "snapshots"), exist_ok=True)
    os.makedirs(osp.join(exp_path, "plots"), exist_ok=True)
    os.makedirs(osp.join(exp_path, "objects"), exist_ok=True)
    os.makedirs(osp.join(exp_path, "plots", "sifid_images", "true"), exist_ok=True)
    os.makedirs(osp.join(exp_path, "plots", "sifid_images", "false"), exist_ok=True)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_obj(exp, obj, file):
    try: pickle.dump(obj, open(osp.join("train", exp, "objects", file), 'wb'))
    except: raise("cannot save config file")

def load_obj(exp, file):
    try: return pickle.load(open(osp.join("train", exp, "objects", file), 'rb'))
    except: raise("cannot load file: " + str(file))

def get_args(args):
    if args.load_checkpoint:
        exp_args = load_obj(args.exp, "args.obj")
        setattr(exp_args, "d_id", args.d_id)
        setattr(exp_args, "load_checkpoint", True)
        setattr(exp_args, "max_steps", args.max_steps)
        args = exp_args
    else:
        save_obj(args.exp, args, "args.obj")
        keys = []
        for action in parser._actions:
            if getattr(action, "help") != "": keys.append(getattr(action, "option_strings")[0][2:])

        with open(osp.join("train", args.exp, f"args.txt"), 'w', encoding='utf8') as f:
            for key, value in vars(args).items():
                if key in keys: f.write(str(key) + ": " + str(value) + "\n")
    return args

def get_config(args, num_labels):
    if args.load_checkpoint: config = load_obj(args.exp, "config.obj")
    else:
        config = Config(args, num_labels)
        save_obj(args.exp, config, "config.obj")
    return config

def main(args):
    if torch.cuda.is_available(): torch.cuda.set_device(args.d_id)
    create_folder(args.exp)
    args = get_args(args)
    content_dataset = Content_dataset(args, "Places365")
    style_dataset = Style_dataset(args, "images_720_2")
    num_labels = style_dataset.num_labels
    config = get_config(args, num_labels)
    print("Style: Train discriminator on " + str(num_labels) + " classes")

    network = Network(args, config)
    if args.d_id != -1: network = network.cuda()
    Training(args, network, style_dataset, content_dataset).train()

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    set_random_seed(args.seed)
    main(args)

