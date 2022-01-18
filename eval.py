from os import path as osp
import os
from argparse import Namespace
import numpy as np
import tqdm
from PIL import Image
import yaml
import argparse
import pickle
import cv2

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import transforms

from data_aug.augment import Denormalize
from data_aug.dataset import Eval_style, Eval_content
from model.model import Generator

from imaginaire.utils.model_average import ModelAverage

parser = argparse.ArgumentParser()

#general settings
parser.add_argument('--d_id', type=int, default=0, help='cuda device id')
parser.add_argument('--exp', type=str, default='test', metavar='x', help='name of experiment')
parser.add_argument('--eval_video', help='load style from other data class', action='store_true')

class Evaluation(object):
    def __init__(self, args):
        self.args = args
        self.exp_path = osp.join("train", self.args.exp)
        self.exp_args = self.load_obj("args.obj")
        self.config = self.load_obj("config.obj")
        print(self.exp_args)

        self.eval_content_loader = DataLoader(Eval_content(self.exp_args, "eval_content"), 1, False)
        self.eval_style_loader = DataLoader(Eval_style(self.exp_args, "eval_style_subfolder"), 1, False)
        self.denormalize = Denormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        self.gen_model = Generator(self.exp_args, self.config)
        if self.args.d_id != -1:
            self.gen_model = self.gen_model.cuda()
        if self.exp_args.model_avg:
            self.gen_model = ModelAverage(self.gen_model, 0.999, 1000, True)
        self.load_weights()
        self.gen_model.eval()
        if self.args.d_id != -1: self.gen_model = self.gen_model.cuda()
        self.path = osp.join(self.exp_path, "eval_results", "Overall_" + str(self.iter_counter))
        os.makedirs(osp.join(self.path), exist_ok=True)

    def load_obj(self, file):
        try:
            return pickle.load(open(osp.join("train", self.args.exp, file), 'rb'))
        except:
            raise ("cannot load file: " + str(file))

    def load_weights(self):
        try:
            path = osp.join(self.exp_path, "checkpoint.pth")
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
            try:
                self.gen_model.load_state_dict(checkpoint['net_G'])
                print("loading weights successful")
            except:
                try:
                    weight_items = list(checkpoint['net_G'].items())
                    my_model_kvpair = self.gen_model.state_dict()
                    for i, key in enumerate(my_model_kvpair):
                        layer_name, weights = weight_items[i]
                        my_model_kvpair[key] = weights
                    self.gen_model.load_state_dict(my_model_kvpair)
                    print("loading weights successful")
                except:
                    raise ("Not able to load data to state dict")

            self.iter_counter = checkpoint['current_iteration']
        except:
            raise ("loading model failed")

    def load_args(self):
        try:
            with open(os.path.join(self.exp_path, "args.txt")) as f:
                args = f.read()
            args = Namespace(**yaml.load(args, Loader=yaml.FullLoader))
            setattr(args, "d_id", args.d_id)
            setattr(args, "load_checkpoint", True)
            return args
        except:
            raise("failed loading arguments")

    def forward_style(self, image):
        if self.exp_args.model_avg:
            return self.gen_model.averaged_model.G.style_encoder(image)
        else:
            return self.gen_model.G.style_encoder(image)

    def forward_content(self, image):
        if self.exp_args.model_avg:
            return self.gen_model.averaged_model.G.content_encoder(image)
        else:
            return self.gen_model.G.content_encoder(image)

    def forward_decode(self, content_feat, style_feat):
        if self.exp_args.model_avg:
            return self.gen_model.averaged_model.G.decode(content_feat, style_feat)
        else:
            return self.gen_model.G.decode(content_feat, style_feat)

    def get_grid_img(self, image):
        grid = vutils.make_grid(image, nrow=image.size()[0], padding=0)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        return ndarr

    def prepare_img(self, img):
        img = self.denormalize(img)
        return self.get_grid_img(img)

    def adjust_img_like(self, img, counterpart, axis):
        try:
            _, _, w, h = counterpart.size()
            _, _, s_w, s_h = img.size()
        except:
            w, h, _ = np.shape(counterpart)
            s_w, s_h, _ = np.shape(img)

        if axis == 0:
            factor = h / s_h
        else:
            factor = w / s_w

        s_w = int(s_w * factor)
        s_h = int(s_h * factor)

        try:
            return transforms.Resize((s_w, s_h))(img)
        except:
            return np.resize(img, (s_w, s_h, 3))

    def resize_to_edge(self, resize_to, image):
        width, height = image.size
        factor = resize_to / height
        return image.resize((int(image.width * factor), int(image.height * factor)))

    def add_pixels_to_fill_up(self, out, res):
        size_out = np.shape(out)
        size_res = np.shape(res)
        if size_out[0] > size_res[0]:
            res = Image.fromarray(res)
            res = self.resize_to_edge(size_out[0], res)
            res = np.asarray(res)
        if size_out[0] < size_res[0]:
            out = Image.fromarray(out)
            out = self.resize_to_edge(size_res[0], out)
            out = np.asarray(out)
        return out, res


    def eval_video(self):
        video_content = osp.join("data", "eval_video")
        video_style = osp.join("data", "eval_video_style")
        to_pil = transforms.ToPILImage()
        transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.5), std=(0.5))])
        for style in os.listdir(video_style):
            style = Image.open(osp.join(video_style, style))
            style = transform(style).unsqueeze(0)
            if self.args.d_id != -1:
                style = style.cuda()

            for i, video in enumerate(os.listdir(video_content)):
                vidcap = cv2.VideoCapture(osp.join(video_content, video))
                success, content = vidcap.read()
                count = 0
                height, width, layers = content.shape
                size = (width, height)
                out = cv2.VideoWriter(osp.join(self.path, 'output_video' + str(i) + '.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
                style_feat = self.forward_style(style)
                while success:
                    content = cv2.cvtColor(np.array(content), cv2.COLOR_BGR2RGB)
                    content = Image.fromarray(content)
                    content = transform(content)
                    content = content.unsqueeze(0)
                    if self.args.d_id != -1:
                        content = content.cuda()
                    styled_img = self.forward_decode(self.forward_content(content), style_feat)
                    styled_img = self.denormalize(styled_img)
                    styled_img = to_pil(styled_img.squeeze(0)).convert("RGB")
                    styled_img = np.array(styled_img).astype('uint8')
                    # Convert RGB to BGR
                    styled_img = styled_img[:, :, ::-1].copy()

                    # styled_img = np.asarray(to_pil(styled_img.squeeze(0)).convert("RGB")).astype('uint8')
                    # styled_img = np.uint8(styled_img[0].detach().numpy().transpose(1, 2, 0).convert("RGB"))
                    # print(styled_img)
                    # print(np.shape(styled_img))
                    out.write(styled_img)
                    # save frame as JPEG file
                    success, content = vidcap.read()
                    count += 1
                vidcap.release()
                out.release()

    def get_style_feature(self, features):
        if self.exp_args.train_multi_style:
            dist = [0.5, 0.5]
        else:
            dist = [1.0]

        ones_like = torch.ones_like(features[0])
        style_feat =  ones_like * dist[0] * features[0]
        if self.exp_args.train_multi_style:
            style_feat += ones_like * ones_like * dist[1] * features[1]
        return style_feat

    def evaluate(self):
        if self.args.eval_video:
            self.eval_video()
        else:
            for i, content_img in tqdm.tqdm(enumerate(self.eval_content_loader), desc='Content : '):
                out = None
                for style_img, style_img_2, _ in self.eval_style_loader:
                    if self.args.d_id != -1:
                        style_img = style_img.cuda()
                        content_img = content_img.cuda()
                        if self.exp_args.train_multi_style:
                            style_img_2 = style_img_2.cuda()

                    with torch.no_grad():
                        style_feat = self.forward_style(style_img)
                        if self.exp_args.train_multi_style:
                            style_feat_2 = self.forward_style(style_img_2)
                            style_feat = self.get_style_feature([style_feat, style_feat_2])

                        styled_img = self.forward_decode(self.forward_content(content_img), style_feat)

                    styled_img = self.adjust_img_like(styled_img, content_img, 0)
                    style_img = self.adjust_img_like(style_img, content_img, 0)

                    content = self.prepare_img(content_img)
                    style = self.prepare_img(style_img)
                    styled = self.prepare_img(styled_img)

                    if self.exp_args.train_multi_style:
                        style_img_2 = self.adjust_img_like(style_img_2, content_img, 0)
                        style_2 = self.prepare_img(style_img_2)
                        res = np.concatenate([content, style, style_2, styled], axis=0)
                    else:
                        res = np.concatenate([content, style, styled], axis=0)

                    if not isinstance(out, np.ndarray):
                        out = res
                        size, _, _ = np.shape(out)
                    else:
                        out, res = self.add_pixels_to_fill_up(out, res)
                        out = np.concatenate([out, res], axis=1)

                out = Image.fromarray(out)
                out.save(osp.join(self.path, str(i) + "_trans_img.jpg"))

if __name__ == "__main__":
    args = parser.parse_args()
    Evaluation(args).evaluate()
