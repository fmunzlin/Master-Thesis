import math
import numpy as np
import os.path as osp
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torchvision.utils as vutils

from data_aug.augment import Denormalize

from imaginaire.generators.funit import Decoder, MLP
from imaginaire.generators.coco_funit import COCOFUNITTranslator
from imaginaire.generators.funit import FUNITTranslator
from imaginaire.discriminators.funit import ResDiscriminator
from imaginaire.utils.init_weight import weights_init

def get_grid_img(image):
    grid = vutils.make_grid(image, nrow=image.size()[0])
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return ndarr

def init_model(model, init):
    if init == "none":
        gain = 1
    elif init == "xavier":
        gain = math.sqrt(2)
    elif init == "xavier_uniform":
        gain = 1
    elif init  == "orthogonal":
        gain = 1
    else:
        return model
    return model.apply(weights_init(init_type=init, gain=gain, bias=None))

class Network(nn.Module):
    def __init__(self, args, config):
        super(Network, self).__init__()
        self.args = args
        self.config = config
        self.G = Generator(self.args, self.config)
        self.D = Discriminator(self.args, self.config)

        self.denormalize = Denormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        self.model_checkpoints_folder = osp.join("train", self.args.exp)
        self.text_images = self.load_text_images()
        self.sifid_counter = 0
        self.mode = ""

    def gen_forward(self, content_data, style_data):
        gen_obj, self.snapshots = self.G.forward(content_data, style_data)
        dis_obj = self.D.forward(gen_obj, style_data)
        if self.mode == "val" and self.sifid_counter <= self.args.sifid_images:
            self.save_sifid_images(gen_obj)
        return gen_obj, dis_obj

    def dis_forward(self, content_data, style_data):
        with torch.no_grad():
            gen_obj, self.snapshots = self.G.forward(content_data, style_data)
        gen_obj['styled_image'].requires_grad = True
        dis_obj = self.D.forward(gen_obj, style_data)
        return gen_obj, dis_obj

    def save_sifid_images(self, gen_obj):
        path = osp.join(self.model_checkpoints_folder, "plots", "sifid_images")
        for style, styled in zip(gen_obj["style_image_1"], gen_obj["styled_image"]):
            if self.sifid_counter == self.args.sifid_images:
                break
            img = transforms.ToPILImage()(self.denormalize(style))
            img.save(osp.join(path, "true", str(self.sifid_counter) + ".jpg"))

            img = transforms.ToPILImage()(self.denormalize(styled))
            img.save(osp.join(path, "false", str(self.sifid_counter) + ".jpg"))
            self.sifid_counter += 1

    def load_text_images(self):
        if self.args.train_multi_style:
            return [self.get_text_img("Content"),
                    self.get_text_img("Recon"),
                    self.get_text_img("Style"),
                    self.get_text_img("Style"),
                    self.get_text_img("Styled")]
        else:
            return [self.get_text_img("Content"),
                    self.get_text_img("Recon"),
                    self.get_text_img("Style"),
                    self.get_text_img("Styled")]

    def get_text_img(self, text):
        width = self.args.crop_size
        height = self.args.crop_size

        image = Image.new(mode="RGB", size=(width, height), color="white")
        canvas = ImageDraw.Draw(image)
        font = ImageFont.truetype('Lato-Bold.ttf', size=20)
        canvas.text((int(width/4), int(height/4)), text, font=font, fill=(0, 0, 0))
        image = transforms.ToTensor()(image).unsqueeze(0)
        if self.args.d_id != -1: image = image.cuda()
        return image

    def save_snapshots(self, iter_counter):
        for i, (text_img, img) in enumerate(zip(self.text_images, self.snapshots)):
            self.snapshots[i] = get_grid_img(torch.cat((text_img, self.denormalize(img)), dim=0))

        images = Image.fromarray(np.concatenate(self.snapshots, axis=0))
        images.save(osp.join(self.model_checkpoints_folder, "snapshots", str(iter_counter) + "_snapshot.jpg"))

class Generator(nn.Module):
    def __init__(self, args, config):
        super(Generator, self).__init__()
        self.args = args
        self.config = config
        self.mode = ""
        if self.args.coco_funit:
            self.G = COCOFUNITTranslator(**self.config.coco)
            if self.args.vgg:
                self.G.content_encoder = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())[:-7])
                self.G.decoder = Decoder(**self.config.vgg_decoder)
                self.G.mlp_content = MLP(**self.config.vgg_coco_mlp_content)
        else:
            self.G = FUNITTranslator(**self.config.funit)
            if self.args.vgg:
                self.G.content_encoder = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())[:-7])
                self.G.decoder = Decoder(**self.config.vgg_decoder)

        self.G = init_model(self.G, self.args.init_G)
        self.G.train()

    def set_mode(self, value):
        self.mode = value

    def set_gradient(self, mode):
        for param in self.parameters():
            param.requires_grad = mode
        if self.args.vgg and mode == True:
            for param in self.G.content_encoder.parameters():
                param.requires_grad = False

    def get_style_feature(self, style_image_1, style_image_2):
        style_feat = self.G.style_encoder(style_image_1)
        if self.args.train_multi_style:
            ones_like = torch.ones_like(style_feat)
            style_feat = ones_like * 0.5 * style_feat
            style_feat += ones_like * 0.5 * self.G.style_encoder(style_image_2)
        return style_feat


    def forward(self, content_data, style_data):
        content_image = content_data
        style_image_1, style_label_1, style_image_2, _ = style_data
        styled_image = self.G.decode(self.G.content_encoder(content_image),
                                     self.get_style_feature(style_image_1, style_image_2))
        recon_image = self.G.forward(content_image)
        gen_obj = dict(content_image=content_image,
                       style_image_1=style_image_1,
                       style_label_1=style_label_1,
                       styled_image=styled_image,
                       recon_image=recon_image)

        if self.args.train_multi_style:
            snapshots = [content_image, recon_image, style_image_1, style_image_2, styled_image]
        else:
            snapshots = [content_image, recon_image, style_image_1, styled_image]

        return gen_obj, snapshots

class Discriminator(nn.Module):
    def __init__(self, args, config):
        super(Discriminator, self).__init__()
        self.args = args
        self.config = config
        self.model_checkpoints_folder = osp.join("train", self.args.exp)
        self.D = ResDiscriminator(**self.config.dis_model)
        self.D = init_model(self.D, self.args.init_D)
        self.D.train()

    def set_gradient(self, mode):
        for param in self.parameters():
            param.requires_grad = mode

    def forward(self, gen_obj, style_data):
        style_image_1, style_label_1, style_image_2, style_label_2 = style_data
        fake_disc_out, fake_disc_feat = self.D.forward(gen_obj['styled_image'], style_label_1)
        true_disc_out, true_disc_feat = self.D.forward(style_image_1, style_label_1)
        dis_obj = dict(fake_disc_out=fake_disc_out,
                       fake_disc_feat=fake_disc_feat,
                       true_disc_out=true_disc_out,
                       true_disc_feat=true_disc_feat)
        return dis_obj