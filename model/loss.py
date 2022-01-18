from os import path as osp
import numpy as np
import pickle
import random

import torch
import torch.nn as nn

from model.plot import Plot
import sifid.sifid

from imaginaire.losses.gan import GANLoss

class Loss():
    def __init__(self, args, num_classes):
        self.args = args
        self.num_classes = num_classes
        self.model_checkpoints_folder = osp.join("train", self.args.exp)
        self.plot = Plot(args, num_classes)
        self.gan_loss = GANLoss(self.args.gan_loss_method)
        self.mode = ""
        self.logs = self.load_logs()

    def set_mode(self, value):
        self.mode = value

    def load_logs(self):
        init_log = self.load_init_logging_file()
        if self.args.load_checkpoint:
            try:
                logs_old = pickle.load(open(osp.join(self.model_checkpoints_folder, "objects", "logs.obj"), 'rb'))
                if logs_old.keys() == init_log.keys():
                    print("loaded logs successfully")
                    return logs_old
                else:
                    missing_keys = list(set(init_log) - set(logs_old))
                    for key in missing_keys:
                        logs_old[key] = init_log[key]
                    print("Added the following keys to the log dict, because they were missing: " + ''.join(missing_keys))
                    return logs_old
            except:
                raise("Not able to load log data, use init dict")
        else:
            return init_log

    def load_init_logging_file(self):
        return dict(store=dict(recon_loss=[],
                               fm_loss=[],
                               gan_gen_loss=[],
                               gan_dis_loss=[],
                               cur_loss=[],
                               dis_win_rate=[]),
                    logits=dict(dis=dict(zip(range(self.num_classes),
                                             [dict(fake=[], true=[]) for _ in range(self.num_classes)])),
                                gen=dict(zip(range(self.num_classes),
                                             [dict(fake=[]) for _ in range(self.num_classes)]))),
                    sifid_scores=[],
                    headline=dict(recon_loss="Pixel Reconstruction Loss",
                                  fm_loss="Feature Matching Loss",
                                  gan_gen_loss="GAN Loss within Generator",
                                  gan_dis_loss="GAN Loss within Discriminator",
                                  cur_loss="Global Loss",
                                  dis_win_rate="Winning rate of the discriminator"))

    def save_logs(self):
        pickle.dump(self.logs, open(osp.join(self.model_checkpoints_folder, "objects", "logs.obj"), 'wb'))

    def write_log(self, log, value):
        self.logs["store"][log].append(np.float(value.detach()))

    def log_loss(self, log, value):
        if self.mode == "val":
            self.write_log(log, value)

    def log_logit(self, label, mode, key, logit):
        logit = np.float(logit.mean(3).mean(2).squeeze(1).detach())
        self.logs["logits"][mode][label][key].append(logit)

    def log_logits(self, label, mode, key, logits):
        if self.mode == "val":
            if logits.size()[0] == 1:
                self.log_logit(label[0], mode, key, logits)
            else:
                for i, logit in enumerate(logits):
                    self.log_logit(label[i], mode, key, logit.unsqueeze(0))

    def get_sifid_score(self):
        path = osp.join(self.model_checkpoints_folder, "plots", "sifid_images")
        score = sifid.sifid.calculate_sifid_given_paths(osp.join(path, "true"), osp.join(path, "false"),
                                                        1, self.args.d_id, 64, "jpg")
        self.logs["sifid_scores"].append(np.mean(score))

    def recon_loss(self):
        loss = nn.L1Loss()(self.gen_obj['content_image'], self.gen_obj['recon_image'])
        loss *= self.args.lrp
        self.log_loss("recon_loss", loss)
        return loss

    def feature_matching_loss(self):
        loss = nn.L1Loss()(self.dis_obj['fake_disc_feat'], self.dis_obj['true_disc_feat'])
        loss *= self.args.lfm
        self.log_loss("fm_loss", loss)
        return loss

    def gan_loss_G(self):
        loss = self.gan_loss(self.dis_obj['fake_disc_out'], True, dis_update=False)
        loss *= self.args.lg
        self.log_loss("gan_gen_loss", loss)
        self.log_logits(self.gen_obj['style_label_1'].cpu().numpy(), "gen", "fake", self.dis_obj['fake_disc_out'])
        return loss

    def rnd_flip_label(self):
        if random.random() < self.args.D_noise:
            temp = self.dis_obj['fake_disc_out']
            self.dis_obj['fake_disc_out'] = self.dis_obj['true_disc_out']
            self.dis_obj['true_disc_out'] = temp

    def gan_loss_D(self):
        if self.args.D_noise > 0:
            self.rnd_flip_label()
        fake_loss = self.gan_loss(self.dis_obj['fake_disc_out'], False, dis_update=True)
        true_loss = self.gan_loss(self.dis_obj['true_disc_out'], True, dis_update=True)

        loss = (true_loss + fake_loss) * self.args.lg
        self.log_loss("gan_dis_loss", loss)
        self.log_logits(self.gen_obj['style_label_1'].cpu().numpy(), "dis", "true", self.dis_obj['true_disc_out'])
        self.log_logits(self.gen_obj['style_label_1'].cpu().numpy(), "dis", "fake", self.dis_obj['fake_disc_out'])
        return loss

    def calc_true_accuracy(self, output):
        return torch.mean(torch.greater(output, torch.zeros_like(output)).float())

    def calc_fake_accuracy(self, output):
        return torch.mean(torch.less(output, torch.zeros_like(output)).float())

    def calc_accuracy_G(self):
        return self.calc_true_accuracy(self.dis_obj['fake_disc_out']) + 1e-6

    def calc_accuracy_D(self):
        return self.calc_fake_accuracy(self.dis_obj['fake_disc_out']) * 0.5 + \
               self.calc_true_accuracy(self.dis_obj['true_disc_out']) * 0.5

    def compute_G_loss(self, gen_obj, dis_obj):
        self.gen_obj = gen_obj
        self.dis_obj = dis_obj
        loss = self.recon_loss()
        loss += self.feature_matching_loss()
        loss += self.gan_loss_G()
        accuracy = self.calc_accuracy_G()
        return loss, accuracy

    def compute_D_loss(self, gen_obj, dis_obj):
        self.gen_obj = gen_obj
        self.dis_obj = dis_obj
        loss = self.gan_loss_D()
        accuracy = self.calc_accuracy_D()
        return loss, accuracy

    def get_plots(self):
        self.get_sifid_score()
        self.save_logs()
        self.plot.generate_plots(self.logs)