import os.path as osp
import tqdm

import torch
from torch.utils.data import DataLoader

from model.loss import Loss

from imaginaire.utils.model_average import ModelAverage

class Training(object):
    def __init__(self, args, network, style_dataset, content_dataset):
        self.args = args
        self.discr_success = 0.5
        self.dis_win_function = lambda dis_accuracy: self.discr_success * (1. - 0.01) + 0.01 * dis_accuracy

        self.content_dataset = content_dataset
        self.style_dataset = style_dataset
        self.val_content_dataset = self.content_dataset.get_validation_dataset()
        self.val_style_dataset = self.style_dataset.get_validation_dataset()

        self.val_content_dataloader = DataLoader(self.val_content_dataset, self.args.batch_size, False)
        self.val_style_dataloader = DataLoader(self.val_style_dataset, self.args.batch_size, False)

        self.loss = Loss(args, self.style_dataset.num_labels)
        self.network = network
        if self.args.model_avg:
            self.network.G = ModelAverage(self.network.G, 0.999, 1000, True)
            self.optimizer_G = self.get_optimizer(self.network.G.module.parameters(), self.args.lr_G,
                                                  self.args.beta1_G, self.args.beta2_G, self.args.eps_G)
        else: self.optimizer_G = self.get_optimizer(self.network.G.parameters(), self.args.lr_G,
                                                    self.args.beta1_G, self.args.beta2_G, self.args.eps_G)
        self.optimizer_D = self.get_optimizer(self.network.D.parameters(), self.args.lr_D, self.args.beta1_D, self.args.beta2_D, self.args.eps_D)
        self.scheduler_G = self.get_schedular(self.optimizer_G)
        self.scheduler_D = self.get_schedular(self.optimizer_D)

        self.iter_counter = 1
        if self.args.load_checkpoint: self.load_checkpoint_state()

    def get_schedular(self, optim):
        return torch.optim.lr_scheduler.StepLR(optim, step_size=self.args.scheduler_stepsize, gamma=0.5)

    def get_optimizer(self, params, lr, beta1, beta2, eps):
        return torch.optim.Adam(params=params, lr=lr, betas=(beta1, beta2), eps=eps)

    def load_checkpoint_state(self):
        try:
            path = osp.join("train", self.args.exp, "checkpoint.pth")
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
            try:
                self.network.G.load_state_dict(checkpoint['net_G'])
                self.network.D.load_state_dict(checkpoint['net_D'])
            except:
                try:
                    checkpoint_weights = list(checkpoint['net_D'].items())
                    state_dict = self.network.D.state_dict()
                    for i, key in enumerate(state_dict): state_dict[key] = checkpoint_weights[i][1]
                    self.network.D.load_state_dict(state_dict)

                    checkpoint_weights = list(checkpoint['net_G'].items())
                    state_dict = self.network.G.state_dict()
                    for i, key in enumerate(state_dict): state_dict[key] = checkpoint_weights[i][1]
                    self.network.G.load_state_dict(state_dict)
                except:
                    raise ("Not able to load data to state dict")

            self.optimizer_G.load_state_dict(checkpoint['opt_G'])
            self.optimizer_D.load_state_dict(checkpoint['opt_D'])
            self.scheduler_G.load_state_dict(checkpoint['sch_G'])
            self.scheduler_D.load_state_dict(checkpoint['sch_D'])
            self.iter_counter = checkpoint['current_iteration']
            print("successfully loaded checkpoint to network")
        except:
            raise("loading model failed")

    def save_checkpoint(self):
        torch.save(
            {'net_G': self.network.G.state_dict(),
             'net_D': self.network.D.state_dict(),
             'opt_G': self.optimizer_G.state_dict(),
             'opt_D': self.optimizer_D.state_dict(),
             'sch_G': self.scheduler_G.state_dict(),
             'sch_D': self.scheduler_D.state_dict(),
             'current_iteration': self.iter_counter},
              osp.join("train", self.args.exp, "checkpoint.pth"))

    def set_mode(self, value):
        self.loss.mode = value
        self.network.mode = value
        if self.args.model_avg: self.network.G.module.mode = value
        else: self.network.G.mode = value

    def compute_gradient_for(self, G, D):
        if self.args.model_avg: self.network.G.module.set_gradient(G)
        else: self.network.G.set_gradient(G)
        self.network.D.set_gradient(D)

    def G_update(self, content_data, style_data):
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        self.compute_gradient_for(G=True, D=False)
        gen_obj, dis_obj = self.network.gen_forward(content_data, style_data)
        loss, accuracy = self.loss.compute_G_loss(gen_obj, dis_obj)
        loss.backward()
        self.optimizer_G.step()
        if self.args.model_avg: self.network.G.update_average()

    def D_update(self, content_data, style_data):
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        self.compute_gradient_for(G=False, D=True)
        gen_obj, dis_obj = self.network.dis_forward(content_data, style_data)
        loss, accuracy = self.loss.compute_D_loss(gen_obj, dis_obj)
        loss.backward()
        self.optimizer_D.step()
        self.discr_success = self.dis_win_function(accuracy)
        self.loss.write_log("dis_win_rate", self.discr_success)

    def end_of_iteration(self):
        if self.args.scheduler_stepsize != -1:
            self.scheduler_G.step()
            self.scheduler_D.step()
        if self.iter_counter % self.args.save_img == 0: self.network.save_snapshots(self.iter_counter)
        if self.iter_counter % self.args.val_step == 0:
            self.save_checkpoint()
            self.validate()
            self.loss.get_plots()

    def validate(self):
        self.set_mode("val")
        self.network.sifid_counter = 0
        for content_data, style_data in zip(self.val_content_dataloader, self.val_style_dataloader):
            with torch.no_grad():
                gen_loss, _ = self.loss.compute_G_loss(*self.network.gen_forward(content_data, style_data))
                dis_loss, _ = self.loss.compute_D_loss(*self.network.dis_forward(content_data, style_data))
            self.loss.log_loss("cur_loss", gen_loss + dis_loss)
        self.set_mode("train")

    def train(self):
        self.set_mode("train")
        for self.iter_counter in tqdm.tqdm(range(self.iter_counter, self.args.max_steps + 1)):
            content_data = self.content_dataset.get_batch()
            style_data = self.style_dataset.get_batch()
            for _ in range(self.args.G_step): self.G_update(content_data, style_data)
            for _ in range(self.args.D_step): self.D_update(content_data, style_data)
            self.end_of_iteration()
        if self.args.d_id != -1: torch.cuda.empty_cache()