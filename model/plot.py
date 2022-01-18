import pandas as pd
import math
import io
from os import path as osp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class Plot():
    def __init__(self, args, num_classes):
        self.args = args
        self.num_classes = num_classes
        self.model_checkpoints_folder = osp.join("train", self.args.exp)

    def generate_plots(self, logs):
        self.get_plots(logs)
        self.plot_logits(logs)
        self.plot_logit_distance(logs)
        self.plot_sifid_scores(logs)

    def get_moving_avg_of_list(self, data):
        numbers_series = pd.Series(data)
        windows = numbers_series.rolling(self.args.moving_avg)
        moving_averages = windows.mean()
        moving_averages_list = moving_averages.tolist()
        moving_averages_list = moving_averages_list[self.args.moving_avg - 1:]
        if len(moving_averages_list) != 0:
            return moving_averages_list
        else:
            return data

    def get_concat_h(self, im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def get_concat_v(self, im1, im2):
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    def get_plot_grid(self, image_list):
        num = len(image_list)
        row = int(math.sqrt(num))
        target_size = image_list[0].size
        target = None
        res = None

        for i, img in enumerate(image_list):
            img = img.resize(target_size)
            if target == None:
                target = img
            else:
                target = self.get_concat_h(target, img)
            if (i+1) % row == 0 and i != 0:
                if res == None:
                    res = target
                else:
                    res = self.get_concat_v(res, target)
                target = None

        if target != None:
            empty_img = Image.fromarray(np.zeros_like(np.asarray(img)))
            while i % row != 0:
                target = self.get_concat_h(target, empty_img)
                i += 1
            res = self.get_concat_v(res, target)
        return res

    def plot_logits(self, logs):
        for key, _ in logs["logits"].items():
            check = False
            plt_list = []
            for label in range(self.num_classes):
                fake_logit = logs["logits"][key][label]["fake"]
                if len(fake_logit) > 0:
                    #yaxis = self.get_moving_avg_of_list(fake_logit)
                    yaxis = fake_logit
                    xaxis = list(range(len(yaxis)))
                    plt.plot(xaxis, yaxis, label="fake")
                    check = True

                if key == "dis":
                    true_logit = logs["logits"][key][label]["true"]
                    if len(true_logit) > 0:
                        # yaxis = self.get_moving_avg_of_list(true_logit)
                        yaxis = true_logit
                        xaxis = list(range(len(yaxis)))
                        plt.plot(xaxis, yaxis, label="true")
                        check = True

                if check == True:
                    plt.xlabel("Moving average")
                    plt.ylabel("logit")
                    plt.title("Label: " + str(label))
                    plt.legend()
                    plt.grid(True)
                    img_buf = io.BytesIO()
                    plt.savefig(img_buf, format='jpg')
                    plt_list.append(Image.open(img_buf))
                    plt.close()
                    check = False
            if len(plt_list) == 1:
                grid_plot = plt_list[0]
            elif len(plt_list) > 1:
                grid_plot = self.get_plot_grid(plt_list)
            grid_plot.save(osp.join(self.model_checkpoints_folder, "plots", str(key) + "_logits.png"))
            img_buf.close()

    def plot_logit_distance(self, logs):
        include_num = 100
        plt_list = []
        for label in range(self.num_classes):
            fake_logits = logs["logits"]["dis"][label]["fake"][-include_num:]
            true_logits = logs["logits"]["dis"][label]["true"][-include_num:]
            if len(fake_logits) > 0:
                xaxis = list(range(min(include_num, len(fake_logits))))
                yaxis = []
                for t_logit, f_logit in zip(true_logits, fake_logits):
                    yaxis.append(np.absolute(t_logit - f_logit))
                plt.plot(xaxis, yaxis, label=yaxis[-1])
                plt.xlabel("Datapoints")
                plt.ylabel("logit distance")
                plt.title("Label: " + str(label))
                plt.grid(True)
                plt.legend()
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='jpg')
                plt_list.append(Image.open(img_buf))
                plt.close()

        if len(plt_list) == 1:
            grid_plot = plt_list[0]
        elif len(plt_list) > 1:
            grid_plot = self.get_plot_grid(plt_list)
        grid_plot.save(osp.join(self.model_checkpoints_folder, "plots", "logit_distance.png"))
        img_buf.close()

    def get_plots(self, logs):
        for key, headline in logs["headline"].items():
            if len(logs["store"][key]) > 0:
                losses = logs["store"][key]
                yaxis = self.get_moving_avg_of_list(losses)
                xaxis = list(range(len(yaxis)))
                plt.plot(xaxis, yaxis, label=losses[-1])
                plt.xlabel("Datapoints with moving average during validation")
                plt.ylabel("loss")
                plt.title(headline)
                plt.grid(True)
                plt.legend()
                plt.savefig(osp.join(self.model_checkpoints_folder, "plots", key + ".png"))
                plt.close()

    def plot_sifid_scores(self, logs):
        yaxis = logs["sifid_scores"]
        xaxis = list(range(len(yaxis)))
        plt.plot(xaxis, yaxis, label=yaxis[-1])
        plt.title("SIFID score over time")
        plt.xlabel("Validation steps")
        plt.ylabel("fid_score")
        plt.grid(True)
        plt.savefig(osp.join(self.model_checkpoints_folder, "plots", "fid_score.jpg"), format='jpg')
        plt.close()

