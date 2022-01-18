import os.path as osp
import yaml

class Config():
    def __init__(self, args, num_classes):
        self.args = args
        self.num_classes = num_classes
        self.config = self.get_config()
        self.sanity_check()
        self.get_model_params()

    def sanity_check(self):
        assert not self.args.spectral_norm_G == True == self.args.model_avg

    def get_model_params(self):
        self.dis_model = dict(image_channels=self.config['image_channels'],
                              num_classes=self.config['num_classes'],
                              num_filters=self.config['dis_num_filters'],
                              max_num_filters=self.config['dis_max_num_filters'],
                              num_layers=self.config['dis_num_layers'],
                              padding_mode=self.config['dis_padding_mode'],
                              weight_norm_type=self.config['dis_weight_norm_type'])

        # FUNIT
        self.funit = dict(num_filters = self.config['num_filters'],
                          num_filters_mlp = self.config['num_filters_mlp'],
                          style_dims = self.config['style_dims'],
                          num_res_blocks = self.config['num_res_blocks'],
                          num_mlp_blocks = self.config['num_mlp_blocks'],
                          num_downsamples_style = self.config['num_downsamples_style'],
                          num_downsamples_content = self.config['num_downsamples_content'],
                          num_image_channels = self.config['image_channels'],
                          weight_norm_type = self.config['weight_norm_type'],
                          # ConvBlocks=self.config['ConvBlocks'],
                          nonlinearity=self.config['nonlinearity'])

        # COCO
        self.coco = dict(num_filters = self.config['num_filters'],
                         num_filters_mlp = self.config['num_filters_mlp'],
                         style_dims = self.config['style_dims'],
                         usb_dims = self.config['coco_usb_dims'],
                         num_res_blocks = self.config['num_res_blocks'],
                         num_mlp_blocks = self.config['num_mlp_blocks'],
                         num_downsamples_style = self.config['num_downsamples_style'],
                         num_downsamples_content = self.config['num_downsamples_content'],
                         num_image_channels = self.config['image_channels'],
                         # ConvBlocks=self.config['ConvBlocks'],
                         weight_norm_type = self.config['weight_norm_type']
                         )

        # decoder vgg coco/funit
        self.vgg_decoder = dict(num_enc_output_channels=self.config['vgg_dim'],
                                style_channels=self.config['num_filters_mlp'],
                                num_image_channels=self.config['image_channels'],
                                num_upsamples=self.config['num_downsamples_content'],
                                weight_norm_type=self.config['weight_norm_type'],
                                # ConvBlocks=self.config['ConvBlocks'],
                                nonlinearity=self.config['nonlinearity'])

        # mlp vgg coco
        # num_layers set in imaginaire file as variable
        self.vgg_coco_mlp_content = dict(input_dim=self.config['vgg_dim'],
                                         output_dim=self.config['style_dims'],
                                         latent_dim=self.config['num_filters_mlp'],
                                         num_layers=2,
                                         activation_norm_type=self.config['weight_norm_type'],
                                         nonlinearity=self.config['nonlinearity'])


    def get_config(self):
        config_file = self.load_config_file()
        return self.adjust_config_file(config_file)

    def adjust_config_file(self, config):
        if self.args.spectral_norm_G:
            config.update({'weight_norm_type': "spectral"})
        config.update({'num_classes': self.num_classes})
        return config

    def load_config_file(self):
        try:
            with open(osp.join("config", self.args.config_file), 'r') as f:
                return yaml.load(f, Loader=yaml.FullLoader)
        except:
            raise ("error while loading config file")