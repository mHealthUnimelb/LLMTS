class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 2
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 4
        # self.num_classes = 3
        self.dropout = 0.35
        self.features_len = 315#190#315 #18  #315 is for window_size = 2500 

        # training configs
        self.num_epoch = 40#200 #150 #40 #80 #40

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4#0.001#3e-4

        # beta distribution parameter
        self.alpha = 2
        self.beta = 8

        # data parameters
        self.drop_last = True
        self.batch_size =  256 #128 #256 #256 #128 # how about change to 256.. to run 128 for only temporl loss..

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

        self.TSlength_aligned = 2500


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        # self.jitter_ratio = 0.05
        self.jitter_ratio = 0.8
        self.max_seg = 12 #20


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 64#100
        self.timesteps = 10
