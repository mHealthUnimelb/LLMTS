import os
import json
import torch

class Config:
    def __init__(self, args):
        # Basic settings
        self.seed = args.seed

        # Training parameters
        # self.epochs = 30
        self.batch_size = args.batch_size
        # self.learning_rate = 1e-4
        # self.weight_decay = 1e-5
        
        # Model parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        # self.device = torch.device("cpu")
        # self.hid_dim = 128  # Hidden dimension
        self.mask_ratio = 0.5  # Ratio of patches to mask
        # self.patch_size = 16   # Number of patches
        # self.num_heads = 1  # Number of attention heads
        # self.embed_time = 16  # Embedding dimension
        
        # Dataset parameters
        self.dataset_name = "physionet"  # physionet,mimic, activity
        self.quantization = 0.0  # Quantization for the dataset
        self.n = 10000000  # Number of samples
        self.percentage_tp_to_sample = 0.1  # Percentage of time points to sample
        # self.patience = 5  # Early stopping patienc
        self.interp_lr = 0.0001

        ## random_mask
        self.mask_ori = False
        self.max_len = 512
        self.mask_ratio_per_seg = 0.15
        self.segment_num = 3
        self.latent_mask_ratio = 0.5

        # Downstream 
        self.masking_mode = "end"  # Options: "end", "beginning", "middle", "random_blocks", "random"
        self.num_blocks = 2  # Only used for "random_blocks" mode
        self.history = 24 ## for physionet/mimic, 24, for physionet 3000
        self.pred_window = 1000
        

        
    def load(self, config_path):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        for key, value in config_dict.items():
            if key == 'device':
                self.device = torch.device(value)
            else:
                setattr(self, key, value)
                
    def save(self, config_path):
        """Save configuration to JSON file"""
        config_dict = self.__dict__.copy()
        
        # Convert non-serializable objects
        if 'device' in config_dict:
            config_dict['device'] = str(config_dict['device'])
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)