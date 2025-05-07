import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import warnings

from data_provider.utils import get_data

warnings.filterwarnings('ignore')


class Dataset_P12(Dataset):
    def __init__(self, args=None, root_path=None, dataset='P12', device=torch.device("cpu"), q=0.016, upsampling_batch=False):
        self.data_objects = get_data(args, dataset, device, q, upsampling_batch)
        self.num_class = args.num_classes

class Dataset_MIMIC(Dataset):
    def __init__(self, args=None, root_path=None, dataset='MIMIC', device=torch.device("cpu"), q=0.016, upsampling_batch=False):
        self.data_objects = get_data(args, dataset, device, q, upsampling_batch)
        self.class_names = [0, 1]
        self.num_class = args.num_classes