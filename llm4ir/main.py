import torch

import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate,model_evaluate_simclr
from models.TC import TC
from utils import _calc_metrics, copy_Files
from models.model import base_Model
from models_ns.model import base_Model_simclr
# Args selections
start_time = datetime.now()


parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Etstcc', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='Epilepsy', type=str,
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--ana_ratio', default=1, type=float,
                    help='change the learning rate')
parser.add_argument('--method', default='ts-tcc', type=str,
                    help='Experiment Description')
args = parser.parse_args()


with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('We are using %s now.' %device)

# device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0

ana_ratio = args.ana_ratio

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug(f'Medf:    {configs.augmentation.max_seg}')
logger.debug(f'lr:    {configs.lr}')
logger.debug(f'Ana_ratio:   {ana_ratio}')
logger.debug("=" * 45)

# Load datasets
data_path = f"./data/{data_type}"
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode, ana_ratio)
logger.debug("Data loaded ...")

# Load Model
if args.method == 'TS-TCC':
    model = base_Model(configs).to(device)
    temporal_contr_model = TC(configs, device).to(device)
    logger.debug("TS-TCC loaded ...")
else:
    model = base_Model_simclr(configs).to(device)
    logger.debug("SimCLR loaded ...")


## load the fine-tuned model 

### change to your path

if args.method == 'TS-TCC':
    checkpoint = torch.load('./experiments_logs/Etstcc/run_1/train_linear_seed_0/saved_models_tstcc/ckp_last.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    temporal_contr_model.load_state_dict(checkpoint['temporal_contr_model_state_dict'])
    logger.debug("TS-TCC Model loaded ...")   
    model.eval()
    temporal_contr_model.eval()
else:
    checkpoint = torch.load('./experiments_logs/Etstcc/run_1/train_linear_seed_0/saved_models_simclr/ckp_last.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.debug("SimCLR Model loaded ...")
    model.eval()



logger.debug('\nEvaluate on the Test set:')
    # Run evaluation
if args.method == 'TS-TCC':
    test_loss, test_acc, _, _, auprc = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
    print(f'Test loss: {test_loss:0.4f} | Test Accuracy: {test_acc:0.4f} | AUPRC: {auprc:0.4f}')
else:
    test_loss, test_acc, _, _, auprc = model_evaluate_simclr(model, configs, test_dl, device, training_mode)
    print(f'Test loss: {test_loss:0.4f} | Test Accuracy: {test_acc:0.4f} | AUPRC: {auprc:0.4f}')



