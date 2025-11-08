"""
Written by George Zerveas

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""

# import os
# os.environ['TRANSFORMERS_CACHE'] = '/code/huggingface/hub'
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")
import os
import sys
import time
import pickle
import json

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Project modules
from options import Options
from running_fsca import setup, pipeline_factory, validate, test, check_progress, NEG_METRICS
from utils import utils
from datasets.data_fsca import data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.ts_transformer import model_factory
from models.gpt4ts import gpt4ts
from models.GNNLLM_fsca_mTAN import GNNLLM_fsca
from models.loss import get_loss_module
from optimizers import get_optimizer

import wandb

torch.set_num_threads(1)    

def main(config):

    total_epoch_time = 0
    total_eval_time = 0

    total_start_time = time.time()

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Build data
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config['data']]
    all_data = data_class(args=config, dataset=config['data'], device=torch.device("cpu"), q=config['quantization'], upsampling_batch=False)

    if config['training_flag']:
        # training
        train_loader = all_data.data_objects["train_dataloader"]
        val_loader = all_data.data_objects["val_dataloader"]

    test_loader = all_data.data_objects["test_dataloader"]


    config['label_values'] = list(range(len(all_data.class_names)))

    # Create model
    logger.info("Creating model ...")
    if config['training_flag']:
        model = GNNLLM_fsca(config, all_data)
    else:
        model = GNNLLM_fsca(config, all_data)

    if config['freeze']:
        for name, param in model.named_parameters():
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))


    # Initialize optimizer

    if config['global_reg']:
        weight_decay = config['l2_reg']
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])
    optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`
    lr = config['lr']  # current learning step
    # Load model and optimizer state
    if args.load_model:
        if config['test_only'] == 'testset':
            model = utils.load_model(model, model_path=config['load_model'], optimizer=None, resume=False,
                                                         change_output=False,
                                                         lr=None,
                                                         lr_step=None,
                                                         lr_factor=None)
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_factor'])
    model.to(device)

    loss_module = get_loss_module(config)

    if config['test_only'] == 'testset':  # Only evaluate and skip training
        if args.load_model:
            model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_factor'])
        model.to(device)

        _, collate_fn, runner_class = pipeline_factory(config)

        test_evaluator = runner_class(model, test_loader, device, loss_module,
                                            print_interval=config['print_interval'], console=config['console'])

        aggr_metrics_test, per_batch_test = test(test_evaluator, config=config)
        return
    
    # Initialize data generators
    _, collate_fn, runner_class = pipeline_factory(config)

    trainer = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                                 print_interval=config['print_interval'], console=config['console'])
    val_evaluator = runner_class(model, val_loader, device, loss_module,
                                       print_interval=config['print_interval'], console=config['console'])
    test_evaluator = runner_class(model, test_loader, device, loss_module,
                                        print_interval=config['print_interval'], console=config['console'])

    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])

    best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf depending on key metric
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}

    # Evaluate on validation before training
    aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                          best_value, epoch=0)
    metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    metrics.append(list(metrics_values))

    logger.info('Starting training...')
    for epoch in tqdm(range(start_epoch + 1, config["epochs"] + 1), desc='Training Epoch', leave=False):
        mark = epoch if config['save_all'] else 'last'
        epoch_start_time = time.time()
        aggr_metrics_train = trainer.train_epoch(epoch, config)  # dictionary of aggregate epoch metrics
        epoch_runtime = time.time() - epoch_start_time
        print()
        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
            if k in ['loss', 'accuracy', 'precision']:
                if k == 'loss':
                    k = 'train loss'
                wandb.log({k: v, "epoch": epoch})
        
        logger.info(print_str)
        logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(epoch_runtime)))

        wandb.log({"epoch_runtime": epoch_runtime, "epoch": epoch})

        total_epoch_time += epoch_runtime
        avg_epoch_time = total_epoch_time / (epoch - start_epoch)
        avg_batch_time = avg_epoch_time / len(train_loader)
        logger.info("Avg epoch train. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_epoch_time)))
        logger.info("Avg batch train. time: {} seconds".format(avg_batch_time))

        # evaluate if first or last epoch or at specified interval
        if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0):
            aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config,
                                                                  best_metrics, best_value, epoch)
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))

            # test
            aggr_metrics_test, per_batch_test = test(test_evaluator, config=config)

        # Learning rate scheduling
        if epoch == config['lr_step'][lr_step]:
            lr = lr * config['lr_factor'][lr_step]
            if lr_step < len(config['lr_step']) - 1:  # so that this index does not get out of bounds
                lr_step += 1
            logger.info('Learning rate updated to: ', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Difficulty scheduling
        if config['harden'] and check_progress(epoch):
            train_loader.dataset.update()
            val_loader.dataset.update()

    # Export evolution of metrics over epochs
    header = metrics_names
    metrics_filepath = os.path.join(config["output_dir"], "metrics_" + config["experiment_name"] + ".xls")
    book = utils.export_performance_metrics(metrics_filepath, metrics, header, sheet_name="metrics")

    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(config["records_file"], config["initial_timestamp"], config["experiment_name"],
                          best_metrics, aggr_metrics_val, comment=config['comment'])

    logger.info('Best {} was {}. Other metrics: {}'.format(config['key_metric'], best_value, best_metrics))
    wandb.log({"BestACC": best_value})
    logger.info('All Done!')

    res_path = os.path.join(config['output_dir'], 'res.txt')
    formatted_string = 'Best {} was {}. Other metrics: {}'.format(config['key_metric'], best_value, best_metrics)
    with open(res_path, 'w') as file:
        file.write(formatted_string)

    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))

    return best_value


if __name__ == '__main__':

    args = Options().parse()  # `argsparse` object

    parts = args.layer_index.split('*')

    args.gpt_layers = int(parts[0]) if parts[0] else None

    if len(parts) > 1 and parts[1]:
        args.gnn_layer_index = [int(x) for x in parts[1].split('_')]
        args.gnn_layer_index_str = '_'.join(str(x) for x in args.gnn_layer_index)
    else:
        args.gnn_layer_index = []
        args.gnn_layer_index_str = ""

    if len(parts) > 2 and parts[2]:
        args.l_gnn_layer_index = [int(x) for x in parts[2].split('_')]
        args.l_gnn_layer_index_str = '_'.join(str(x) for x in args.l_gnn_layer_index)
    else:
        args.l_gnn_layer_index = []
        args.l_gnn_layer_index_str = ""
    
    # patch_size_stride
    args.patch_size, args.stride = [int(x) for x in args.patch_size_stride.split()]

    # wandb
    group_name = '{}_ps{}_{}_gl{}_{}*{}_b{}_l{}_e{}_wv{}_wf{}_dl{}_df{}_s{}_d{}'.format(
        args.model, 
        args.patch_size,
        args.stride,
        args.gpt_layers,
        args.gnn_layer_index_str,

        args.l_gnn_layer_index_str,
        args.batch_size,
        args.lr,
        args.epochs,
        args.w_l2s_v,

        args.w_l2s_flag,
        args.d_l_comp,
        args.d_ff,
        args.seed,
        args.dropout)

    setting = '{}'.format(args.experiment_name)

    wandb_group_name = f'{group_name}'

    wandb_run_name = f'{setting}'

    args.data_dir = './datasets/' + args.experiment_name

    ### wandb
    if args.wandb_flag:
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"{args.wd_project}", 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=wandb_run_name,
            group=wandb_group_name,
            config=args) 
    else:
        run = wandb.init(mode="disabled")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Dir {args.output_dir} created")
    else:
        print(f"Dir {args.output_dir} has existed")

    config = setup(args)  # configuration dictionary
    main(config)
