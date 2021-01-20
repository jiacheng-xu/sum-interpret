# Set up base env and utils

import argparse
import logging
import os
import pickle
import random
import statistics
import sys
from datetime import datetime
from typing import Dict, List
import multiprocessing
import torch
from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartModel, BartTokenizer
import numpy as np
import pandas as pd

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

kld = torch.nn.KLDivLoss(log_target=True, reduction='none')

now = datetime.now()

logger = logging.getLogger('sum')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(f"{now.strftime('%m')}{now.strftime('%d')}.log")
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('<br>%(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)


def load_pickle(dir, fname) -> Dict:
    with open(os.path.join(dir, fname), 'rb') as rfd:
        data = pickle.load(rfd)
    return data


def pnum(num):
    return "{:.4f}".format(num)


def add_dataname_to_suffix(args, args_dir) -> str:
    out = f"{args_dir}_{args.data_name}"
    out = add_temp_to_suffix(args, out)
    if not os.path.exists(out):
        os.makedirs(out)
    return out

def add_temp_to_suffix(args, args_dir) -> str:
    out = f"{args_dir}_{args.temp}"
    # if not os.path.exists(out):
        # os.makedirs(out)
    return out

def dec_print_wrap(func):
    def wrapper(*args, **kwargs):
        logging.info("=" * 20)
        out = func(*args, **kwargs)
        logging.info("-" * 20)
        return out
    return wrapper


def read_meta_data(dir, fname):
    file_package = load_pickle(dir, fname)
    data: List = file_package['data']
    meta = file_package['meta']
    return data, meta


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def common_args():

    task_choice = ['inp_grad', 'int_grad', 'random', 'occ', 'lime', 'lead']
    eval_mode = ['sel_tok', 'rm_tok', 'sel_sent', 'rm_sent']
    settings = ['ctx', 'ctx-novel', 'ctx-fusion', 'lm', 'ctx-hard']

    parser = argparse.ArgumentParser()
    parser.add_argument("-model_family", default='bart')
    parser.add_argument("-data_name", default='xsum', help='name of dataset')
    parser.add_argument("-mname_lm", default='facebook/bart-large')
    parser.add_argument("-mname_sum", default='facebook/bart-large-xsum')
    parser.add_argument('-truncate_sent', default=15,
                        help='the max sent used for perturbation')
    parser.add_argument('-truncate_word', default=70,
                        help='the max token in each single sentence')
    parser.add_argument(
        '-dir_meta', default="/mnt/data0/jcxu/meta_pred", help="The location to meta data.")
    parser.add_argument('-dir_base', default="/mnt/data0/jcxu/output_base")
    parser.add_argument('-dir_stat', default="/mnt/data0/jcxu/csv")

    parser.add_argument("--debug", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate debug mode.")
    parser.add_argument("-task", dest='task', choices=task_choice)
    parser.add_argument("-device", help="device to use", default='cuda:0')
    parser.add_argument('-max_example', default=5000,
                        help='The max number of examples (documents) to look at.')
    parser.add_argument('-num_run_cut', default=40)
    parser.add_argument('-batch_size', default=400,type=int)
    parser.add_argument('-eval_mode', dest='eval_mode', choices=eval_mode)
    parser.add_argument('-temp',type=float,default=0.5)
    return parser


def fix_args(args):

    args.dir_base = add_dataname_to_suffix(args, args.dir_base)
    args.dir_meta = add_dataname_to_suffix(args, args.dir_meta)
    args.dir_stat = add_dataname_to_suffix(args, args.dir_stat)
    if args.debug:
        args.device = 'cpu'
        args.mname_sum = 'sshleifer/distilbart-xsum-6-6'
    if hasattr(args, 'task'):
        args.dir_task = f"/mnt/data0/jcxu/task_{args.task}"
        args.dir_task = add_dataname_to_suffix(args, args.dir_task)
        args.dir_eval_save = f"/mnt/data0/jcxu/eval_{args.task}_{args.eval_mode}"
        args.dir_eval_save = add_dataname_to_suffix(args,args.dir_eval_save)
    return args


random.seed(2021)
