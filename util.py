# Set up base env and utils

import sys
import logging
from datetime import datetime

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
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

def dec_print_wrap(func):
    def wrapper():
        logging.info("=" * 20)
        func()
        logging.info("-" * 20)
    return wrapper

# Transformers
from transformers import BartForConditionalGeneration, BartTokenizer,BartModel
import torch

from datasets import load_dataset
import random
import statistics
import pickle
import argparse