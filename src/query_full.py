from typing import List
from statistics import quantiles
import pandas as pd
import csv
import os
import statistics
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import pyplot
from util import *
from helper import *

import nltk
from nltk.corpus import stopwords
stp = stopwords.words('english')

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
# colors = mcolors.TABLEAU_COLORS

keys_matter = ['pos', 'lm_imp', 'lm_full', 'lm2full', 'imp_cnn_full', 'lm2imp', 'imp_cnn_imp', 'imp_cnn2imp', 'imp_full', 'imp2full', 'token', 'prefix',
               'pert_var',  'pert_sents', 'top_lm', 'top_imp', 'top_full', 'top_impood', 'top_attn', 'pert_top', 'fusion', 'novel', 'lm', 'ctx', 'easy', 'pert_delta']

# cat = ['none', 'fusion','novel',  'easy']
cat = ['none']


def show_quantiles(var):
    print([round(q, 1) for q in quantiles(var, n=10)])


def load_csv(fname_w_dir):
    data = []
    with open(fname_w_dir, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            data.append(row)
    return data


def comp_var(inp_list):
    if not inp_list:
        return 0
    max_value = max(inp_list)
    v = statistics.mean([abs(this_v - max_value) for this_v in inp_list])
    return v


def create_dict_of_keys(pack, all_keys):
    retrieved = {}
    for k in keys_matter:
        if k in all_keys:
            index = all_keys.index(k)
            v = pack[index]

            try:
                v = eval(v)
                v = float(v)
            except:
                pass
            retrieved[k] = v
    return retrieved


def rt_values(query_dict, keys: List):
    out = []
    for k in keys:
        if k in query_dict:
            out.append(query_dict[k])
    return out


def process_one_pack(data_pack, keys, xaxis_name):
    ret_pairs = create_dict_of_keys(data_pack, keys)
    # if random.random() < 0.01:
    # print(ret_pairs)
    flag = True
    if xaxis_name == 'lm_full':
        distance_x_inp = rt_values(
            ret_pairs, ['lm_full', 'imp_cnn_full'])
        distance_x = min(distance_x_inp)
    else:
        distance_x_inp = rt_values(
            ret_pairs, ['lm_imp',  'imp_cnn_imp', ])
        distance_x = min(distance_x_inp)
    if ret_pairs['top_impood'][0][1] == '<s>':
        flag = False

    #     print(pnum(distance_x))
    #     print(ret_pairs['token'])
    #     print(ret_pairs['top_lm'])
    #     print(ret_pairs['top_imp'])
    #     print(ret_pairs['top_impood'])
    distance_y = min(rt_values(ret_pairs, ['imp_full', 'imp2full']))
    if distance_x > 1.8 and distance_y < 0.25:
        logger.info(distance_x)
        logger.info(ret_pairs['token'])
        logger.info(ret_pairs['prefix'])
        logger.info(ret_pairs['top_lm'])
        # logger.info(ret_pairs['top_imp'])
        logger.info(ret_pairs['top_impood'])
        logger.info(ret_pairs['top_full'])

    prefix = ret_pairs['prefix'].strip()
    if len(prefix) < 2:
        flag = False
    last_prefix = prefix.split(" ")[-1]
    token = ret_pairs['token']
    text = f"{last_prefix} {token}"
    text = text.strip()
    if random.random() < 0.1:
        print(text)

    return distance_x, distance_y, text, flag


def query_func(data_base: str, texts: List[str]):
    prefixs = [word.split(" ")[0]+" " for word in texts]
    prefix_cnts = [data_base.count(x) for x in prefixs]
    cnts = [data_base.count(f"{word}") for word in texts]
    returns = []
    for r, t in zip(prefix_cnts, cnts):
        if r < 1:
            continue
        returns.append(t)
    return returns


if __name__ == "__main__":
    debug = True
    # fname = '/mnt/data0/jcxu/output_file.csv'
    fname = '/mnt/data0/jcxu/csv_xsum/meta.csv'
    # fname = '/mnt/data0/jcxu/output_file_test.csv'
    fname = '/mnt/data0/jcxu/csv_xsum_0.5/viz.csv'
    read_out = load_csv(fname)
    xaxis = 'lm_full'
    # xaxis = 'lm_imp'

    cnndm = "/mnt/data0/jcxu/dev_cnndm_sum.log"
    xsum = '/home/jcxu/sum-interpret/train_ref_sum.log'
    with open(cnndm, 'r') as fd:
        lines = fd.read().splitlines()
        lines = [l for l in lines if l.startswith('vali')]
    lines = [l.split('\t')[1] for l in lines]
    print(lines[0])
    cnndm_s = "\n".join(lines)

    with open(xsum, 'r') as fd:
        lines = fd.read().splitlines()
        lines = [l for l in lines]
    lines = [l.split('\t')[1] for l in lines]
    print(lines[0])
    xsum_s = "\n".join(lines)

    key = read_out[0]
    data = read_out[1:]
    if debug:
        data = data[:5000]
    X, Y, Var, Max = [], [], [], []
    Q = []
    cat_k, cat_v = [], []
    for d in data:
        try:
            distance_x, distance_y, query,  flag = process_one_pack(
                d, key, xaxis_name=xaxis)
            if not flag:
                continue
            # if distance_x<1.5 or distance_y<1.5:
                # continue
            X.append(distance_x)
            Y.append(distance_y)
            Q.append(query)
        except TypeError:
            pass
    # print(f"Num char: {len(cnndm_s)}")
    # print(f"Num char: {len(xsum_s)}")
    # lm
    lms = [c for a, b, c in zip(X, Y, Q) if a < 0.5 and b < 0.5]
    bias = [c for a, b, c in zip(X, Y, Q) if a > 1.5 and b < 0.5]
    pt_bias = [c for a, b, c in zip(X, Y, Q) if a < 0.5 and b > 1.5]
    ctx = [c for a, b, c in zip(X, Y, Q) if b > 0.5 and a>0.5]
    bag= []
    # query_func(cnndm_s,lms)
    # app_set = lms
    # cnn_result = query_func(cnndm_s, app_set)
    # xsum_result = query_func(xsum_s, app_set)
    # bag.append(f"{statistics.mean(cnn_result)}\t{statistics.mean(xsum_result)}")
    app_set = bias
    cnn_result = query_func(cnndm_s, app_set)
    xsum_result = query_func(xsum_s, app_set)
    bag.append(f"{statistics.mean(cnn_result)}\t{statistics.mean(xsum_result)}")
    # app_set = pt_bias
    # cnn_result = query_func(cnndm_s, app_set)
    # xsum_result = query_func(xsum_s, app_set)
    # bag.append(f"{statistics.mean(cnn_result)}\t{statistics.mean(xsum_result)}")
    # app_set = ctx
    # cnn_result = query_func(cnndm_s, app_set)
    # xsum_result = query_func(xsum_s, app_set)
    # bag.append(f"{statistics.mean(cnn_result)}\t{statistics.mean(xsum_result)}")
    print("\n".join(bag))