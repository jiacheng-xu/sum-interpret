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
    if distance_x < 0.5 and distance_y > 1.5:
        logger.info(f"X:{distance_x}\tY:{distance_y}")

        logger.info(ret_pairs['prefix'])
        logger.info(f"Token: {ret_pairs['token']}")
        logger.info(f"LM: {ret_pairs['top_lm']}")
        logger.info(f"IMP:{ret_pairs['top_imp']}")
        logger.info(f"IMP OOD: {ret_pairs['top_impood']}")
        logger.info(f"FULL: {ret_pairs['top_full']}")
    # pert_distribution = ret_pairs['pert_distb']
    # pert_distribution = [float(m) for m in pert_distribution]
    # variation = comp_var(pert_distribution)
    # max_pert = max(pert_distribution)
    variation = ret_pairs['pert_var']
    max_pert = ret_pairs['pert_top']
    pert_delta = ret_pairs['pert_delta']
    if 'nan' in [max_pert, variation]:
        flag = False
    cat_vals = []
    cat_keys = []
    for thisk, thisv in ret_pairs.items():
        if thisk in cat:
            cat_vals.append(thisv)
            cat_keys.append(thisk)

    return distance_x, distance_y, variation, max_pert, pert_delta, cat_keys, cat_vals, flag


def draw_map_figure(x, y, pert_v, pert_max, delta, map_name, xaxis_name, label_name, label_values):
    xlabel = r'$d$(LM, FULL)'
    ylabel = r'$d$(LM-FT, FULL)'

    if label_name == 'none':
        labels = []
        no_ctx_lm = 'no_ctx_lm'
        no_ctx_td = 'no_ctx_td'
        no_ctx_other = 'no_ctx_other'

        ctx_single = 'ctx_single'
        ctx_double = 'ctx_double'
        ctx_others = 'ctx_other'

        CNT_EZ = "CTX-Ez"
        CNT_HD = "CTX-Hd"
        LM = "LM"
        for a, b, c, d, change in zip(x, y, pert_v, pert_max, delta):
            # if c < 0.2 and d < 0.3:
            # l = 'var & max'
            # elif c > 0.3 and d < 0.3:
            #     l = 'VAR & max'
            if c < 0.15 and d > 0.7:
                if a < 0.5 and b < 0.5:
                    l = no_ctx_lm
                elif a > 1.5 and b < 0.5:
                    l = no_ctx_td
                else:
                    l = no_ctx_other
                labels.append(l)
                continue

            if c > 0.4 and d > 0.7:
                l = ctx_single
            elif change != '' and float(change) > 0.2:
                l = ctx_double
            else:
                l = ctx_others

            labels.append(l)
        plat = colors[: len(set(labels))]
    else:
        labels = label_values
        plat = colors[:2]
    d = {xlabel: pd.Series(x),
         ylabel: pd.Series(y),
         "label": pd.Series(labels),
         }

    df = pd.DataFrame(d)
    # sns.set(style="whitegrid")

    # fig, ax = plt.subplots(figsize=(3, 3))
    # f = plt.figure(figsize=(1,1))
    # fig, ax =pyplot.subplots(figsize=(2,2))
    # gs = f.add_gridspec(1,1)
    # ax = f.add_subplot(gs[0, 0])
    # sns.set(rc={'figure.figsize':(9,9)})
    # plt.rcParams['figure.figsize']=(10,10)

    # f = plt.figure(figsize=(3, 3))
    # kw = dict(num=5, color=scatter.cmap(0.7), fmt="$ {x:.2f}", func=lambda s: np.sqrt(s/.3)/3)
    # legend2 = ax.legend(*scatter.legend_elements(**kw),loc="lower right", title="Price")
    s = 2
    enum_types = list(set(labels))
    # for en in enum_types:
    #     oa = [a for a, b, c in zip(x, y, labels) if c == en]
    #     ob = [b for a, b, c in zip(x, y, labels) if c == en]
    #     scatter = ax.scatter(oa, ob, s=s, marker='o', label=str(en), alpha=0.5,edgecolors='none')
    g = sns.JointGrid(data=df, x=xlabel, y=ylabel,
                      hue='label', palette=plat, height=4)
    # g.fig.set_size_inches(3,3)

    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    # ez_patch = mpatches.Patch(color=plat[0], label=CNT_EZ)
    ez_patch = mlines.Line2D([], [], color=plat[0], marker='o',
                             linestyle='None', markersize=10, label=CNT_EZ)

    # hd_patch = mpatches.Patch(color=plat[1], label=CNT_HD)
    hd_patch = mlines.Line2D([], [], color=plat[1], marker='o', linestyle='None',
                             markersize=10, label=CNT_HD)
    # lm_patch = mpatches.Patch(color=plat[2], label=LM)
    lm_patch = mlines.Line2D([], [], color=plat[2], marker='o', linestyle='None',
                             markersize=10, label=LM)
    """
    bbox_to_anchor = (-7.5, 1.25)
    plt.legend(handles=[lm_patch, ez_patch, hd_patch], loc='upper left', ncol=3, bbox_to_anchor=bbox_to_anchor, handletextpad=0.1)
    g.plot_joint(sns.scatterplot, s=s, alpha=.5, edgecolors='none',
                 legend=False
                 )
    """
    g.plot_joint(sns.scatterplot, s=s, alpha=.5,
                 edgecolors='none', legend=False)

    g.plot_marginals(sns.histplot, bins=40, linewidth=0, multiple="stack")
    # g.ax.legend(loc=2)

    # g=sns.JointGrid(data=df, x='x', y='y', hue='label', size=s)
    # g.plot(sns.scatterplot, sns.histplot)

    # vMa = [a for a,b,c in zip(x,y,labels) if c == LM]
    # vMb = [b for a,b,c in zip(x,y,labels) if c == LM]
    # scatter = ax.scatter(vMa, vMb, marker='+',s=s, label=LM)
    # VMa = [a for a,b,c in zip(x,y,labels) if c == CNT_EZ]
    # VMb = [b for a,b,c in zip(x,y,labels) if c == CNT_EZ]
    # scatter = ax.scatter(VMa, VMb, marker='+',s=s, label=CNT_EZ)
    # ax.fill_between([0,2], [0.5, 0.5],[2,2] , color='C0', alpha=0.1)
    # ax.fill_between([0,0.5], [0., 0.0],[0.5,0.5] , color='C1', alpha=0.1)
    # ax.fill_between([0.5,2], [0., 0.0],[0.5,0.5] , color='grey', alpha=0.1)
    # ax.legend()

    # scatter = ax.scatter(x, y, s=4)
    # ax.set_xlabel(f"{xaxis_name}")
    # ax.set_ylabel(r'$d$(LM-FT, FULL)')
    # sns.scatterplot(x=x, y=y, hue=labels,s=4)
    # sns.jointplot(x=x, y=y, hue=labels, kind="kde")
    # plt.tight_layout()
    # plt.show()

    plt.savefig(map_name, format='pdf')

    # fig.tight_layout()
    # fig.show()
    # fig.savefig(map_name, format='pdf')


if __name__ == "__main__":
    debug = False
    parser = common_args()
    args = parser.parse_args()
    args = fix_args(args)
    logger.info(args)
    # fname = '/mnt/data0/jcxu/output_file.csv'
    fname = '/mnt/data0/jcxu/csv_xsum/meta.csv'
    # fname = '/mnt/data0/jcxu/output_file_test.csv'
    # fname = '/mnt/data0/jcxu/csv_xsum/viz_t_0.5.csv'
    fname = f"{args.dir_stat}/viz.csv"
    read_out = load_csv(fname)
    xaxis = 'lm_full'
    # xaxis = 'lm_imp'

    key = read_out[0]
    data = read_out[1:]
    if debug:
        data = data[:100]
    X, Y, Var, Max = [], [], [], []
    delta = []
    cat_k, cat_v = [], []
    for d in data:
        try:
            distance_x, distance_y, variation, max_pert, pert_delta, cat_keys, cat_vals,  flag = process_one_pack(
                d, key, xaxis_name=xaxis)
            if not flag:
                continue
            # if distance_x<1.5 or distance_y<1.5:
                # continue
            X.append(distance_x)
            Y.append(distance_y)
            Max.append(max_pert)
            Var.append(variation)
            delta.append(pert_delta)
            cat_k = cat_keys
            cat_v.append(cat_vals)
        except TypeError:
            pass
    show_quantiles(X)
    show_quantiles(Y)
    show_quantiles(Var)
    show_quantiles(Max)
    for label in cat:
        name_of_map = f"map_{label}_{xaxis}.pdf"
        if label == 'none':
            draw_map_figure(X, Y, Var, Max, delta,
                            name_of_map, xaxis, label, None)
        else:
            lb_idx = cat_keys.index(label)
            label_val = [m[lb_idx] for m in cat_v]
            draw_map_figure(X, Y, Var, Max, name_of_map,
                            xaxis, label, label_val)
