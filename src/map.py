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
colors = sns.color_palette("colorblind", 6)


keys_matter = ['pos', 'lm_imp', 'lm_full', 'lm2full', 'imp_cnn_full', 'lm2imp', 'imp_cnn_imp', 'imp_cnn2imp', 'imp_full', 'imp2full', 'token', 'prefix',
               'pert_var',  'pert_sents', 'top_lm', 'top_imp', 'top_full', 'top_impood', 'top_attn', 'pert_top', 'fusion', 'novel', 'lm', 'ctx', 'easy', 'pert_delta']

# cat = ['none', 'fusion','novel',  'easy']
cat = ['none']


def show_quantiles(var):
    print([round(q, 2) for q in quantiles(var, n=4)])


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
    xlabel = r'$d(LM_{\emptyset}, S_{full})$'
    ylabel = r'$d(S_{\emptyset}, S_{full})$'
    style = ['o', '+']
    if label_name == 'none':
        labels = []

        CNT_EZ = "CT-Ez"
        CNT_HD = "CT-Hd"
        LM = "LM"
        Other = "Other"
        for a, b, c, d, change in zip(x, y, pert_v, pert_max, delta):
            # if c < 0.3 and d > 0.6:
            #     l = LM
            #     l = Other
            # elif c < 0.5 and d < 0.3:
            #     l = CNT_HD
            # elif c > 0.5 and d > 0.6:
            #     l = CNT_EZ
            #     l = Other
            # else:
            #     l = Other
            if d < 0.5 and a > 0.5 and b > 0.5:
                l = CNT_HD
            else:
                l = Other

            labels.append(l)
        plat = colors[: len(set(labels))]
        plat = [colors[0]] + ['gray']
    else:
        labels = label_values
        plat = colors[:2]
    stys = []
    for l in labels:
        if l != CNT_HD:
            stys.append(style[0])
        else:
            stys.append(style[1])
    # stys = [style[0] for l in labels if l != CNT_HD else style[1]]
    d = {xlabel: pd.Series(x),
         ylabel: pd.Series(y),
         "label": pd.Series(labels),
         "style": pd.Series(stys),
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
    hue_order = [LM, CNT_EZ, CNT_HD, Other]
    hue_order = [CNT_HD, Other]
    sns.set_context("paper")
    g = sns.JointGrid(data=df, x=xlabel, y=ylabel,
                      hue='label',  palette=plat, height=4, hue_order=hue_order)
    # g.fig.set_size_inches(3,3)
    lm_color = colors[3]
    ctx_color = colors[2]
    c_ctx_easy = ctx_color

    # c_other = colors[1]
    c_other = 'gray'
    pt_color = colors[4]
    td_color = colors[5]

    c_ctx_hd = colors[0]

    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    mk_size = 6
    label_text_size = 12
    # ez_patch = mlines.Line2D([], [], color=c_ctx_easy, marker='o',linestyle='None', markersize=mk_size, label=CNT_EZ)

    hd_patch = mlines.Line2D([], [], color=c_ctx_hd, marker='+', linestyle='None',
                             markersize=mk_size, label=CNT_HD)

    # lm_patch = mlines.Line2D([], [], color=lm_color, marker='o', linestyle='None',markersize=mk_size, label=LM)
    # other_path = mlines.Line2D([], [], color=c_other, marker='o', linestyle='None',markersize=mk_size, label=Other)
    g.plot_joint(sns.scatterplot, s=s, alpha=.5, edgecolors='none',
                 legend=False,
                 style=stys
                 )
    # g.plot_joint(sns.scatterplot, s=s, alpha=.5,
    #              edgecolors='none', legend=False)

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
    line_alpha = 0.5
    lw = 1
    g.ax_joint.fill_between([0.5, 2], [0.5, 0.5], [
                            2, 2],  color='none', edgecolor=ctx_color, alpha=line_alpha, lw=lw)  # Context

    g.ax_joint.fill_between([0, 0.5], [0., 0.0], [
                            0.5, 0.5], color='none', alpha=line_alpha, edgecolor=lm_color, lw=lw)  # LM
    g.ax_joint.fill_between([0, 0.5], [1.5, 1.5], [
                            2, 2], color='none', alpha=line_alpha, edgecolor=pt_color, lw=lw)  # PT
    g.ax_joint.fill_between([1.5, 2], [0., 0.0], [
                            0.5, 0.5], color='none', alpha=line_alpha, edgecolor=td_color, lw=lw)  # TD

    g.ax_joint.text(1.25, 1.25, 'Context',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=label_text_size, color=ctx_color,
                    )
    g.ax_joint.text(0.25, 1.75, 'PT',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=label_text_size, color=pt_color,
                    )
    g.ax_joint.text(1.75, 0.25, 'FT',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=label_text_size, color=td_color,
                    )
    g.ax_joint.text(0.25, 0.25, 'LM',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=label_text_size, color=lm_color,
                    )
    # g.ax_joint.text(0., 0., 'LM',
    #                 horizontalalignment='center',
    #                 verticalalignment='center',
    #                 fontsize=18, color=lm_color)
    # g.ax_joint.annotate('PT', xy=(2, 2), xytext=(0.2, 1.7), color=pt_color,
    # arrowprops=dict(facecolor='black', shrink=0.05),
    # )
    # g.ax_joint.annotate('TrainData', xy=(2, 2), xytext=(1.7, 0.2), color=td_color,
    #                     # arrowprops=dict(facecolor='black', shrink=0.05),
    #                     )
    # g.ax_joint.annotate('LM', xy=(0.25, 0.25), xytext=(0.2, 0.2), color=lm_color,
    #                     # arrowprops=dict(facecolor='none', shrink=0.05),
    #                     )
    bbox_to_anchor = (-4.5, 1.25)
    # plt.legend(handles=[lm_patch, ez_patch, hd_patch, other_path], loc='upper left',ncol=4, bbox_to_anchor=bbox_to_anchor, handletextpad=0, columnspacing=0.2)
    plt.legend(handles=[hd_patch], loc='upper left', ncol=1,
               bbox_to_anchor=bbox_to_anchor, handletextpad=0, columnspacing=0.2)
    # g.legend(handles=[lm_patch, ez_patch, hd_patch], loc='best', ncol=3,
    #  bbox_to_anchor=bbox_to_anchor,
    #  handletextpad=0.1
    #  )
    # ax.fill_between([0.5,2], [0., 0.0],[0.5,0.5] , color='grey', alpha=0.1)
    # ax.legend()

    # scatter = ax.scatter(x, y, s=4)
    # ax.set_xlabel(f"{xaxis_name}")
    # ax.set_ylabel(r'$d$(LM-FT, FULL)')
    # sns.scatterplot(x=x, y=y, hue=labels,s=4)
    # sns.jointplot(x=x, y=y, hue=labels, kind="kde")
    # plt.tight_layout(rect=[0,0,0.8,1])
    # plt.show()

    plt.savefig(map_name, format='pdf', bbox_inches="tight")

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
    # fname = '/mnt/data0/jcxu/csv_xsum/meta.csv'
    # fname = '/mnt/data0/jcxu/output_file_test.csv'
    # fname = '/mnt/data0/jcxu/csv_xsum/viz_t_0.5.csv'
    fname = f"{args.dir_stat}/viz.csv"
    read_out = load_csv(fname)
    xaxis = 'lm_full'
    # xaxis = 'lm_imp'

    key = read_out[0]
    data = read_out[1:]
    if debug:
        data = data[:300]
    max_count = 3800
    cnt = 0
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
            cnt+=1
            if cnt>max_count:
                break
        except TypeError:
            pass
    show_quantiles(X)
    show_quantiles(Y)
    show_quantiles(Var)
    show_quantiles(Max)
    print(cnt)
    for label in cat:
        name_of_map = f"map_{args.data_name}_{label}_{xaxis}.pdf"
        draw_map_figure(X, Y, Var, Max, delta, name_of_map, xaxis, label, None)
