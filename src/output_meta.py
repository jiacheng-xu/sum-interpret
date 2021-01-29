# Add labels
from multiprocessing import Pool
import nltk
from nltk.corpus import stopwords
from helper import *
from util import *
import os
import pickle
import pandas as pd
import numpy as np


def classify_ctx_fusion(data_pkg):
    d_imp_full = data_pkg['imp_full']

    if 'pert_distb_double' not in data_pkg:
        return False, d_imp_full
    double_top = data_pkg['pert_top1_double']
    single_top = data_pkg['pert_top']
    # print(f"{single_top}\t{double_top}")
    if double_top - single_top > 0.3:
        ranks = np.argsort(data_pkg['pert_distb'])
        # print(data_pkg['prefix'])
        # print(data_pkg['tgt_token'])
        return True, d_imp_full
    return False, d_imp_full


def classify_ctx(data_pkg):
    d_imp_full = data_pkg['imp_full']
    if d_imp_full > 0.5:
        return True
    else:
        return False


def classify_ctx_easy(data_pkg):
    pert_distb = data_pkg['pert_distb']
    max_one = max(pert_distb)
    if max_one > 0.8:
        return True
    else:
        return False


def confidence_improve(data_pkg):
    top_full = data_pkg['top_full']
    top_imp = data_pkg['top_imp']
    full0 = top_full[0]
    imp0 = top_imp[0]
    if full0[1] == imp0[1]:
        return True
    else:
        return False


def classify_lm(data_pkg):
    d_lm_imp = data_pkg['lm_imp']
    d_impood_imp = data_pkg['imp_cnn_imp']
    x = min(d_lm_imp, d_impood_imp)
    d_imp_full = data_pkg['imp_full']
    if x < 1 and d_imp_full < 1:
        return True
    else:
        return False


stop = stopwords.words('english')


def classify_ctx_novel(data_pkg, meta_data, tokenizer):
    input_doc_ids = meta_data['doc_token_ids']
    target_token = data_pkg['token']

    if target_token in stop:
        return False
    target_token_id = tokenizer.encode(target_token)[1]
    target_token_id_space = tokenizer.encode(" "+target_token)[1]

    if target_token_id in input_doc_ids or target_token_id_space in input_doc_ids:
        return False
    else:
        return True


def categorize(step_data, tokenizer):
    isfusion, fusion_y = classify_ctx_fusion(step_data)
    is_novel = classify_ctx_novel(step_data, meta_data, tokenizer)
    islm = classify_lm(step_data)
    isctx = classify_ctx(step_data)
    iseasy = classify_ctx_easy(step_data)
    if (not iseasy) and isctx:
        hard = True
    else:
        hard = False
    return {
        'fusion': isfusion,
        'novel': is_novel,
        'lm': islm,
        'ctx': isctx,
        'hard': hard,
        'all': True
    }


def example_print(step,meta_data,tokenizer):
    viz_dict = {}
    for k, v in step.items():
        if isinstance(v, torch.Tensor):
            continue
        elif isinstance(v, float):
            v = pnum(v)
        viz_dict[k] = v

    doc = viz_dict['pert_sents']
    doc = [
        f"{idx}: {pnum(viz_dict['pert_distb'][idx])}    {s}" for idx, s in enumerate(doc)]
    doc_in_str = "\n".join(doc)
    viz_dict['pert_sents'] = doc_in_str
    del viz_dict['map_index']
    del viz_dict['pert_distb']
    if 'pert_distb_double' in viz_dict:
        combos = viz_dict['pert_comb']
        tmp = viz_dict['pert_distb_double']
        sort_res = argsort(tmp)[::-1]
        tmp = [pnum(t) for t in tmp]
        comb = []
        l = min(10, len(combos))
        for i in range(l):
            j = sort_res[i]
            comb.append(f"{combos[j]} - {tmp[j]}")
        x = "\n".join(comb)
        viz_dict['comb'] = x
        del viz_dict['pert_comb']
        del viz_dict['pert_distb_double']
        delta = float(viz_dict['pert_top1_double']) - \
            float(viz_dict['pert_top'])
        viz_dict['pert_delta'] = delta

    isfusion, fusion_y = classify_ctx_fusion(step)
    is_novel = classify_ctx_novel(step, meta_data, tokenizer)
    islm = classify_lm(step)
    isctx = classify_ctx(step)
    iseasy = classify_ctx_easy(step)
    viz_dict['fusion'] = isfusion
    viz_dict['novel'] = is_novel
    viz_dict['lm'] = islm
    viz_dict['ctx'] = isctx
    viz_dict['easy'] = iseasy
    return viz_dict


tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


def process_one(dir_base, dir_meta, f):
    print(dir_base, dir_meta, f)
    output_base = load_pickle(dir_base, f)
    step_data, meta_data = read_meta_data(dir_meta, f)
    all_data = []
    for t, step in enumerate(output_base):
        out = example_print(step,meta_data, tokenizer)
        out['id'] = f
        keys = list(out.keys())
        values = list(out.values())
        d = pd.DataFrame([values], columns=keys)
        all_data.append(d)
    return all_data


if __name__ == "__main__":
    parser = common_args()
    args = parser.parse_args()
    args = fix_args(args)
    logger.info(args)
    dir_meta = args.dir_meta
    dir_eval_save = args.dir_eval_save
    dir_base = args.dir_base

    files = os.listdir(dir_base)

    # occ_files = os.listdir('/mnt/data0/jcxu/task_occ_xsum_0.5')
    print(dir_eval_save)
    files_meta = os.listdir(dir_meta)
    # files = [f for f in files if f in occ_files]

    print(len(files))
    random.shuffle(files)
    # files = files[:100]
    cnt = 0
    fusion = 0
    lm = 0
    ctx = 0
    ctx_easy = 0
    fusion_ys = []
    all_ys = []
    dict_bud = {}
    all_dists = {}
    tags = ['fusion', 'novel', 'lm', 'ctx', 'hard', 'all']

    for tag in tags:
        all_dists[tag] = {}
    dir_latex = '/mnt/data0/jcxu/latex'
    all_data = []
    inps = [(dir_base, dir_meta, f) for f in files]
    with Pool(multiprocessing.cpu_count()) as p:
        result = p.starmap(process_one, inps)
    # for idx, f in enumerate(files):
    #     print(idx)
    #     output_base = load_pickle(dir_base, f)
    #     step_data, meta_data = read_meta_data(dir_meta, f)
    #     for t, step in enumerate(output_base):

    #         out = example_print(step, tokenizer)
    #         out['id'] = f
    #         keys = list(out.keys())
    #         values = list(out.values())
    #         d = pd.DataFrame([values], columns=keys)
    #         all_data.append(d)
    flatten = lambda t: [item for sublist in t for item in sublist]
    final_result = flatten(result)
    df = pd.concat(final_result, ignore_index=True)
    df.to_csv(path_or_buf=f"{args.dir_stat}/viz.csv", index=True)
