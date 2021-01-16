# Add labels
from util import *
import os
import pickle


def classify_ctx_fusion(data_pkg):
    d_imp_full = data_pkg['imp_full']

    if 'pert_distb_double' not in data_pkg:
        return False, d_imp_full
    double_top = data_pkg['pert_top1_double']
    single_top = data_pkg['pert_top']
    # print(f"{single_top}\t{double_top}")
    if double_top - single_top > 0.3:
        ranks = np.argsort(data_pkg['pert_distb'])

        print(data_pkg['prefix'])
        print(data_pkg['tgt_token'])
        return True, d_imp_full
    return False, d_imp_full


def classify_ctx(data_pkg):
    d_imp_full = data_pkg['imp_full']
    if d_imp_full > 0.5:
        return True
    else:
        return False


def classify_ctx_easy(data_pkg):
    d_imp_full = data_pkg['imp_full']
    if d_imp_full > 1.8:
        return True
    else:
        return False


def classify_lm(data_pkg):
    d_lm_imp = data_pkg['lm_imp']
    d_impood_imp = data_pkg['imp_cnn_imp']
    x = min(d_lm_imp, d_impood_imp)
    d_imp_full = data_pkg['imp_full']
    if x < 0.5 and d_imp_full < 0.5:
        return True
    else:
        return False


def classify_ctx_novel(data_pkg,meta_data):
    input_doc_ids = meta_data['doc_token_ids']
    target_token = data_pkg['tgt_token_id']
    if target_token in input_doc_ids:
        return False
    else:
        return True


if __name__ == "__main__":
    path = '/mnt/data0/jcxu/output_base_xsum'
    dir_meta = '/mnt/data0/jcxu/meta_pred_xsum'
    files = os.listdir(path)
    cnt = 0
    fusion = 0
    lm = 0
    ctx = 0
    ctx_easy = 0
    fusion_ys = []
    all_ys = []
    files = files[:100]
    for f in files:
        with open(os.path.join(path, f), 'rb') as fd:
            data = pickle.load(fd)
        step_data, meta_data = read_meta_data(dir_meta, f)
        for t, step in enumerate(data):
            isfusion, fusion_y = classify_ctx_fusion(step)
            all_ys.append(fusion_y)
            if isfusion:
                fusion_ys.append(fusion_y)
            is_novel = classify_ctx_novel(step, meta_data)
            islm = classify_lm(step)
            isctx = classify_ctx(step)
            iseasy = classify_ctx_easy(step)
            cnt += 1
            if isfusion:
                fusion += 1
            if islm:
                lm += 1
            if isctx:
                ctx += 1
            if iseasy:
                ctx_easy += 1

    print(cnt)
    print(lm/cnt)
    print(fusion/cnt)
    print(ctx/cnt)
    print(ctx_easy/cnt)
    print(statistics.mean(all_ys))
    print(statistics.mean(fusion_ys))
