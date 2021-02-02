import math
import numpy as np
from util import *
task = 'attn'
task = 'int_grad'
task = f"{task}_sent_sel"

# sent_path = f"/mnt/data0/jcxu/task_{task}_sent_sel_xsum_0.5"
# eval_path = f"/mnt/data0/jcxu/eval_{task}_sent_sel_sel_tok_xsum_0.5"

path = f"/mnt/data0/jcxu/task_{task}_xsum_0.5"
eval_path = f"/mnt/data0/jcxu/eval_{task}_sel_tok_xsum_0.5"

meta_path = '/mnt/data0/jcxu/meta_pred_xsum_0.5'
output_path = f"/mnt/data0/jcxu/viz/{task}"

output_bas_path = f"/mnt/data0/jcxu/output_base_xsum_0.5"


def visulize_one_attribution(meta_data, doc, output):
    T = len(meta_data)

    all_data = []
    for t in range(T):
        if isinstance(output[t], dict) and 'doc_token_ids' in output[t]:
            doc = output[t]['doc_token_ids']
            out_scores = output[t]['output']
        else:
            out_scores = output[t]
        if isinstance(out_scores, torch.Tensor):
            out_scores = out_scores.tolist()
        msg = []
        _meta = meta_data[t]
        msg.append(f"Prefix: {_meta['prefix']} TGT: {_meta['tgt_token']}")
        if task == 'occ':
            rank = np.argsort(out_scores)
        else:
            rank = np.argsort(out_scores)[::-1]
        dup_doc = doc.copy()
        dup_doc = [x if x > 2 else 1 for x in dup_doc]
        dup_doc = [tokenizer.decode(x, skip_special_tokens=True)
                   for x in dup_doc]

        for i in range(k):
            dup_doc[rank[i]] = f"<strong>{i}_{dup_doc[rank[i]]}</strong>"
        msg.append(''.join(dup_doc))
        msg.append("")
        all_data.append(msg)
    return all_data


def decode_eval_result(eval_pkg, meta_data):
    all_data = []
    amz = False
    for t, ev_rs in enumerate(eval_pkg):
        target_token_ids = meta_data[t]['tgt_token_id']
        tmp = []
        loss = ev_rs['loss']
        met = ev_rs['meta']
        prob = ev_rs['prob']
        logit = ev_rs['logit']
        prob = torch.nn.functional.softmax(logit/0.5, dim=-1)
        for idx, l, m in zip(range(len(loss)), loss, met):
            buget = m['bud']
            
            p = prob[idx][target_token_ids]
            # print(f"Bud: {buget}\tProb: {pnum(p)}")
            tmp.append(f"Bud: {buget}\tProb: {pnum(p)}")

        # tmp.append(f"Delta: {pnum(prob_delta)}")
        if math.exp(-loss[1]) - math.exp(-loss[0]) > 0.5:
            tmp.append("#####")
            amz = True
        tmp.append("")
        all_data.append(tmp)
    return all_data, amz


def print_base(output_pkg):
    for t, unit in enumerate(output_pkg):
        print(t)
        print(unit['tgt_token'])
        print(unit['top_impood'])
        print(unit['top_full'])


max = 2000
cnt = 0
files = os.listdir(path)
files = ['34832755.pkl','37267397.pkl']
k = 10
for f in files:
    cnt += 1
    data = load_pickle(path, f)
    eval_result = load_pickle(eval_path, f)
    # sent_data = load_pickle(sent_path, f)
    outputbase = load_pickle(output_bas_path, f)
    meta = load_pickle(meta_path, f)
    print_base(outputbase)
    uid = eval_result[0]['meta'][0]['id']
    doc = data['doc_token_ids']
    output = data['output']

    meta_data = meta['data']
    meta_meta = meta['meta']
    eval_res, amz = decode_eval_result(eval_result, meta_data)
    att_result = visulize_one_attribution(meta_data, doc, output)
    if amz:
        uid = f"!{uid}"
    everything = []
    for e, a in zip(eval_res, att_result):
        everything += e
        everything += a
    with open(os.path.join(output_path, f"{uid}.html"), 'w') as fd:
        for x in everything:
            fd.write(x)
            fd.write('<br>')
    if cnt > max:
        break
