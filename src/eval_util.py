from collections import OrderedDict
from typing import List
from helper import *
from util import *


def merge(intervals):
    out = []
    for i in sorted(intervals, key=lambda i: i[0]):
        if out and i[0] <= out[-1][1]:
            out[-1][1] = max(out[-1][1], i[1])
        else:
            out += i,
    return out


def prepare_concat_input_seq(sel_budget: int, input_doc: List[int], top_indicies: List[int], ctx_window=1, sep_token_id=6) -> List[int]:
    # .=4  ,=6  space=1437

    already_locations = []
    cursor = 0
    max_doc_len = len(input_doc) - 1  # <eos>
    min_doc_len = 1  # <s>
    queue = []
    while len(already_locations) < sel_budget:
        if queue:
            candidate_loc = queue.pop(0)
            if candidate_loc in already_locations:
                continue
            elif candidate_loc >= min_doc_len and candidate_loc < max_doc_len:
                already_locations.append(candidate_loc)
                continue
        # if the queue is empty
        sel_idx = top_indicies[cursor]
        cursor += 1
        if sel_idx > max_doc_len:
            continue
        left, right = max(
            min_doc_len, sel_idx-ctx_window), min(sel_idx + ctx_window+ 1, max_doc_len)
        queue.append(sel_idx)
        for idx in range(left, right):
            queue.append(idx)
        if cursor >= len(top_indicies):
            break
    selected_tokens = [0]
    already_locations.sort()
    for idx, loc in enumerate(already_locations):
        selected_tokens.append(input_doc[loc])
        if idx + 1 < len(already_locations):
            next_tok_idx = already_locations[idx+1]
            if next_tok_idx - 1 != loc:
                selected_tokens.append(sep_token_id)
    selected_tokens.append(2)

    return selected_tokens


def rm_tok(sel_budget: int, input_doc: List[int], top_indicies: List[int], ctx_window=1, sep_token_id=6, mask_token_id=1) -> List[int]:
    # .=4  ,=6  space=1437
    selected_tokens = input_doc.copy()
    already_locations = []

    cursor = 0
    max_doc_len = len(input_doc) - 1  # <eos>
    min_doc_len = 1  # <s>
    queue = []
    while len(already_locations) < sel_budget:
        if queue:
            candidate_loc = queue.pop(0)
            if candidate_loc in already_locations:
                continue
            elif candidate_loc >= min_doc_len and candidate_loc < max_doc_len:
                already_locations.append(candidate_loc)
                continue
        # if the queue is empty
        sel_idx = top_indicies[cursor]
        cursor += 1
        if sel_idx > max_doc_len:
            continue
        left, right = max(
            min_doc_len, sel_idx-ctx_window), min(sel_idx + ctx_window+1, max_doc_len)
        queue.append(sel_idx)
        for idx in range(left, right):
            queue.append(idx)
        if cursor >= len(top_indicies):
            break
    for loc in already_locations:
        selected_tokens[loc] = mask_token_id
    return selected_tokens


def render_sel_sent(sel_budget: int, list_sent_tokens: List[List[int]], top_indicies: List[int], sep_token_id=50118) -> List[int]:
    # .=4  ,=6  space=1437
    # without <s> and </s>
    selected_tokens = []
    cursor = 0
    try:
        while sel_budget > 0:
            sel_idx = top_indicies[cursor]
            cursor += 1
            if selected_tokens:
                span_bpe = [sep_token_id] + list_sent_tokens[sel_idx]
            else:
                span_bpe = list_sent_tokens[sel_idx]
            selected_tokens += span_bpe
            sel_budget -= 1

    except IndexError:
        pass
    return selected_tokens


def render_rm_sent(sel_budget, list_sent_tokens, top_indicies, total_num_sent, sep_token_id=50118):
    sel_sent_index = list(range(total_num_sent))
    cursor = 0
    while sel_budget > 0:
        sel_idx = top_indicies[cursor]
        sel_budget -= 1
        cursor += 1
        if sel_idx >= total_num_sent:
            continue
        if cursor >= total_num_sent:
            break
        sel_sent_index[sel_idx] = None

    sel_sent_index = [x for x in sel_sent_index if x != None]
    output = []
    for sel_id in sel_sent_index:
        if output:
            output.append(sep_token_id)

        output += list_sent_tokens[sel_id]
    return output


def batch_data(pack_data_inp, tokenizer, device):
    group_by_prefix = [[] for _ in range(100)]

    for d in pack_data_inp:
        eff_len = d['prefix'].size()[-1]
        inp_text = tokenizer.decode(d['inp'])
        d['inp_text'] = inp_text
        group_by_prefix[eff_len].append(d)
    group_by_prefix = [g for g in group_by_prefix if g]
    for group in group_by_prefix:
        gather_texts = [member['inp_text'] for member in group]
        batch_enc = tokenizer.prepare_seq2seq_batch(
            src_texts=gather_texts)

        input_ids = torch.LongTensor(batch_enc['input_ids']).to(device)
        attn_masks = torch.LongTensor(batch_enc['attention_mask']).to(device)
        gather_prefix = [member['prefix'] for member in group]
        gather_prefix = torch.cat(gather_prefix).to(device)
        gather_tgt = [member['tgt'] for member in group]
        gather_tgt = torch.LongTensor(gather_tgt).to(device)
        gather_metas = [member['meta'] for member in group]
        yield input_ids, attn_masks, gather_prefix, gather_tgt, gather_metas


def yield_random_rank(input_doc):
    pass


def summarize_result(result: List[dict], args):
    loss_dict = {}
    task = args.task
    for r in result:
        loss = r['loss']
        meta = r['meta']
        for l, m in zip(loss, meta):
            mode = m['mode']
            bud = m['bud']
            if bud in loss_dict:
                loss_dict[bud].append(l)
            else:
                loss_dict[bud] = [l]

    for k, v in loss_dict.items():
        avg_v = statistics.mean(v)
        loss_dict[k] = avg_v
    od = OrderedDict(sorted(loss_dict.items()))
    keys = list(od.keys())
    values = list(od.values())
    print(f"Task: {task};Mode: {mode}")
    keys = [str(k) for k in keys]
    values = [str(v) for v in values]
    print(",".join(keys))
    print(",".join(values))
    out = [task, mode, ",".join(keys), ",".join(values)]
    result_dict = {
        'task': task,
        'mode': mode,

    }
    return ",".join(out)


def agg_time_step_task_output(seq_of_pred: List[torch.Tensor], device):
    if isinstance(seq_of_pred[0], List):
        seq_of_pred = [torch.tensor(x) for x in seq_of_pred]
    if len(seq_of_pred[0].size()) < 2:
        seq_of_pred = [x.unsqueeze(0) for x in seq_of_pred]
    all_preds = torch.cat(seq_of_pred).to(device)
    all_pred_abs = torch.abs(all_preds)
    return all_pred_abs


def comp_sent_avg_activation(pred: List[float], map_index):
    l = min(len(pred), len(map_index))
    pred = pred[:l]
    map_index = map_index[:l]
    total_sents_to_consider = map_index[-1] + 1
    mapped_scores = []
    for sent_idx in range(total_sents_to_consider):
        last_occ = max(loc for loc, val in enumerate(
            map_index) if val == sent_idx)
        first_occ = min(loc for loc, val in enumerate(
            map_index) if val == sent_idx)
        mapped_scores.append(pred[first_occ:last_occ+1])
    mean_score = [statistics.mean(s) for s in mapped_scores]

    rank = [i[0]
            for i in sorted(enumerate(mean_score), key=lambda x:x[1])][::-1]
    return rank, total_sents_to_consider


def extract_from_baseline(meta_data, step_data, args, budget: List[int], device):
    document_tokens = meta_data['doc_token_ids'][:args.hard_max_len]
    map_index = meta_data['map_index']
    sent_token_ids = meta_data['sent_token_ids']
    eval_mode = args.eval_mode
    uid = meta_data['id']

    return_data = []
    # for token, get index of token to add or remove
    if 'sent' in eval_mode:
        # aggeragate units in sent

        for t, step in enumerate(step_data):
            # first element is the index of the top 1 sent
            total_sents_to_consider = len(sent_token_ids)
            sent_ids = list(range(total_sents_to_consider))
            if args.task == 'random':
                random.shuffle(sent_ids)
                rank_of_sent_index = sent_ids
            elif args.task == 'lead':
                rank_of_sent_index = sent_ids

            if 'sel' in eval_mode:
                for bud in budget:
                    rendered_tokens = render_sel_sent(
                        sel_budget=bud, list_sent_tokens=sent_token_ids, top_indicies=rank_of_sent_index)
                    unit = assemble_units(
                        rendered_tokens, step_data, t, bud, eval_mode, uid)
                    return_data.append(unit)
            else:
                for bud in budget:
                    rendered_tokens = render_rm_sent(
                        sel_budget=bud, list_sent_tokens=sent_token_ids, top_indicies=rank_of_sent_index, total_num_sent=total_sents_to_consider)
                    unit = assemble_units(
                        rendered_tokens, step_data, t, bud, eval_mode, uid)
                    return_data.append(unit)
    elif 'tok' in eval_mode:
        for t, step in enumerate(step_data):
            # first element is the index of the top 1 sent
            total_num_tok = len(document_tokens)
            token_ids = list(range(total_num_tok))

            if args.task == 'random':
                random.shuffle(token_ids)
                rank_of_tok_index = token_ids
            elif args.task == 'lead':
                rank_of_tok_index = token_ids

            if 'sel' in eval_mode:
                for bud in budget:
                    rendered_tokens = prepare_concat_input_seq(
                        bud, document_tokens, rank_of_tok_index)
                    unit = assemble_units(
                        rendered_tokens, step_data, t, bud, eval_mode, uid)
                    return_data.append(unit)
            else:
                for bud in budget:
                    rendered_tokens = rm_tok(
                        bud, document_tokens, rank_of_tok_index)
                    unit = assemble_units(
                        rendered_tokens, step_data, t, bud, eval_mode, uid)
                    return_data.append(unit)

    return return_data


def assemble_units(rendered_tokens, step_data, t, bud, eval_mode, uid):
    d = {
        "inp": rendered_tokens,
        "prefix": step_data[t]['prefix_token_ids'],
        "tgt": step_data[t]['tgt_token_id'],
        "meta": {
            'prefix': step_data[t]['prefix'],
            'token': step_data[t]['token'],
            'pos': step_data[t]['pos'],
            'bud': bud,
            'mode': eval_mode,
            'id': uid
        }
    }
    return d


def process_occlusion(task_output):
    output = task_output['output']
    if isinstance(output[0], list):
        output = [[1 - x for x in one] for one in output]
        task_output['output'] = output
        return task_output
    for idx, out in enumerate(output):
        inner_out = out['output']
        inner_out = [1-x for x in inner_out]
        output[idx]['output'] = inner_out

    task_output['output'] = output
    return task_output


def extract_from_task_output(task_output, meta_data, step_data, args, budget: List[int], device):
    # 'doc_token_ids': new_input_doc,
    # 'map_index':meta_data['map_index'],
    # 'sent_token_ids':meta_data['sent_token_ids'],
    # 'output': outputs
    if 'occ' in args.task:
        task_output = process_occlusion(task_output)
    document_tokens = task_output['doc_token_ids'][:args.hard_max_len]
    map_index = task_output['map_index']
    # Truncate sent
    sent_token_ids = task_output['sent_token_ids']
    doc_tok_len = len(document_tokens)
    max_sent_idx = map_index[doc_tok_len-1]
    sent_token_ids = sent_token_ids[:max_sent_idx+1]
    map_index = map_index[:doc_tok_len]
    pred_output = task_output['output']
    eval_mode = args.eval_mode
    uid = meta_data['id']
    return_data = []
    # for token, get index of token to add or remove
    if 'sent' in eval_mode:
        if args.sent_pre_sel:
            raise NotImplementedError(
                "Do not expect sent_pre_sel working with sentence level evaluation.")

        agg_pred_output = agg_time_step_task_output(pred_output, device)
        # aggeragate units in sent
        agg_pred_output = agg_pred_output.cpu().tolist()
        for t, step_pred in enumerate(agg_pred_output):
            # first element is the index of the top 1 sent
            total_sents_to_consider = len(sent_token_ids)
            sent_ids = list(range(total_sents_to_consider))

            rank_of_sent_index, total_sents_to_consider = comp_sent_avg_activation(
                step_pred, map_index)

            if 'sel' in eval_mode:
                for bud in budget:
                    rendered_tokens = render_sel_sent(
                        sel_budget=bud, list_sent_tokens=sent_token_ids, top_indicies=rank_of_sent_index)
                    unit = assemble_units(
                        rendered_tokens, step_data, t, bud, eval_mode, uid)
                    return_data.append(unit)
            else:
                for bud in budget:
                    rendered_tokens = render_rm_sent(
                        sel_budget=bud, list_sent_tokens=sent_token_ids, top_indicies=rank_of_sent_index, total_num_sent=total_sents_to_consider)
                    unit = assemble_units(
                        rendered_tokens, step_data, t, bud, eval_mode, uid)
                    return_data.append(unit)
    elif 'tok' in eval_mode:
        if args.sent_pre_sel:
            agg_pred_output = [t_pred['output'] for t_pred in pred_output]
        else:
            agg_pred_output = agg_time_step_task_output(pred_output, device)
            agg_pred_output = agg_pred_output.cpu().tolist()
        for t, step_pred in enumerate(agg_pred_output):
            rank_of_tok_index = argsort(step_pred)[::-1]
            if args.sent_pre_sel:
                document_tokens = pred_output[t]['doc_token_ids']
            else:
                pass
            # print(step_pred[rank_of_tok_index[0]])
            # print(step_pred[rank_of_tok_index[-1]])
            if 'sel' in eval_mode:
                for bud in budget:
                    rendered_tokens = prepare_concat_input_seq(
                        bud, document_tokens, rank_of_tok_index)
                    unit = assemble_units(
                        rendered_tokens, step_data, t, bud, eval_mode, uid)
                    return_data.append(unit)
            else:
                for bud in budget:
                    rendered_tokens = rm_tok(
                        bud, document_tokens, rank_of_tok_index)
                    unit = assemble_units(
                        rendered_tokens, step_data, t, bud, eval_mode, uid)
                    return_data.append(unit)
    return return_data
