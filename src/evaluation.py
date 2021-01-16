# batch evaluation
import time
import datetime
from util import *
from helper_run_bart import run_full_model_slim, init_bart_family
from eval_util import *


def run_eval(model_sum, data_feeder):
    result_bag = []
    for data_stream in data_feeder:
        input_ids, attn_masks, gather_prefix, gather_tgt, gather_metas = data_stream
        output_tok, _, _, loss = run_full_model_slim(
            model_sum,
            input_ids=input_ids,
            attention_mask=attn_masks,
            decoder_input_ids=gather_prefix,
            targets=gather_tgt,
            device=device)
        loss = loss.tolist()
        result_bag.append({'loss': loss,
                           'meta': gather_metas})
    return result_bag


def prepare_inp_grad(dir, fname, output_base_data, meta_data, device, context_window=2, distance_thres=0.5):
    # prepare: input explanation, prefix, and gt
    list_doc = meta_data['doc_token_ids'].squeeze().tolist()
    with open(os.path.join(dir, fname), 'rb') as fd:
        ig_output = pickle.load(fd)
    ig_output = ig_output['output']
    if len(ig_output[0].size()) < 2:
        ig_output = [x.unsqueeze(0) for x in ig_output]
    whole_igs = torch.cat(ig_output).to(device)
    whole_igs = torch.abs(whole_igs)        # NOTICE
    k = max(budget_ig) * 2
    values, indicies = torch.topk(whole_igs, k=k, dim=-1)
    list_indicies = indicies.cpu().tolist()
    seq_len = len(output_base_data)
    return_inputs = []
    return_prefix = []
    return_tgt = []
    return_meta = []
    for idx in range(seq_len):
        # we only care about no lm tokens
        distance_imp_full = output_base_data[idx]['imp_full']
        if distance_imp_full <= distance_thres:
            continue
        return_meta.append({
            'prefix': output_base_data[idx]['prefix'],
            'token': output_base_data[idx]['token'],
            'pos': output_base_data[idx]['pos']})
        max_indicies = list_indicies[idx]

        prefix = output_base_data[idx]['prefix_token_ids']
        prefix = prefix.repeat(len(budget_ig), 1)
        return_prefix.append(prefix)
        tgt_token_id = output_base_data[idx]['tgt_token_id']
        return_tgt.append([tgt_token_id] * len(budget_ig))
        temp_group = []
        for bud in budget_ig:
            inp_exp = yield_input_seq(bud, list_doc, max_indicies)
            temp_group.append(inp_exp)
        temp_group = [tokenizer.decode(y) for y in temp_group]
        batch_inp = tokenizer.prepare_seq2seq_batch(src_texts=temp_group)
        return_inputs.append(batch_inp)
    return (return_inputs, return_prefix, return_tgt, return_meta)


if __name__ == "__main__":

    debug = False
    parser = common_args()
    args = parser.parse_args()
    logger.info(args)
    device = args.device
    args = fix_args(args)
    if debug:
        args.device = 'cpu'
        args.mname_sum = 'sshleifer/distilbart-xsum-6-6'
    result_file = '/mnt/data0/jcxu/eval.txt'
    # # init BART models
    if 'sent' in args.eval_mode:
        budget_ig = [0, 1, 2, 3, 4]
    else:
        budget_ig = [0, 1, 2, 4, 8, 16]

    _, model_sum, _, tokenizer = init_bart_family(
        args.mname_lm, args.mname_sum, device, no_lm=True, no_ood=True)
    logger.info("Done loading BARTs.")
    model_pkg = {'sum': model_sum, 'tok': tokenizer}

    all_files_base = os.listdir(args.dir_base)
    if args.task in ['random', 'lead']:
        viable_files = all_files_base
    else:
        all_files_eval = os.listdir(args.dir_task)
        viable_files = list(set(all_files_base) & set(all_files_eval))
    viable_files = viable_files[:500]
    all_result = []
    try:
        for f in viable_files:
            outputs = []
            uid = f.split('.')[0]
            output_base_data = load_pickle(args.dir_base, f)
            step_data, meta_data = read_meta_data(args.dir_meta, f)
            if args.task in ['int_grad', 'inp_grad']:
                task_output = load_pickle(args.dir_task, f)
                pack_data_inp = extract_from_task_output(
                    task_output, meta_data, step_data, args, budget_ig, device)
            elif args.task in ['random', 'lead']:
                pack_data_inp = extract_from_baseline(
                    meta_data, step_data, args, budget_ig, device)
            data_gen = batch_data(pack_data_inp, tokenizer, device)
            if random.random() < 0.1:
                print("working ...")
            this_result = run_eval(model_pkg['sum'], data_gen)

            all_result += this_result
    except KeyboardInterrupt:
        logger.info(f"Done {len(all_result)}")
    one_line_out = summarize_result(all_result, args)
    with open(result_file, 'a') as fd:
        ct = time.strftime('%m%d%H')
        fd.write('\n')
        fd.write(ct + ","+one_line_out)
