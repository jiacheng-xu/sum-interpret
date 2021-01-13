# batch evaluation
from util import *
from helper_run_bart import run_full_model_slim, init_bart_family
from eval_util import *
budget_ig = [0, 1, 2, 4]





def data_loader():
    # gather the same length decoder prefix inputs, batch the input document
    # sort the length of
    pass


def run_eval(model_sum, docs, prefixs, tgts, metas):
    result_bag = []
    for d, p, t, m in zip(docs, prefixs, tgts, metas):
        output_tok, output_prob, loss = run_full_model_slim(
            model_sum,
            input_ids=torch.LongTensor(d['input_ids']),
            attention_mask=torch.tensor(d['attention_mask']), decoder_input_ids=p,
            targets=torch.LongTensor(t),
            device=device)
        list_d = d['input_ids']
        loss = loss.tolist()
        result_bag.append(loss)
        for idx, idk in enumerate(list_d):
            print(
                f"Loss:{loss[idx]}\tInput:{tokenizer.decode(idk)}\tOutput:{output_tok[idx]}")
        print('')

    return result_bag


def prepare_random(output_base_data, meta_data, device, context_window=2, distance_thres=0.5, is_lead=False):
    # prepare: input explanation, prefix, and gt
    list_doc = meta_data['doc_token_ids'].squeeze().tolist()

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

        prefix = output_base_data[idx]['prefix_token_ids']
        prefix = prefix.repeat(len(budget_ig), 1)
        return_prefix.append(prefix)
        tgt_token_id = output_base_data[idx]['tgt_token_id']
        return_tgt.append([tgt_token_id] * len(budget_ig))
        temp_group = []
        for bud in budget_ig:
            inp_exp = yield_input_seq_random(bud, list_doc, is_lead)
            temp_group.append(inp_exp)
        temp_group = [tokenizer.decode(y) for y in temp_group]
        batch_inp = tokenizer.prepare_seq2seq_batch(src_texts=temp_group)
        return_inputs.append(batch_inp)
    return (return_inputs, return_prefix, return_tgt, return_meta)


def prepare_int_grad(dir, fname, output_base_data, meta_data, device, context_window=2, distance_thres=0.5):
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
    debug = True
    parser = common_args()
    args = parser.parse_args()
    logger.info(args)
    device = args.device
    args = fix_args(args)
    if debug:
        args.device = 'cpu'
        args.mname_sum = 'sshleifer/distilbart-xsum-6-6'

    # # init BART models

    _, model_sum, _, tokenizer = init_bart_family(
        args.mname_lm, args.mname_sum, device, no_lm=True, no_ood=True)
    logger.info("Done loading BARTs.")
    model_pkg = {'sum': model_sum, 'tok': tokenizer}

    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    all_files_base = os.listdir(args.dir_base)
    all_files_eval = os.listdir(args.dir_task)
    viable_files = list(set(all_files_base) & set(all_files_eval))
    viable_files = viable_files[:100]
    all_result = []
    try:
        for f in viable_files:
            outputs = []
            uid = f.split('.')[0]
            output_base_data = load_pickle(args.dir_base, f)
            step_data, meta_data = read_meta_data(args.dir_meta, f)
            if args.task == 'random':
                (return_groups, return_prefix, return_tgt, return_meta) = prepare_random(
                    output_base_data=output_base_data, meta_data=meta_data,
                    device=device, is_lead=True)
            elif args.task == 'int_grad':
                (return_groups, return_prefix, return_tgt, return_meta) = prepare_int_grad(
                    dir=args.dir_task, fname=f, output_base_data=output_base_data, meta_data=meta_data,
                    device=device)
            elif args.task == 'inp_grad':
                (return_groups, return_prefix, return_tgt, return_meta) = prepare_inp_grad(
                    dir=args.dir_task, fname=f, output_base_data=output_base_data, meta_data=meta_data,
                    device=device)
            else:
                raise NotImplementedError
            this_result = run_eval(model_sum, return_groups,
                                   return_prefix, return_tgt, return_meta)
            all_result += this_result
    except KeyboardInterrupt:
        logger.info(f"Done {len(all_result)}")
    # print(all_result)
    print(len(all_result))
    groups = [[] for _ in range(len(budget_ig))]
    print_list = []
    for idx, g in enumerate(groups):
        collect = [point[idx] for point in all_result]
        collect_mean = statistics.mean(collect)
        print_list.append(str(collect_mean))
    print("\t".join(print_list))
