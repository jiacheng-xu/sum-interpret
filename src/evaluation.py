# batch evaluation
from util import *
from main import init_bart_family
from helper_run_bart import run_full_model_slim
budget_ig = [0, 1, 2, 4]


def yield_input_seq(bud, list_doc, topk_indicies, context_window=2, sep_token_id=6):
    # .=4  ,=6  space=1437
    return_bpes = []
    cursor = 0
    max_doc_len = len(list_doc) - 1  # <eos>
    min_doc_len = 1  # <s>
    while bud > 0:
        top = topk_indicies[cursor]
        cursor += 1

        left, right = max(
            min_doc_len, top-context_window), min(top + context_window, max_doc_len)
        if return_bpes:
            span_bpe = [sep_token_id]+list_doc[left:right]
        else:
            span_bpe = list_doc[left:right]
        return_bpes += span_bpe
        bud -= 1
    return return_bpes


def run_eval(docs, prefixs, tgts, metas):
    for d, p, t, m in zip(docs, prefixs, tgts,metas):
        
        output_tok, output_prob, loss = run_full_model_slim(
            model_sum,
            input_ids=torch.LongTensor(d['input_ids']),
            attention_mask=torch.tensor(d['attention_mask']), decoder_input_ids=p,
            targets=torch.LongTensor(t),
            device=device)
        list_d = d['input_ids']
        print("---------------")
        for idx, idk in enumerate(list_d) :
            print( f"Loss:{loss[idx]}\tInput:{tokenizer.decode(idk)}\tOutput:{output_tok[idx]}" )
        print(m)
        # print(output_tok, loss)
        print('')
        


def prepare_int_grad(dir, fname, step_data, meta_data, device, context_window=2):
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
    seq_len = len(step_data)
    return_groups = [[] for _ in range(seq_len)]
    return_prefix = [None for _ in range(seq_len)]
    return_tgt = [None for _ in range(seq_len)]
    return_meta = []
    for idx in range(seq_len):
        return_meta.append({
            'prefix':step_data[idx]['prefix'],
            'token':step_data[idx]['token'],
            'pos':step_data[idx]['pos'],
        }
            )
        max_indicies = list_indicies[idx]

        prefix = step_data[idx]['prefix_token_ids']
        prefix = prefix.repeat(len(budget_ig), 1)
        return_prefix[idx] = prefix
        tgt_token_id = step_data[idx]['tgt_token_id_backup']
        return_tgt[idx] = [tgt_token_id] * len(budget_ig)
        temp_group = []
        for bud in budget_ig:
            inp_exp = yield_input_seq(bud, list_doc, max_indicies)
            temp_group.append(inp_exp)
        temp_group = [tokenizer.decode(y) for y in temp_group]
        batch_inp = tokenizer.prepare_seq2seq_batch(src_texts=temp_group)
        return_groups[idx] = batch_inp
    return (return_groups, return_prefix, return_tgt, return_meta)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-device", help="device to use", default='cuda:0')
    parser.add_argument("-data_name", default='xsum', help='name of dataset')
    parser.add_argument("-mname_lm", default='facebook/bart-large')
    parser.add_argument("-mname_sum", default='facebook/bart-large-xsum')
    parser.add_argument("-batch_size", default=40)
    parser.add_argument('-max_samples', default=1000)
    parser.add_argument('-dir_save', default="/mnt/data0/jcxu/interpret_output",
                        help="The location to save output data. ")
    parser.add_argument('-task', default='ig',
                        help="The model to evaluate, including ig, random, lime, occ, ")
    parser.add_argument('-dir_ig', default='/mnt/data0/jcxu/output_ig')
    parser.add_argument('-dir_meta', default='/mnt/data0/jcxu/meta_data_ref')
    args = parser.parse_args()
    logger.info(args)
    device = args.device

    # init BART models
    _, model_sum, _, tokenizer = init_bart_family(
        args.mname_lm, args.mname_sum, device, no_lm=True, no_ood=True)
    logger.info("Done loading BARTs.")
    model_pkg = {'sum': model_sum, 'tok': tokenizer}

    if args.task == 'ig':
        dir_read = args.dir_ig
    else:

        raise NotImplementedError
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    all_files = os.listdir(dir_read)
    for f in all_files:
        outputs = []
        step_data, meta_data = read_meta_data(args.dir_meta, f)
        uid = meta_data['id']
        (return_groups, return_prefix, return_tgt, return_meta) = prepare_int_grad(dir=dir_read, fname=f,
                                                                      step_data=step_data, meta_data=meta_data, device=device)
        run_eval(return_groups, return_prefix, return_tgt,return_meta)
