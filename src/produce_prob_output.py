# The main file. We are going to get source attribution and content attribution from this file.
# After that, we can analyze the data and compile all the numbers

from helper_run_bart import (gen_original_summary, init_bart_lm_model,
                             init_bart_sum_model, tokenize_text, extract_tokens, run_full_model, run_lm, init_spacy, run_attn, write_pkl_to_disk, run_full_model_slim, init_bart_family)
from helper import get_sum_data
from util import *
from helper import *
import string
import torch.nn.functional as F

# anneal_options = [0.9 ** i for i in range(5)] + [1.] + [1.1 ** i for i in range(5)]


def fix_logit(inp_logit, distb_fix, truncate_vocab_size=50264):
    inp_logit = inp_logit.squeeze()
    # inp_logit = inp_logit.unsqueeze(0)
    inp_logit = inp_logit[:truncate_vocab_size]
    mapped_prob = torch.matmul(inp_logit, distb_fix)
    mapped_prob = mapped_prob.unsqueeze(0)
    return mapped_prob


def compute_group_dyna_logits(logits, logit_signatures, distb_fix):

    logits = [fix_logit(x, distb_fix) for x in logits]
    softmax_version = [F.softmax(x) for x in logits]
    anneal_version = [torch.cat([F.softmax(x/temp)
                                 for temp in anneal_options]).unsqueeze(0) for x in logits]    # batch, 50264
    src = []
    src_sig = []
    l = len(softmax_version)
    for idx in range(l):
        src += [anneal_version[idx]] * (l-1)
        src_sig += [logit_signatures[idx]] * (l-1)
    tgt = []
    tgt_sig = []
    for idx in range(l):
        for jdx in range(l):
            if idx == jdx:
                continue
            tgt += [softmax_version[jdx]]
            tgt_sig += [logit_signatures[jdx]]
    src = torch.cat(src)    # 12, 11, 50264
    tgt = torch.cat(tgt)
    tgt = tgt.unsqueeze(1)  # 12, 1, 50264
    tgt = tgt.repeat((1, len(anneal_options), 1))
    s = torch.sum(torch.abs(src - tgt), dim=-1)
    min_s = torch.min(s, dim=-1)
    min_s = min_s[0].cpu().tolist()  # 12

    name = [f"{x}2{y}" for (x, y) in zip(src_sig, tgt_sig)]
    # Create a zip object from two lists
    zipbObj = zip(name, min_s)
    # Create a dictionary from zip object
    dictOfWords = dict(zipbObj)
    return dictOfWords


def _step_src_attr(input_ids, prefix_ids, summary_prefix: str, document_sents: List[str], model_pkg, device):
    # print(f"start:{start_matching_index}\n{summary}")
    # summary_prefix = get_summ_prefix(
    #     tgt_token=interest.strip(), raw_output_summary=summary, start_matching_index=start_matching_index)
    # new_start_match = len(summary_prefix) + len(interest)
    # print(summary_prefix)
    if not summary_prefix:
        summary_prefix = model_pkg['tok'].bos_token
    input_ids = input_ids[:, :500]
    batch_size = input_ids.size()[0]
    implicit_input = torch.LongTensor(
        [[0, 2] for _ in range(batch_size)]).to(device)
    # sum_model_output, p_sum = run_full_model(model_pkg['sum'], model_pkg['tok'], [document], device=device, sum_prefix=[summary_prefix], output_dec_hid=True)
    _, p_full, logit_full, _ = run_full_model_slim(model=model_pkg['sum'], input_ids=input_ids, attention_mask=None, decoder_input_ids=prefix_ids, targets=None, device=device
    )
    # perturbation document_sents
    num_perturb_sent = len(document_sents)
    sum_model_output_pert, p_sum_pert = run_full_model(model_pkg['sum'], model_pkg['tok'], document_sents, device=device, sum_prefix=[summary_prefix] * num_perturb_sent, output_dec_hid=False)

    lm_output_topk, p_lm, logit_lm = run_lm(model_pkg['lm'], model_pkg['tok'], device=device, sum_prefix=summary_prefix)
    # lm_output_topk is a list of tokens

    # implicit_output, p_implicit = run_implicit(model_pkg['sum'], model_pkg['tok'], sum_prefix=summary_prefix, device=device)
    _, p_imp, logit_imp, _ = run_full_model_slim(
        model_pkg['sum'], implicit_input, None, prefix_ids, None, device=device, T=0.7)

    # Out of domain model
    # implicit_ood_output, p_implicit_ood = run_implicit(model_pkg['ood'], model_pkg['tok'], sum_prefix=summary_prefix, device=device)
    _, p_imp_ood, logit_imp_ood, _ = run_full_model_slim(
        model_pkg['ood'], implicit_input, None, prefix_ids, None, device, T=0.7)

    # _, p_ood, _ = run_full_model_slim(model_pkg['ood'], input_ids=input_ids, attention_mask=None, decoder_input_ids=prefix_ids, targets=None,device=device)

    most_attn, p_attn = run_attn(
        model_pkg['sum'], input_ids=input_ids, prefix_ids=prefix_ids, device=device)

    p_lm = fix_distribution(p_lm, distb_fix, device=device)
    p_imp = fix_distribution(p_imp, distb_fix, device=device)
    p_full = fix_distribution(p_full, distb_fix, device=device)
    p_imp_ood = fix_distribution(p_imp_ood, distb_fix, device=device)
    # p_full_ood = fix_distribution(p_full_ood, distb_fix, device=device)
    # for attention, set <s> to be zero
    p_attn[:,0] = 0
    p_attn = fix_distribution(p_attn, distb_fix, device=device)
    signature = ['lm', 'imp', 'full', 'imp_cnn',  'attn']
    distributions = [p_lm, p_imp, p_full, p_imp_ood,  p_attn]

    record = compute_group_deduct(distributions, signature)
    # record = compute_group_jaccard(distributions, signature)
    # new_record = compute_group_dyna_logits(
    # [logit_lm, logit_imp, logit_full, logit_imp_ood], logit_signatures=['lm', 'imp', 'full', 'imp_cnn'], distb_fix=distb_fix)
    # record = {**record, **new_record}
    record['p_lm'] = p_lm.detach().cpu()
    record['p_imp'] = p_imp.detach().cpu()
    record['p_attn'] = p_attn.detach().cpu()
    record['p_full'] = p_full.detach().cpu()
    record['p_imp_ood'] = p_imp_ood.detach().cpu()
    # record['p_full_ood'] = p_ood.detach().cpu()
    record['p_pert'] = p_sum_pert.detach().cpu()
    record['pert_sents'] = document_sents
    pert_top1, pert_var, pert_distb = feat_perturb(
        p_sum_pert, p_full, distb_fix, device)
    record['pert_top'] = pert_top1
    record['pert_var'] = pert_var
    record['pert_distb'] = pert_distb
    top_lm = show_top_k(p_lm, summary_prefix, 'lm', tokenizer)
    top_imp = show_top_k(p_imp, summary_prefix, 'imp', tokenizer)
    top_full = show_top_k(p_full, summary_prefix, 'full', tokenizer)
    top_impood = show_top_k(p_imp_ood, summary_prefix,
                            'imp_ood', tokenizer=tokenizer)
    # top_fullood = show_top_k(p_full_ood, summary_prefix, 'full_ood', tokenizer=tokenizer)
    top_attn = show_top_k(p_attn, summary_prefix, 'attn', tokenizer)
    record['top_lm'] = top_lm
    record['top_imp'] = top_imp
    record['top_full'] = top_full
    record['top_impood'] = top_impood
    record['top_attn'] = top_attn
    return record


# The entrance for one document, more stuff to unpack


@dec_print_wrap
def src_attribute(step_data: List, input_doc_ids: torch.Tensor, document_str: str, uid: str, model_pkg: dict, device):
    """Source Attribution"""
    # input_doc_ids: [1, 400]
    pkl_outputs, csv_outputs = [], []
    csv_key = None
    # doc_tok_ids, doc_str, doc_str_lower = tokenize_text(
    #     model_pkg['tok'], document)
    # logger.debug(f"Example: {doc_str[:600]} ...")
    desired_key_for_csv = ['pert_distb', 'pert_var', 'pert_sents',
                           'lm_imp', 'imp_cnn_imp', 'imp_full',
                        #    'lm2imp', 'imp_cnn2imp', 'imp2full',
                           'token', 'pos',
                           'top_lm', 'top_imp', 'top_full', 'top_impood', 'top_attn', 't', 'T', 'prefix', 'tgt_token']

    document_sents = document_str.split("\n")
    T = len(step_data)
    for idx, step in enumerate(step_data):
        prefix_token_ids = step['prefix_token_ids']
        prefix = step['prefix']
        tgt_token = step['tgt_token']
        record = _step_src_attr(input_ids=input_doc_ids, prefix_ids=prefix_token_ids,
                                summary_prefix=prefix, document_sents=document_sents, model_pkg=model_pkg, device=device)
        # record = record | step
        record = {**record, **step}
        # 'prefix': summary_prefix,
        # 'query': interest
        record['t'] = idx
        record['T'] = T
        record['prefix'] = prefix
        record['tgt_token'] = tgt_token
        pkl_outputs.append(record)

        trim_record = {}
        for k, v in record.items():
            if k in desired_key_for_csv:
                trim_record[k] = v
        k = list(trim_record.keys())
        v = list(trim_record.values())
        csv_outputs.append(v)
        csv_key = k
    return pkl_outputs, csv_key, csv_outputs


if __name__ == '__main__':

    parser = common_args()
    args = parser.parse_args()
    logger.info(args)
    args = fix_args(args)

    # Run a PEGASUS/BART model to explain the local behavior
    # Sample one article from datasets

    device = args.device

    # init BART models

    model_lm, model_sum, model_sum_ood, tokenizer = init_bart_family(
        args.mname_lm, args.mname_sum, device)
    logger.info("Done loading BARTs.")
    # sp_nlp = init_spacy()
    model_pkg = {'lm': model_lm, 'sum': model_sum,
                 'ood': model_sum_ood, 'tok': tokenizer}
    distb_fix = init_vocab_distb_fix(tokenizer).float()
    # device = 'cuda:1'
    # device = 'cpu'
    distb_fix = distb_fix.to(device)
    # {Do some perturbation to one example, run the model again, check if the token exist, write the result on the disk}

    feat_array = []
    model_prediction = []
    modified_text = []
    return_data = []
    all_outs = []
    all_meta_files = os.listdir(args.dir_meta)
    try:
        for f in all_meta_files:
            outputs = []
            exist = check_exist_file(args.dir_base, f)
            if exist:
                logger.debug(f"{f} already exists")
                continue
            step_data, meta_data = read_meta_data(args.dir_meta, f)
            uid = meta_data['id']
            document = meta_data['document']
            doc_token_ids = meta_data['doc_token_ids'].to(device)
            return_data, csv_key, csv_v = src_attribute(step_data, doc_token_ids,
                                                        document, uid, model_pkg, device)
            all_outs += csv_v
            write_pkl_to_disk(args.dir_base, fname_prefix=uid,
                              data_obj=return_data)
            df = pd.DataFrame(csv_v, columns=csv_key)
            df.to_csv(os.path.join(args.dir_stat,uid+'.csv'))
    except KeyboardInterrupt:
        logger.info('Done Collecting data ...')

    df = pd.DataFrame(all_outs, columns=csv_key)
    agg_file = os.path.join(args.dir_stat, 'meta.csv')
    df.to_csv(agg_file)
    logger.info(f"write to {agg_file}")
