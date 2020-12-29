# Entrance for ensembling (src attribution) for distributions

from helper import get_summ_prefix
from util import *

from helper import get_sum_data
from helper_run_bart import (gen_original_summary, init_bart_lm_model,
                             init_bart_sum_model, tokenize_text, extract_tokens, run_full_model, run_lm, init_spacy, run_implicit, run_attn, write_pkl_to_disk)


def init_bart_family(name_lm, name_sum, device):
    lm_model, tok = init_bart_lm_model(name_lm, device)

    sum_model, _ = init_bart_sum_model(name_sum, device)
    sum_out_of_domain, _ = init_bart_sum_model(
        "facebook/bart-large-cnn", device)
    return lm_model, sum_model, sum_out_of_domain, tok


def init_lime():
    pass


def init_ig():
    pass


def _step_src_attr(interest: str, summary: str, document: str, document_sents: List[str], model_pkg, device, start_matching_index=0):
    # print(f"start:{start_matching_index}\n{summary}")
    summary_prefix = get_summ_prefix(
        tgt_token=interest.strip(), raw_output_summary=summary, start_matching_index=start_matching_index)
    new_start_match = len(summary_prefix) + len(interest)
    # print(summary_prefix)
    if not summary_prefix:
        summary_prefix = model_pkg['tok'].bos_token

    sum_model_output, p_sum = run_full_model(
        model_pkg['sum'], model_pkg['tok'], [document], device=device, sum_prefix=[summary_prefix], output_dec_hid=True)

    # perturbation document_sents
    num_perturb_sent = len(document_sents)
    sum_model_output_pert, p_sum_pert = run_full_model(
        model_pkg['sum'], model_pkg['tok'], document_sents, device=device, sum_prefix=[summary_prefix] * num_perturb_sent, output_dec_hid=False)

    lm_output_topk, p_lm = run_lm(
        model_pkg['lm'], model_pkg['tok'], device=device, sum_prefix=summary_prefix)
    # lm_output_topk is a list of tokens

    implicit_output, p_implicit = run_implicit(
        model_pkg['sum'], model_pkg['tok'], sum_prefix=summary_prefix, device=device)

    # Out of domain model!
    implicit_ood_output, p_implicit_ood = run_implicit(
        model_pkg['ood'], model_pkg['tok'], sum_prefix=summary_prefix, device=device)

    ood_model_output, p_ood = run_full_model(
        model_pkg['ood'], model_pkg['tok'], [document], device=device, sum_prefix=[summary_prefix], output_dec_hid=True)

    most_attn, attn_distb = run_attn(
        model_pkg['sum'], model_pkg['tok'], document, sum_prefix=summary_prefix, device=device)

    record = {}
    record['p_lm'] = p_lm.detach().cpu()
    record['p_imp'] = p_implicit.detach().cpu()
    record['p_attn'] = attn_distb.detach().cpu()
    record['p_full'] = p_sum.detach().cpu()
    record['p_imp_ood'] = p_implicit_ood.detach().cpu()
    record['p_full_ood'] = p_ood.detach().cpu()
    record['p_pert'] = p_sum_pert.detach().cpu()
    record['pert_sents'] = document_sents
    # staple decoder hidden states
    dec_hid_states = [x[0, -1].detach().cpu()
                      for x in sum_model_output['decoder_hidden_states']]
    record['dec_hid'] = dec_hid_states
    record['prefix'] = summary_prefix
    record['interest'] = interest
    return record, new_start_match


@dec_print_wrap
def src_attribute(document: str, summary: str, uid: str, model_pkg: dict, device, max_num_sent: int = 15):
    """Source Attribution"""
    # doc_tok_ids, doc_str, doc_str_lower = tokenize_text(
    #     model_pkg['tok'], document)
    # logger.debug(f"Example: {doc_str[:600]} ...")
    pred_summary = gen_original_summary(
        model_pkg['sum'], model_pkg['tok'], document, device)[0].strip()    # best summary from the model with beam search
    document_sents = document.split('\n')[:max_num_sent]

    logger.info(f"Model output summary: {pred_summary}")
    tokens, tags = extract_tokens(pred_summary, nlp=model_pkg['spacy'])
    outputs = []
    start_matching_index = 0
    for (tok, tag) in zip(tokens, tags):
        # one step
        record, start_matching_index = _step_src_attr(
            tok, pred_summary, document, document_sents, model_pkg, device, start_matching_index)
        record['token'] = tok
        record['pos'] = tag
        # 'prefix': summary_prefix,
        # 'query': interest
        outputs.append(record)

    outputs.pop(-1)  # remove the EOS punct
    final = {
        'data': outputs,
        'meta': {'document': document,
                 'output': pred_summary,
                 'ref': summary,
                 'id': uid}
    }
    return final


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-device", help="device to use", default='cuda:0')
    parser.add_argument("-data_name", default='xsum', help='name of dataset')
    parser.add_argument("-mname_lm", default='facebook/bart-large')
    parser.add_argument("-mname_sum", default='facebook/bart-large-xsum')
    parser.add_argument("-batch_size", default=40)
    parser.add_argument('-max_samples', default=1000)
    parser.add_argument('-truncate_sent', default=15,
                        help='the max sent used for perturbation')
    parser.add_argument('-dir_save', default="/mnt/data0/jcxu/interpret_output",
                        help="The location to save output data. ")
    args = parser.parse_args()
    logger.info(args)
    if not os.path.exists(args.dir_save):
        os.makedirs(args.dir_save)

    # Run a PEGASUS/BART model to explain the local behavior
    # Sample one article from datasets
    dev_data = get_sum_data(args.data_name)
    train_data = get_sum_data(args.data_name, split='train')
    device = args.device

    auto_data_collect = True

    # init BART models
    model_lm, model_sum, model_sum_ood, bart_tokenizer = init_bart_family(
        args.mname_lm, args.mname_sum, device)
    logger.info("Done loading BARTs.")
    sp_nlp = init_spacy()
    model_pkg = {'lm': model_lm, 'sum': model_sum, 'ood': model_sum_ood, 'tok': bart_tokenizer,
                 'spacy': sp_nlp}
    # {Do some perturbation to one example, run the model again, check if the token exist, write the result on the disk}
    feat_array = []
    model_prediction = []
    modified_text = []
    return_data = []

    try:
        for data_point in dev_data:
            document = data_point['document']
            summary = data_point['summary']
            uid = data_point['id']

            return_data = src_attribute(
                document, summary, uid, model_pkg, device, max_num_sent=args.truncate_sent)
            write_pkl_to_disk(args.dir_save, fname_prefix=uid +
                              "_p", data_obj=return_data)
    except KeyboardInterrupt:
        logger.info('Done Collecting data ...')
