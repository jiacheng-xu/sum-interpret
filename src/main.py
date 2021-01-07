# The main file. We are going to get source attribution and content attribution from this file.
# After that, we can analyze the data and compile all the numbers

import string


from helper import get_summ_prefix
from util import *

from helper import get_sum_data
from helper_run_bart import (gen_original_summary, init_bart_lm_model,
                             init_bart_sum_model, tokenize_text, extract_tokens, run_full_model, run_lm, init_spacy, run_attn, write_pkl_to_disk, run_full_model_slim)


def init_lime():
    pass


def init_ig():
    pass


def _step_src_attr(input_ids, prefix_ids, summary_prefix: str, document_sents: List[str], model_pkg, device):
    # print(f"start:{start_matching_index}\n{summary}")
    # summary_prefix = get_summ_prefix(
    #     tgt_token=interest.strip(), raw_output_summary=summary, start_matching_index=start_matching_index)
    # new_start_match = len(summary_prefix) + len(interest)
    # print(summary_prefix)
    if not summary_prefix:
        summary_prefix = model_pkg['tok'].bos_token

    batch_size = input_ids.size()[0]
    implicit_input = torch.LongTensor(
        [[0, 2] for _ in range(batch_size)]).to(device)
    # sum_model_output, p_sum = run_full_model(model_pkg['sum'], model_pkg['tok'], [document], device=device, sum_prefix=[summary_prefix], output_dec_hid=True)
    _, p_sum, _ = run_full_model_slim(
        model=model_pkg['sum'], input_ids=input_ids, attention_mask=None, decoder_input_ids=prefix_ids, targets=None
    )
    # perturbation document_sents
    num_perturb_sent = len(document_sents)
    sum_model_output_pert, p_sum_pert = run_full_model(
        model_pkg['sum'], model_pkg['tok'], document_sents, device=device, sum_prefix=[summary_prefix] * num_perturb_sent, output_dec_hid=False)

    lm_output_topk, p_lm = run_lm(
        model_pkg['lm'], model_pkg['tok'], device=device, sum_prefix=summary_prefix)
    # lm_output_topk is a list of tokens

    # implicit_output, p_implicit = run_implicit(model_pkg['sum'], model_pkg['tok'], sum_prefix=summary_prefix, device=device)
    _, p_implicit, _ = run_full_model_slim(
        model_pkg['sum'], implicit_input, None, prefix_ids, None, device=device)

    # Out of domain model
    # implicit_ood_output, p_implicit_ood = run_implicit(model_pkg['ood'], model_pkg['tok'], sum_prefix=summary_prefix, device=device)
    _, p_implicit_ood, _ = run_full_model_slim(
        model_pkg['ood'], implicit_input, None, prefix_ids, None, device)

    _, p_ood, _ = run_full_model_slim(
        model_pkg['ood'],input_ids=input_ids, attention_mask=None, decoder_input_ids=prefix_ids, targets=None)

    most_attn, attn_distb = run_attn(
        model_pkg['sum'], input_ids=input_ids, prefix_ids=prefix_ids, device=device)

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
    # dec_hid_states = [x[0, -1].detach().cpu()
    #   for x in sum_model_output['decoder_hidden_states']]
    # record['dec_hid'] = dec_hid_states
    # record['prefix'] = summary_prefix
    # record['interest'] = interest
    return record


# The entrance for one document, more stuff to unpack


@dec_print_wrap
def src_attribute(step_data: List, input_doc_ids: torch.Tensor, document_str: str, uid: str, model_pkg: dict, device):
    """Source Attribution"""
    # input_doc_ids: [1, 400]

    # doc_tok_ids, doc_str, doc_str_lower = tokenize_text(
    #     model_pkg['tok'], document)
    # logger.debug(f"Example: {doc_str[:600]} ...")

    """
    pred_summary = gen_original_summary(
        model_pkg['sum'], model_pkg['tok'], document, device)[0].strip()    # best summary from the model with beam search
    
    logger.info(f"Model output summary: {pred_summary}")
    """

    document_sents = document_str.split("\n")
    for idx, step in enumerate(step_data):
        prefix_token_ids = step['prefix_token_ids']
        prefix = step['prefix']
        tgt_token = step['tgt_token']
        record = _step_src_attr(input_ids=input_doc_ids, prefix_ids=prefix_token_ids,
                                summary_prefix=prefix, document_sents=document_sents, model_pkg=model_pkg, device=device)

        # 'prefix': summary_prefix,
        # 'query': interest
        outputs.append(record)

    # outputs.pop(-1)  # remove the EOS punct
    # final = {
    #     'data': outputs
    # }
    return outputs


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
    parser.add_argument('-dir_meta', default='/mnt/data0/jcxu/meta_data_ref')
    parser.add_argument('-dir_save', default="/mnt/data0/jcxu/output_base",
                        help="The location to save output data. ")
    args = parser.parse_args()
    logger.info(args)
    if not os.path.exists(args.dir_save):
        os.makedirs(args.dir_save)

    # Run a PEGASUS/BART model to explain the local behavior
    # Sample one article from datasets

    device = args.device

    # init BART models

    model_lm, model_sum, model_sum_ood, bart_tokenizer = init_bart_family(
        args.mname_lm, args.mname_sum, device)
    logger.info("Done loading BARTs.")
    # sp_nlp = init_spacy()
    model_pkg = {'lm': model_lm, 'sum': model_sum, 'ood': model_sum_ood, 'tok': bart_tokenizer}

    # {Do some perturbation to one example, run the model again, check if the token exist, write the result on the disk}

    feat_array = []
    model_prediction = []
    modified_text = []
    return_data = []
    all_meta_files = os.listdir(args.dir_meta)
    try:
        for f in all_meta_files:
            outputs = []
            step_data, meta_data = read_meta_data(args.dir_meta, f)
            uid = meta_data['id']
            document = meta_data['document']
            doc_token_ids = meta_data['doc_token_ids'].to(device)
            return_data = src_attribute(step_data, doc_token_ids,
                          document, uid, model_pkg, device)

            write_pkl_to_disk(args.dir_save, fname_prefix=uid,
                              data_obj=return_data)
    except KeyboardInterrupt:
        logger.info('Done Collecting data ...')
