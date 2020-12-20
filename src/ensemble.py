# Entrance for ensembling (src attribution) for distributions

# import random
# import torch
# from nltk.corpus import stopwords
# import nltk
# from typing import List
# import string
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import logger
# import sys
# from src.func_model import init_bart_sum_model, init_bart_lm_model, run_model, tokenize_text, run_lm, run_explicit, run_implicit, run_attn, run_full_model
# from src.lime_helper import train_dt
# from lib.lm_feat_supp import map_tok_bigram_past, map_tok_bigram_future, map_tok_sent_doc, reg_tokenize

from helper import get_summ_prefix
from util import *

from helper import get_sum_data
from helper_run_bart import (gen_original_summary, init_bart_lm_model,
                             init_bart_sum_model, tokenize_text, extract_tokens, run_full_model, run_lm, init_spacy, run_implicit, run_attn)


def init_bart_family(name_lm, name_sum, device):
    lm_model, tok = init_bart_sum_model(name_lm, device)
    sum_model, _ = init_bart_lm_model(name_sum, device)
    return lm_model, sum_model, tok


def init_lime():
    pass


def init_ig():
    pass


def _step_src_attr(interest: str, summary: str, document: str, model_pkg, device):
    summary_prefix = get_summ_prefix(
        tgt_token=interest, raw_output_summary=summary)
    if not summary_prefix:
        summary_prefix = model_pkg['tok'].bos_token

    sum_model_output, p_sum = run_full_model(
        model_pkg['sum'], model_pkg['tok'], [document], device=device, sum_prefix=summary_prefix, output_dec_hid=True)

    lm_output_topk, p_lm = run_lm(
        model_pkg['lm'], model_pkg['tok'], device=device, sum_prefix=summary_prefix)
    # lm_output_topk is a list of tokens

    implicit_output, p_implicit = run_implicit(
        model_pkg['sum'], model_pkg['tok'], sum_prefix=summary_prefix, device=device)

    most_attn, attn_distb = run_attn(
        model_pkg['sum'], model_pkg['tok'], [document], sum_prefix=summary_prefix, device=device)

    record = {}
    record['p_lm'] = p_lm.detach().cpu()
    record['p_imp'] = p_implicit.detach().cpu()
    record['p_attn'] = attn_distb.detach().cpu()
    record['p_full'] = p_sum.detach().cpu()

    # staple decoder hidden states
    dec_hid_states = [x[0, -1].detach().cpu()
                      for x in sum_model_output['decoder_hidden_states']]
    record['dec_hid'] = dec_hid_states
    return record


@dec_print_wrap
def src_attribute(document: str, summary: str, uid: str, model_pkg: dict, device):
    # doc_tok_ids, doc_str, doc_str_lower = tokenize_text(
    #     model_pkg['tok'], document)
    # logger.debug(f"Example: {doc_str[:600]} ...")
    pred_summary = gen_original_summary(
        model_pkg['sum'], model_pkg['tok'], document, device)[0].strip()
    logger.info(f"Model output summary: {pred_summary}")
    tokens, tags = extract_tokens(pred_summary, nlp=model_pkg['spacy'])
    for (tok, tag) in zip(tokens, tags):
        # one step
        record = _step_src_attr(tok, pred_summary, document, model_pkg, device)
        print("rn")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-device", help="device to use", default='cuda:1')
    parser.add_argument("-data_name", default='xsum', help='name of dataset')
    parser.add_argument("-mname_lm", default='facebook/bart-large')
    parser.add_argument("-mname_sum", default='facebook/bart-large-xsum')
    parser.add_argument("-batch_size", default=40)
    parser.add_argument('-max_samples', default=1000)
    parser.add_argument('-dir_save', default="/mnt/data0/jcxu",
                        help="The location to save output data. ")
    args = parser.parse_args()

    # Run a PEGASUS/BART model to explain the local behavior
    # Sample one article from datasets
    dev_data = get_sum_data(args.data_name)
    train_data = get_sum_data(args.data_name, split='train')
    device = args.device

    auto_data_collect = True

    # init BART models
    model_lm, model_sum, bart_tokenizer = init_bart_family(
        args.mname_lm, args.mname_sum, device)
    logger.info("Done loading BARTs.")
    sp_nlp = init_spacy()
    model_pkg = {'lm': model_lm, 'sum': model_sum, 'tok': bart_tokenizer,
                 'spacy': sp_nlp}
    # {Do some perturbation to one example, run the model again, check if the token exist, write the result on the disk}
    feat_array = []
    model_prediction = []
    modified_text = []
    return_data = []
    total_cnt = 10000
    cnt = 0
    try:
        for data_point in dev_data:
            document = data_point['document']
            summary = data_point['summary']
            uid = data_point['id']
            src_attribute(document, summary, uid, model_pkg, device)
    except KeyboardInterrupt:
        logger.info('Done Collecting data ...')
