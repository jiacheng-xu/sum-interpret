# Entrance for ensembling (src attribution) for distributions

# import random
# import torch
# from nltk.corpus import stopwords
# import nltk
# from typing import List
# import string
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import logging
# import sys
# from src.func_model import init_bart_sum_model, init_bart_lm_model, run_model, tokenize_text, run_lm, run_explicit, run_implicit, run_attn, run_full_model
# from src.lime_helper import train_dt
# from lib.lm_feat_supp import map_tok_bigram_past, map_tok_bigram_future, map_tok_sent_doc, reg_tokenize

from util import *
from src.helper import get_sum_data
from src.helper_run_bart import init_bart_sum_model, init_bart_lm_model

parser = argparse.ArgumentParser()
parser.add_argument("device", help="device to use", default='cuda:1')
parser.add_argument("data_name", default='xsum', help='name of dataset')
parser.add_argument("mname_lm",default='facebook/bart-large')
parser.add_argument("mname_sum",default='facebook/bart-large-xsum')

args = parser.parse_args()


def init_bart_family(name_lm, name_sum):
    init_bart_sum_model()
    init_bart_lm_model()

def init_lime():
    pass


def init_ig():
    pass


if __name__ == '__main__':
    # Run a PEGASUS/BART model to explain the local behavior
    # Sample one article from datasets
    dev_data = get_sum_data(args.data_name)
    train_data = get_sum_data(args.data_name, split='train')
    device = args.device

    auto_data_collect = False

    # init models
    init_bart_family(args.mname_lm, args.mname_sum)
    # Run the model and input a token of interest
    model, tokenizer = init_bart_sum_model(device=device)
    lm, _ = init_bart_lm_model(device=device)
    batch_size = 40
    max_sample = 400
    # {Do some perturbation to one example, run the model again, check if the token exist, write the result on the disk}
    feat_array = []
    model_prediction = []
    modified_text = []
    return_data = []
    total_cnt = 10000
    cnt = 0
    try:
        for data_point in dev_data:
            logging.info(f"Example: {data_point['document'][:6000]}")
            new_example = True
            raw_document = data_point['document']
            token_ids, doc_str, doc_str_lower = tokenize_text(
                tokenizer, raw_document)

            raw_output, _ = run_model(
                model, tokenizer, [raw_document], device=device)
            raw_output_summary = raw_output[0].strip()
            logging.info(f"Model Output: {raw_output_summary}")

            while True:
                if auto_data_collect:
                    if new_example:
                        raw_out_proc = raw_output[0].strip()
                        tokens = raw_out_proc.split(' ')
                        while True:
                            interest = random.choice(tokens[1:])
                            sos = tokens[0]
                            if sos.lower().startswith(interest.lower()):
                                continue
                            else:
                                break
                        logging.info(f"Interest: {interest}")
                        new_example = False
                    else:
                        break
                else:
                    interest = input("Please enter a word of interest:\n")
                    # interest = 'clarification'
                    if 'q' == interest:
                        break

                lower_interest = interest.lower().strip()
                if lower_interest not in doc_str_lower:
                    logging.warning(
                        f"Input token {interest} not found in the input document.")

                summary_prefix = get_summ_prefix(
                    tgt_token=lower_interest, raw_output_summary=raw_output_summary)

                logging.info("Run Summ model...")

                sum_model_output, p_sum = run_full_model(
                    model, tokenizer, [raw_document], device=device, sum_prefix=summary_prefix, output_dec_hid=True)

                lm_output, p_lm = run_lm(lm, tokenizer, device, summary_prefix)

                implicit_output, p_implicit = run_implicit(
                    model, tokenizer, sum_prefix=summary_prefix, device=device)

                most_attn, attn_distb = run_attn(
                    model, tokenizer, [raw_document], sum_prefix=summary_prefix, device=device)

                record = {}
                record['p_lm'] = p_lm.detach().cpu()
                record['p_imp'] = p_implicit.detach().cpu()
                record['p_attn'] = attn_distb.detach().cpu()
                record['p_full'] = p_sum.detach().cpu()

                # staple decoder hidden states
                dec_hid_states = [x[0, -1].detach().cpu()
                                  for x in sum_model_output['decoder_hidden_states']]
                record['dec_hid'] = dec_hid_states
                record['meta'] = {
                    'document': raw_document,
                    'prefix': summary_prefix,
                    'output': raw_output_summary,
                    'ref': data_point['summary'],
                    'id': data_point['id'],
                    'query': interest
                }
                return_data.append(record)
                logging.info(f"Len:{len(return_data)}")
                cnt += 1
            if cnt >= total_cnt:
                break
    except KeyboardInterrupt:
        logging.info('Done Collecting data ...')
    # if len(feat_array) != len(model_prediction):
    #     minlen = min(len(feat_array), len(model_prediction))
    #     feat_array = feat_array[:minlen]
    #     model_prediction = model_prediction[:minlen]
    # assert len(feat_array) == len(model_prediction)
    logging.info(f"Len:{len(return_data)}")
    import pickle
    with open(f"xsum_run_record_{len(return_data)}.pkl", 'wb') as fd:
        pickle.dump(return_data, fd)
