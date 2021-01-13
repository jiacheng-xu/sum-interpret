# The main file. We are going to get source attribution and content attribution from this file.
# After that, we can analyze the data and compile all the numbers

import string

from helper import get_summ_prefix
from util import *
import itertools
from typing import List

from helper import get_sum_data
from helper_run_bart import (
    write_pkl_to_disk, init_spacy, extract_tokens, init_bart_family, gen_original_summary)


def truncate_document(document: str, tokenizer, max_sent_num: int = 15, max_tok_num: int = 70, sent_sep_token_id=50118):
    document_sents = document.split('\n')[:max_sent_num]
    sent_tok_ids: List[List[int]] = []
    sent_toks: List[str] = []

    total_len = 0
    for sent in document_sents:
        toks = tokenizer.encode(
            sent, add_special_tokens=False, max_length=max_tok_num, truncation=True)
        recovered_text: str = tokenizer.decode(toks)
        sent_tok_ids.append(toks)
        sent_toks.append(recovered_text)
        total_len += len(toks)
        if total_len > 600:
            break
    # sent_tok_ids: List[List[int]]
    # input_doc_token_ids: List[int]
    input_doc_token_ids = [0]
    map_of_sent_idx = [0]
    for idx, sent_t in enumerate(sent_tok_ids):
        if idx != 0:
            input_doc_token_ids += [sent_sep_token_id]
            map_of_sent_idx += [idx]

        input_doc_token_ids += sent_t
        l = len(sent_t)
        map_of_sent_idx += [idx] * l
    input_doc_token_ids += [2]
    map_of_sent_idx += [idx]
    return input_doc_token_ids, "\n".join(sent_toks), sent_tok_ids, sent_toks, map_of_sent_idx


if __name__ == '__main__':
    parser = common_args()

    args = parser.parse_args()
    logger.info(args)
    device = args.device
    args = fix_args(args)

    if not os.path.exists(args.dir_meta):
        os.makedirs(args.dir_meta)

    # Run a PEGASUS/BART model to explain the local behavior
    # Sample one article from datasets
    dev_data = get_sum_data(args.data_name)
    # train_data = get_sum_data(args.data_name, split='train')

    # init BART models
    if args.model_family == 'bart':
        model_lm, model_sum, model_sum_ood, tokenizer = init_bart_family(
            args.mname_lm, args.mname_sum, device, no_lm=True, no_ood=True)
    else:
        raise NotImplementedError
    sp_nlp = init_spacy()
    model_pkg = {'lm': model_lm, 'sum': model_sum,
                 'ood': model_sum_ood, 'tok': tokenizer, 'spacy': sp_nlp}

    cnt = 0
    for data_point in dev_data:
        document = data_point['document']
        ref_summary = data_point['summary']
        uid = data_point['id']
        input_doc_token_ids, input_doc_str, sent_tok_ids, sent_list_str, map_tok_to_sent_idx = truncate_document(
            document, tokenizer, args.truncate_sent, max_tok_num=args.truncate_word)
        if len(input_doc_str)< 30:
            continue
        pred_summary = gen_original_summary(
            model_pkg['sum'], model_pkg['tok'], input_doc_str, device)[0].strip()    # best summary from the model with beam search

        logger.info(f"Model output summary: {pred_summary}")
        print(f"Model output summary:{pred_summary}")
        # BUT we are going to use ref summary!!
        tokens, tags = extract_tokens(pred_summary, nlp=model_pkg['spacy'])
        outputs = []
        start_matching_index = 0
        for (tok, tag) in zip(tokens, tags):
            # one step
            summary_prefix = get_summ_prefix(
                tgt_token=tok.strip(), raw_output_summary=pred_summary, start_matching_index=start_matching_index)
            start_matching_index = len(summary_prefix) + len(tok.strip())
            if tok.strip() in string.punctuation:
                continue
            record = {}
            if summary_prefix.endswith(" "):
                summary_prefix = summary_prefix[:-1]
                add_space = True
            else:
                add_space = False
            if add_space:
                target_word_bpe = model_pkg['tok'].encode(" "+tok)[1]
            else:
                target_word_bpe = model_pkg['tok'].encode(tok)[1]

            prefix_token_ids = tokenizer(summary_prefix, return_tensors='pt',)

            prefix_token_ids = prefix_token_ids['input_ids'][:, :-1]
            record['tgt_token_id'] = target_word_bpe
            record['tgt_token'] = model_pkg['tok'].convert_ids_to_tokens(
                target_word_bpe)
            record['prefix'] = summary_prefix
            record['prefix_token_ids'] = prefix_token_ids
            record['token'] = tok
            record['pos'] = tag
            outputs.append(record)

        outputs.pop(-1)  # remove the EOS punct
        final = {
            'data': outputs,
            'meta': {'document': input_doc_str,
                     'doc_token_ids': input_doc_token_ids,
                     'ref': ref_summary,
                     'summary': pred_summary,
                     'sent_token_ids': sent_tok_ids,
                     'sent_text': sent_list_str,
                     'map_index': map_tok_to_sent_idx,
                     'id': uid}}

        write_pkl_to_disk(args.dir_meta, fname_prefix=uid, data_obj=final)
        cnt += 1
        if cnt > args.max_example * 10:
            logger.info(f"Early stop collecting {cnt}")
            break
    logger.info('Done Collecting data ...')
