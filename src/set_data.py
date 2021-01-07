# The main file. We are going to get source attribution and content attribution from this file.
# After that, we can analyze the data and compile all the numbers

import string

from helper import get_summ_prefix
from util import *

from helper import get_sum_data
from helper_run_bart import (
    write_pkl_to_disk, init_spacy, extract_tokens, init_bart_family,gen_original_summary)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-model_family", default='bart')
    parser.add_argument("-data_name", default='xsum', help='name of dataset')
    parser.add_argument("-mname_lm", default='facebook/bart-large')
    parser.add_argument("-mname_sum", default='facebook/bart-large-xsum')
    parser.add_argument('-truncate_sent', default=20,
                        help='the max sent used for perturbation')
    parser.add_argument('-dir_save', default="/mnt/data0/jcxu/meta_data_ref",
                        help="The location to save output data. ")
    parser.add_argument("-device", help="device to use", default='cuda:0')
    args = parser.parse_args()
    logger.info(args)
    device = args.device
    
    if not os.path.exists(args.dir_save):
        os.makedirs(args.dir_save)

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
                 'ood': model_sum_ood, 'tok': tokenizer,'spacy':sp_nlp}
    # {Do some perturbation to one example, run the model again, check if the token exist, write the result on the disk}
    feat_array = []
    model_prediction = []
    modified_text = []
    return_data = []

    for data_point in dev_data:
        document = data_point['document']
        ref_summary = data_point['summary']
        uid = data_point['id']

        document_sents = document.split('\n')[:args.truncate_sent]
        document = "\n".join(document_sents)
        input_doc = tokenizer(document, return_tensors='pt')
        pred_summary = gen_original_summary(
            model_pkg['sum'], model_pkg['tok'], document, device)[0].strip()    # best summary from the model with beam search

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
            'meta': {'document': document,
                     'doc_token_ids': input_doc['input_ids'],
                     'ref': ref_summary,
                     'summary': pred_summary,
                     'id': uid} }

        write_pkl_to_disk(args.dir_save, fname_prefix=uid, data_obj=final)

    logger.info('Done Collecting data ...')
