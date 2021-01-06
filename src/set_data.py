# The main file. We are going to get source attribution and content attribution from this file.
# After that, we can analyze the data and compile all the numbers

import string
from new_ig import _step_ig
from helper import get_summ_prefix
from util import *

from helper import get_sum_data
from helper_run_bart import (write_pkl_to_disk, init_spacy, extract_tokens)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-model_family", default='bart')
    parser.add_argument("-data_name", default='xsum', help='name of dataset')
    parser.add_argument('-truncate_sent', default=20,
                        help='the max sent used for perturbation')
    parser.add_argument('-dir_save', default="/mnt/data0/jcxu/meta_data_ref",
                        help="The location to save output data. ")
    args = parser.parse_args()
    logger.info(args)
    if not os.path.exists(args.dir_save):
        os.makedirs(args.dir_save)

    # Run a PEGASUS/BART model to explain the local behavior
    # Sample one article from datasets
    dev_data = get_sum_data(args.data_name)
    train_data = get_sum_data(args.data_name, split='train')

    # init BART models
    if args.model_family == 'bart':
        from transformers import BartTokenizer
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    else:
        raise NotImplementedError
    sp_nlp = init_spacy()
    model_pkg = {'tok': tokenizer, 'spacy': sp_nlp}
    # {Do some perturbation to one example, run the model again, check if the token exist, write the result on the disk}
    feat_array = []
    model_prediction = []
    modified_text = []
    return_data = []

    for data_point in dev_data:
        document = data_point['document']
        summary = data_point['summary']
        uid = data_point['id']
        summary = summary.strip()
        document_sents = document.split('\n')[:args.truncate_sent]
        document = "\n".join(document_sents)
        input_doc = tokenizer(document, return_tensors='pt',
                              max_length=400, truncation=True, padding=True)

        # BUT we are going to use ref summary!!
        tokens, tags = extract_tokens(summary, nlp=model_pkg['spacy'])
        outputs = []
        start_matching_index = 0
        for (tok, tag) in zip(tokens, tags):
            # one step
            summary_prefix = get_summ_prefix(
                tgt_token=tok.strip(), raw_output_summary=summary, start_matching_index=start_matching_index)
            start_matching_index = len(summary_prefix) + len(tok.strip())
            if tok.strip() in string.punctuation:
                continue
            record = {}
            if not summary_prefix.endswith(" "):
                target_word_bpe = model_pkg['tok'].encode(" "+tok)[1]
                target_word_bpe_backup = model_pkg['tok'].encode(tok)[1]
            else:
                target_word_bpe_backup = model_pkg['tok'].encode(" "+tok)[1]
                target_word_bpe = model_pkg['tok'].encode(tok)[1]

            prefix_token_ids = tokenizer(summary_prefix, return_tensors='pt',)

            prefix_token_ids = prefix_token_ids['input_ids'][:,:-1]
            record['tgt_token_id'] = target_word_bpe
            record['tgt_token_id_backup'] = target_word_bpe_backup
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
                     'ref': summary,
                     'id': uid}}

        write_pkl_to_disk(args.dir_save, fname_prefix=uid, data_obj=final)

    logger.info('Done Collecting data ...')
