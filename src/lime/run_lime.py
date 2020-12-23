from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import sys
from src.util import *

from lib.lime_helper import train_dt
from lib.lm_feat_supp import map_tok_bigram_past, map_tok_bigram_future, map_tok_sent_doc, reg_tokenize

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('<br>%(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

import string

punct = string.punctuation
punct_list = [c for c in punct if c not in ['-']]


def get_xsum_data(split='validation'):
    # Load sentiment 140 dataset and return
    from datasets import load_dataset

    dataset = load_dataset('xsum', split=split)
    # # def renorm_label(example):
    # #     example['sentiment'] = 1 if example['sentiment'] >=3 else 0
    # #     return example
    # updated_dataset = dataset.map(renorm_label)
    logging.info(dataset.features)
    logging.info(dataset[0])
    logging.info("=" * 40)
    return dataset


def init_model(mname='sshleifer/distilbart-cnn-6-6', device='cuda:0'):
    from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
    model = BartForConditionalGeneration.from_pretrained(mname).to(device)
    tokenizer = BartTokenizer.from_pretrained(mname)
    return model, tokenizer


from typing import List

import nltk
from nltk.corpus import stopwords

stop_words = stopwords.words('english')


def check_if_a_bpe_is_a_token(bpe_tokenizer, bpe_id):
    tok = bpe_tokenizer.decode(bpe_id)
    if len(tok) == 0:
        return False
    if tok.startswith(" "):
        return True
    if tok[0] in punct_list:
        return True
    if bpe_id == bpe_tokenizer.bos_token_id or bpe_id == bpe_tokenizer.eos_token_id:
        return True
    if tok[0].isupper():
        return True
    return False


def check_if_bpe_is_a_word(bpe_tokenizer, bpe_id):
    tok = bpe_tokenizer.convert_ids_to_tokens(bpe_id)
    if tok.startswith("▁"):
        return True
    if tok[0] in punct_list:
        return True
    return False


def tokenize_text(tokenizer, raw_string, max_len=500):
    token_ids = tokenizer(raw_string, max_length=max_len, return_tensors='pt', truncation=True)
    token_ids_list = token_ids['input_ids'].tolist()[0]
    doc_str = "".join([tokenizer.decode(x) for x in token_ids_list])
    doc_str_lower = doc_str.lower()
    # reverse_eng_token_str = [tokenizer.decode(token) for token in token_ids_list]
    # lowercased_token_str = [x.lower() for x in reverse_eng_token_str]
    # lower_token_ids = [tokenizer.encode(x) for x in lowercased_token_str]
    # return token_ids, lower_token_ids, reverse_eng_token_str, lowercased_token_str
    return token_ids, doc_str, doc_str_lower


import torch


def run_model(model, tokenizer, input_text: List, device, sum_prefix=""):
    inputs = tokenizer(input_text, max_length=300, return_tensors='pt', truncation=True, padding=True)
    # Generate Summary
    # logging.info(torch.max(inputs['input_ids']))
    if sum_prefix:
        encoder_outputs = model.model.encoder(inputs['input_ids'].to(device), return_dict=True)
        if sum_prefix:
            sum_prefix = sum_prefix.strip()
            batch_size = len(input_text)
            decoder_input_ids = torch.LongTensor(tokenizer.encode(sum_prefix, return_tensors='pt')).to(device)
            decoder_input_ids = decoder_input_ids.expand((batch_size, decoder_input_ids.size()[-1]))
            decoder_input_ids = decoder_input_ids[:, :-1]
        else:
            decoder_input_ids = None

        model_inputs = {"input_ids": None,
                        "past_key_values": None,
                        "encoder_outputs": encoder_outputs,
                        "decoder_input_ids": decoder_input_ids,
                        }
        outputs = model(**model_inputs, use_cache=False, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        # next_token = next_token.unsqueeze(-1)
        next_token = next_token.tolist()

        output = [tokenizer.decode(tk) for tk in next_token]
        logging.info(f"Next token: {output}")
    else:
        summary_ids = model.generate(inputs['input_ids'].to(device), num_beams=1, max_length=30, early_stopping=True)
        output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                  summary_ids]
    return output


def feat_cnt(tgt, texts):
    concat = " ".join(texts)
    count = concat.count(tgt)
    bucket_len = len(bin(count)) - 2
    return bucket_len


def mock_lm_slot_filling(sentence, tgt_token):
    pass


def check_known_bigram(sent, tgt, future, n=2):
    tokens = reg_tokenize(sent)
    if tgt in tokens:
        index = tokens.index(tgt)
        try:
            if future:
                bigram = tokens[index - (n - 1):index]
                bigram = "_".join(bigram)
                if bigram in map_tok_bigram_future:
                    retrieve = map_tok_bigram_future[bigram]
                    if tgt in retrieve:
                        return True

            else:
                bigram = tokens[index + 1:index + n]
                bigram = "_".join(bigram)
                if bigram in map_tok_bigram_past:
                    retrieve = map_tok_bigram_past[bigram]
                    if tgt in retrieve:
                        return True
        except IndexError:
            return False
    else:
        return False
    return False


def feature_ext(original_text, tgt_token, summary_prefix="", max_sent=30):
    # remove <s> and </s> and clean
    lower_original_text = original_text.lower()
    lower_summary_prefix = summary_prefix.lower()
    sos_token, eos_token = tokenizer.bos_token, tokenizer.eos_token
    lower_original_text = lower_original_text.replace(sos_token, '')
    lower_original_text = lower_original_text.replace(eos_token, '')
    tgt_lower = tgt_token.lower()

    # split sentences
    sentences = lower_original_text.split('\n')[:max_sent]

    count = feat_cnt(tgt_token.lower(), sentences)  # feat 1: how often does it show in the context

    known_lm_f, known_lm_p = 0, 0
    sent_pos = [0 for _ in range(max_sent)]
    sent_pos_multiple = [0 for _ in range(max_sent)]
    for idx, sent in enumerate(sentences):
        if tgt_lower in sent:
            sent_pos[idx] = 1  # feat: does it show in sent x
            if sent.count(tgt_lower) > 1:
                sent_pos_multiple[idx] = 1
            # feat 4 is it a known bigram?
            flag_future = check_known_bigram(sent, tgt_lower, True)
            if flag_future:
                known_lm_f += 1
            flag_past = check_known_bigram(sent, tgt_lower, False)
            if flag_past:
                known_lm_p += 1

    # sent_pos                  feat 3              : does it show multiple times in sent x

    # decoder feature
    sum_prefix_tokens = reg_tokenize(lower_summary_prefix)
    last_token = sum_prefix_tokens[-1]
    dec_bigram_f, dec_bigram_b = 0, 0
    if last_token in map_tok_bigram_future:
        retrieve = map_tok_bigram_future[last_token]
        if tgt_lower in retrieve:
            dec_bigram_f = 1
    if tgt_lower in map_tok_bigram_past:
        retrieve = map_tok_bigram_past[tgt_lower]
        if last_token in retrieve:
            dec_bigram_b = 1
    feat_names = ['cnt', 'bigram_next', 'bigram_past', 'dec_bigram_next', 'dec_bigram_past'] + [f"pos_{x}" for x in
                                                                                                range(max_sent)] + [
                     f"pos_mul_{x}" for x
                     in
                     range(max_sent)]

    return [count, known_lm_f, known_lm_p, dec_bigram_f,
            dec_bigram_b] + sent_pos + sent_pos_multiple, feat_names  # 5 + 30 + 30


def index_of_sent_with_tgt(texts, tgt_token):
    out = []
    for idx, sent in enumerate(texts):
        if tgt_token in sent or tgt_token.lower() in sent:
            out.append(idx)
    return out


def index_of_sent_with_tgt_bin(texts, tgt_token) -> List[int]:
    out = []
    for idx, sent in enumerate(texts):
        if tgt_token in sent or tgt_token.lower() in sent:
            out.append(1)
        else:
            out.append(0)
    return out


def pertub_add_mask(texts, tgt_token):
    # replace the occurence of tgt with MASK
    mask_token = tokenizer.unk_token
    sent_idx = index_of_sent_with_tgt(texts, tgt_token)
    if not sent_idx:
        return texts
    sel_sent_idx = random.choice(sent_idx)
    if tgt_token.lower() in texts[sel_sent_idx]:
        texts[sel_sent_idx] = texts[sel_sent_idx].replace(tgt_token.lower(), mask_token)
    elif tgt_token in texts[sel_sent_idx]:
        texts[sel_sent_idx] = texts[sel_sent_idx].replace(tgt_token, mask_token)
    return texts


def pertub_LM_swap(texts, tgt_token):
    # replace the occurence of tgt with a language model bigram.
    # eg. {UK} president => {US} president
    sent_idx = index_of_sent_with_tgt(texts, tgt_token)
    if not sent_idx:
        return texts
    sel_sent_idx = random.choice(sent_idx)
    original_sent_lower = texts[sel_sent_idx].lower()
    tokenized_original_sent_words = reg_tokenize(original_sent_lower)
    tgt_token_lower = tgt_token.lower()
    idx_of_tgt = tokenized_original_sent_words.index(tgt_token_lower)
    if idx_of_tgt + 1 >= len(tokenized_original_sent_words):
        return texts  # if it's already the last word, forget about this case
    next_word = tokenized_original_sent_words[idx_of_tgt + 1]

    if next_word in map_tok_bigram_past:
        bigram_dict = map_tok_bigram_past[next_word]
    else:
        return texts

    rand_word = random.choices(list(bigram_dict.keys()), weights=bigram_dict.values(), k=1)[0]
    logging.info(f"LM swap Replace: {tgt_token} {next_word}  => {rand_word} {next_word}")
    # logging.info(f"--------Replace before: {texts[sel_sent_idx]}")
    if tgt_token.lower() in texts[sel_sent_idx]:
        texts[sel_sent_idx] = texts[sel_sent_idx].replace(tgt_token.lower(), rand_word)
    elif tgt_token in texts[sel_sent_idx]:
        texts[sel_sent_idx] = texts[sel_sent_idx].replace(tgt_token, rand_word)
    logging.info(f"--------Replace after: {texts[sel_sent_idx]}")
    return texts


def pertub_sent_reorder(texts, tgt_token):
    sent_idx = index_of_sent_with_tgt(texts, tgt_token)
    if not sent_idx:
        return texts
    sent = texts.pop(random.choice(sent_idx))
    l_of_doc = len(texts)
    texts.insert(random.choice(range(l_of_doc)), sent)
    return texts


def pertub_insert(texts, tgt_token):
    l_of_doc = len(texts)
    sel_sent_idx = random.choice(range(l_of_doc))
    sent = texts[sel_sent_idx]
    tokens = sent.split(" ")
    tokens.insert(random.choice(range(len(tokens))), tgt_token)
    new_text = " ".join(tokens)
    texts[sel_sent_idx] = new_text
    return texts


import random


def perturb_single_text(original_text, tgt_token, tokenizer):
    sos_token, eos_token = tokenizer.bos_token, tokenizer.eos_token
    text = original_text.replace(sos_token, '')
    text = text.replace(eos_token, '')
    # split sentences
    sentences = text.split('\n')
    perturb_functions = [pertub_sent_reorder, pertub_insert, pertub_add_mask, pertub_LM_swap]
    perturb_weights = [0.7, 1.2, 0.4, 0.6]
    sample_funcs = random.choices(perturb_functions, weights=perturb_weights, k=3)
    for samp_f in sample_funcs:
        sentences = samp_f(sentences, tgt_token)
    concat_sent = "\n".join(sentences)
    return concat_sent


def pertub_text(original_text, tgt_token, raw_output_summary, tokenizer, nsample=300):
    # return pertubed text and feature vector
    # return original text with feature vector
    summary_prefix = raw_output_summary[:raw_output_summary.find(tgt_token)]
    logging.info(f"Prefix: {summary_prefix}")
    # map_tok_bigram_ref, map_tok_trigram_ref, map_tok_sent_doc

    pert_examples = []
    feat_raw, feat_names = feature_ext(original_text, tgt_token, summary_prefix)
    for n in range(nsample):
        flag = True
        sample_sent = perturb_single_text(original_text, tgt_token, tokenizer)
        for pert_ex in pert_examples:
            if sample_sent == pert_ex[0]:
                flag = False
                break
        if flag == False:
            continue
        sample_feat, _ = feature_ext(sample_sent, tgt_token, summary_prefix)
        pert_examples.append((sample_sent, sample_feat))
    return feat_raw, feat_names, pert_examples, summary_prefix


def init_bart_lm():
    from transformers import BartTokenizer,BartForConditionalGeneration
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    TXT = "My friends are <mask> but they eat too many carbs."


def multiple_lm_distillation():
    pass

if __name__ == '__main__':
    # Run a PEGASUS/BART model to explain the local behavior
    # Sample one article from datasets
    data = get_xsum_data()
    device = 'cuda:0'
    auto_data_collect = False
    # Run the model and input a token of interest (Obama)
    model, tokenizer = init_model(device=device)
    batch_size = 40
    max_sample = 400
    # {Do some perturbation to one example, run the model again, check if the token exist, write the result on the disk}
    feat_array = []
    model_prediction = []
    modified_text = []
    try:
        for data_point in data:
            logging.info(f"Example: {data_point['document'][:6000]}")
            new_example = True
            raw_document = data_point['document']
            token_ids, doc_str, doc_str_lower = tokenize_text(tokenizer, raw_document)

            raw_output = run_model(model, tokenizer, [raw_document], device=device)
            raw_output_summary = raw_output[0].strip()
            logging.info(f"Model Output: {raw_output_summary}")

            while True:
                if auto_data_collect:
                    if new_example:
                        raw_out_proc = raw_output[0].strip()
                        tokens = raw_out_proc.split(' ')
                        interest = tokens[0]
                        logging.info(f"Interest: {interest}")
                        new_example = False
                    else:
                        break
                else:
                    interest = input("Please enter a word of interest:\n")
                    # interest = 'Australian'
                    if 'q' in interest:
                        break

                lower_interest = interest.lower().strip()
                if lower_interest not in doc_str_lower:
                    logging.warning(f"Input token {interest} not found in the input document.")
                    continue

                feat_original, feat_names, pert_examples, summary_prefix = pertub_text(doc_str, interest,
                                                                                       raw_output_summary, tokenizer,
                                                                                       nsample=max_sample)

                logging.info("Run Summ model...")

                # feat_array = []
                # model_prediction = []
                # modified_text = []
                batch_for_model = []
                for idx, pert_ex in enumerate(pert_examples):
                    pert_text, pert_feat = pert_ex
                    feat_array.append(pert_feat)
                    modified_text.append(pert_text)
                    batch_for_model.append(pert_text)
                    if len(batch_for_model) == batch_size:
                        model_output = run_model(model, tokenizer, batch_for_model, device=device,
                                                 sum_prefix=summary_prefix)
                        labels = index_of_sent_with_tgt_bin(model_output, interest)
                        model_prediction += labels
                        batch_for_model = []
                if len(batch_for_model) != 0:
                    model_output = run_model(model, tokenizer, batch_for_model, device=device,
                                             sum_prefix=summary_prefix)
                    labels = index_of_sent_with_tgt_bin(model_output, interest)
                    model_prediction += labels
                    batch_for_model = []
                logging.info(f"feat: {len(feat_array)} model pred: {len(model_prediction)}")
                assert len(feat_array) == len(model_prediction)
    except KeyboardInterrupt:
        logging.info('Done Collecting data ...')
    if len(feat_array) != len(model_prediction):
        minlen = min(len(feat_array), len(model_prediction))
        feat_array = feat_array[:minlen]
        model_prediction = model_prediction[:minlen]
    assert len(feat_array) == len(model_prediction)
    train_dt(feat_names, feat_array, model_prediction,
             modified_text)  # train a decision tree model for visualization
