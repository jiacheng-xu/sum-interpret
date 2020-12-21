from collections import Counter
from datasets import load_dataset

dataset = load_dataset("xsum")
train_set = dataset['train']
# dev_set = dataset['validation']

print(train_set)

# Build a unigram and bigram dictionary
# map: unigram -> doc sent,  eg. "Obama" -> { "he is obama", "you are obama", ....} up to 100 sentences for each token

# map: unigram -> summary bigrams. eg. "of" -> "half_of": count=12, "some_of" count=15, .....
# map: unigram -> summary trigrams(future). eg. "about_half": "about_half_of" cnt=10, "about_half_employee" cnt=2

# The bigram and trigrams must show up more than K=5 times to be stored

MAX_SENT_FOR_TOKEN = 100
import re

WORD = re.compile(r'\w+')


def reg_tokenize(text):
    words = WORD.findall(text)
    return words


from typing import Dict


def trim_counter_in_dict(dict_cnts: dict, K=3):
    trimed = {}
    for k, v in dict_cnts.items():
        v = {x: count for x, count in v.items() if count >= K}
        if len(v) > 0:
            trimed[k] = v
    return trimed


def extract_ngram_bidirectional(text, n, map_dict, future=True):
    sum_unigram, sum_bigram = func_ngrams(text, n)
    for bigram_pair in sum_bigram:

        if future:
            key_word = bigram_pair[:-1]
            value =bigram_pair[-1]
        else:
            key_word = bigram_pair[1:]
            value = bigram_pair[0]
        key_word = "_".join(key_word)
        # concat_pair = "_".join(bigram_pair)
        if key_word in map_dict:
            cnter: Counter = map_dict[key_word]
            cnter.update([value])
            map_dict[key_word] = cnter
        else:
            map_dict[key_word] = Counter([value])
    return map_dict


def cache_feat_from_corpus(one_set, debug=False, min_cnt=3):
    cnt = 0
    map_tok_bigram_past = {}
    map_tok_sent_doc = {}
    map_tok_bigram_future = {}
    for example in one_set:
        cnt += 1
        document = example['document']
        summary = example['summary']

        # LM backward bigram ?_of_people
        map_tok_bigram_past = extract_ngram_bidirectional(summary, 2, map_tok_bigram_past, future=False)

        # sum_unigram, sum_bigram = func_ngrams(summary, n=2)
        # for bigram_pair in sum_bigram:
        #     key_word = bigram_pair[1:]
        #     key_word = "_".join(key_word)
        #     concat_pair = "_".join(bigram_pair)
        #     if key_word in map_tok_bigram_past:
        #         cnter: Counter = map_tok_bigram_past[key_word]
        #         cnter.update([concat_pair])
        #         map_tok_bigram_past[key_word] = cnter
        #     else:
        #         map_tok_bigram_past[key_word] = Counter([concat_pair])

        # LM trigrams  half_of_?, key = half_of, out = ?

        map_tok_bigram_future = extract_ngram_bidirectional(summary, 2, map_tok_bigram_future, future=True)
        # sum_unigram, sum_trigram = func_ngrams(summary, n=3)
        # for trigram_pair in sum_trigram:
        #     key_word = trigram_pair[:-1]
        #     key_word = "_".join(key_word)
        #     concat_pair = "_".join(trigram_pair)
        #     if key_word in map_tok_bigram_future:
        #         cnter: Counter = map_tok_bigram_future[key_word]
        #         cnter.update([concat_pair])
        #         map_tok_bigram_future[key_word] = cnter
        #     else:
        #         map_tok_bigram_future[key_word] = Counter([concat_pair])

        document_sents = document.split('\n')
        document_sents = document_sents[:5]  # let's just trim to the first few sentences.
        for doc_sent in document_sents:
            doc_unigram, doc_bigram = func_ngrams(doc_sent, n=2)
            for doc_tok in doc_unigram:
                if doc_tok in map_tok_sent_doc:
                    current_list_of_sents = map_tok_sent_doc[doc_tok]
                    if len(current_list_of_sents) > MAX_SENT_FOR_TOKEN:
                        continue
                    new_tok_sentences = map_tok_sent_doc[doc_tok] + [doc_sent]
                    map_tok_sent_doc[doc_tok] = new_tok_sentences
                else:
                    map_tok_sent_doc[doc_tok] = [doc_sent]

        if debug:
            if cnt > 10000:
                break
    # print(map_tok_bigram_past)
    map_tok_bigram_past = trim_counter_in_dict(map_tok_bigram_past, min_cnt)
    print(map_tok_bigram_past)
    map_tok_bigram_future = trim_counter_in_dict(map_tok_bigram_future, min_cnt)
    return map_tok_bigram_past, map_tok_bigram_future, map_tok_sent_doc


def func_ngrams(inp_str, n=2):
    inp_str = inp_str.lower()
    input = reg_tokenize(inp_str)
    # always return tokens
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return input, output


map_tok_bigram_past, map_tok_bigram_future, map_tok_sent_doc = cache_feat_from_corpus(train_set, debug=True)
