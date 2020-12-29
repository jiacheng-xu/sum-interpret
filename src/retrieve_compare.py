

from datasets import load_dataset
from collections import Counter
from typing import Dict
MAX_SENT_FOR_TOKEN = 15

from helper import *
def func_ngrams(inp_str, n=2):
    inp_str = inp_str.lower()
    input = reg_tokenize(inp_str)
    # always return tokens
    output = []
    for i in range(len(input)-n+1):
        output.append(input[i:i+n])
    return input, output


def trim_counter_in_dict(dict_cnts: dict, K=3):
    trimed = {}
    for k, v in dict_cnts.items():
        v = {x: count for x, count in v.items() if count >= K}
        total_cnts = sum([count for x, count in v.items()])
        v = {x: count/total_cnts for x, count in v.items() if count >= K}
        if len(v) > 0:
            trimed[k] = v
    return trimed


def cache_feat_from_corpus(data, part, n=2, debug=True, min_cnt=3):
    cnt = 0
    result_map = {}
    for example in data:
        cnt += 1
        if debug and cnt > 5000:
            break
        content = example[part]
        sentences = content.split("\n")
        for sent in sentences:
            tokens, match_ngrams = func_ngrams(sent, n=n)
            for ngram in match_ngrams:
                key_word = "_".join(ngram[:-1])
                stuff = ngram[-1]
                if key_word in result_map:
                    cnter = result_map[key_word]
                    cnter.update([stuff])
                    result_map[key_word] = cnter
                else:
                    result_map[key_word] = Counter([stuff])
    trim_result_map = trim_counter_in_dict(result_map)
    return trim_result_map


map_summary_xsum = cache_feat_from_corpus(
    load_dataset('xsum')['train'], 'summary')

map_summary_cnndm = cache_feat_from_corpus(
    load_dataset('cnn_dailymail', '3.0.0')['train'], 'highlights')

if __name__ == "__main__":
    # cache dataset-split-ref/doc
    parser = argparse.ArgumentParser()

    parser.add_argument("-data_name", default='xsum', help='name of dataset')
    parser.add_argument("-split", default='validation')
    parser.add_argument("-doc", action='store_true')
    parser.add_argument("-ref", action='store_true')

    parser.add_argument('-dir_save', default="/mnt/data0/jcxu/",
                        help="The location to save output data. ")
    args = parser.parse_args()

    dataset = load_dataset(args.data_name)
    split = dataset[args.split]
    print(split)
    map_summary = cache_feat_from_corpus(split, 'summary')
    # map_summary = cache_feat_from_corpus(split, 'document')
    """
    for k,v in map_summary.items():
        # print(f"KEY: {k}")
        # print(v)
    """
