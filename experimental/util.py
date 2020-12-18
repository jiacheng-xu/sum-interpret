import copy
from typing import Any, List, Optional
import torch
import os
import json
from typing import List, Dict

from transformers import BatchEncoding, PreTrainedTokenizer


def read_json_data(fdir, fname):
    """Read json file from fdir/fname. Assume it contains key 'data'."""
    fp = open(os.path.join(fdir, fname), 'r')
    data_dict = json.load(fp)['data']
    return data_dict


def get_input_docs_from_json(data_dict: List[Dict], use_add_sent=True):
    outputs = []
    outputs_qa_pairs = []
    for data in data_dict:
        inp_doc, add_sent = data['input_doc'], data['added_sent']
        input_str = inp_doc if not use_add_sent else "{} {}".format(
            inp_doc, add_sent)
        outputs.append(input_str)

        mask_pairs = data['mask_pairs']
        for mask_pair in mask_pairs:
            q, a, wa = mask_pair['q'], mask_pair['a'], mask_pair['wa']
            outputs_qa_pairs.append(
                {
                    'context': input_str,
                    'q': q,
                    'a': a,
                    'wa': wa
                }
            )
    return outputs, outputs_qa_pairs


def attr_visualization():
    pass
