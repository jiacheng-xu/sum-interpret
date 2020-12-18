
# localize the data and modify the data
from typing import List, Dict
from datasets import load_dataset, DatasetDict

import os
import logging
import sys
import json
import torch
import random
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def store_xsum_to_disk(path):
    dataset: DatasetDict = load_dataset(path='xsum', cache_dir=path)
    logging.info(dataset)
    logging.info(dataset.column_names)

    for split_name, split in dataset.items():
        output = []
        output_dict = {}
        for ex in split:
            sentences = ex['document'].split('\n')
            ex['sent_doc'] = sentences
            # output.append(ex)
            output_dict[ex['id']] = ex
        with open(os.path.join(path, f"{split_name}.json"), 'w') as wfd:

            # json.dump(output, wfd)
            json.dump(output_dict, wfd)
        logging.info(f"Done writing {split_name}")


def open_json_file(fname):
    with open(fname, 'r') as fd:

        data = json.load(fp=fd)
    # assert type(data) == Dict
    return data


def run_bert_for_retrieval(path, f, device='cuda:2', enc_doc=True):
    logging.info(f)
    fname = os.path.join(path, f)
    from transformers import BertTokenizer, BertModel
    import torch

    batch_size = 100
    max_seq_len = 100
    mname = 'bert-large-uncased'
    tokenizer = BertTokenizer.from_pretrained(mname)
    model = BertModel.from_pretrained(mname).to(device=device)
    original_data = open_json_file(fname)

    bert_to_run = []
    bert_to_run_id = []
    for k, d in original_data.items():
        doc, summary = d['document'], d['summary']
        if enc_doc:
            txt = doc
        else:
            txt = summary
        bert_to_run.append(txt)
        key = d['id']
        bert_to_run_id.append(key)

    if enc_doc:
        rep_name = 'rep_doc'
    else:
        rep_name = 'rep_sum'
    while bert_to_run:
        examples = bert_to_run[:batch_size]
        example_ids = bert_to_run_id[:batch_size]
        bert_to_run = bert_to_run[batch_size:]
        bert_to_run_id = bert_to_run_id[batch_size:]
        inputs = tokenizer(examples, truncation=True, max_length=100,
                           padding=True, return_tensors="pt").to(device)
        outputs = model(**inputs, return_dict=True)
        # hiddens = outputs['last_hidden_state'][:, 0, :]
        hiddens = outputs['pooler_output']
        np_hid = hiddens.cpu().detach().numpy()
        for idx, ex_id in enumerate(example_ids):
            hid_rep = np_hid[idx].tolist()
            original_data[ex_id][rep_name] = hid_rep

    with open(os.path.join(path, f"{rep_name}_{f}"), 'w') as fd:
        json.dump(
            original_data, fd
        )
    logging.info("Done")


def comp_distance(database, query):
    pass


def load_id_vec_pair(inp_dict, key_name):
    list_ids, list_vecs = [], []
    for k, v in inp_dict.items():
        list_ids.append(k)
        list_vecs.append(v[key_name])
    torch_mat = torch.as_tensor(list_vecs)
    return list_ids, torch_mat


def assign_closest_neighbor(path, train_fname, dev_fname, key_name='rep_doc', device='cuda:2'):
    with open(os.path.join(path, dev_fname), 'r') as fd:
        dev_data = json.load(fd)
    dev_ids, dev_mat = load_id_vec_pair(dev_data, key_name)
    logging.info("Done loading test")
    with open(os.path.join(path, train_fname), 'r') as tfd:
        train_data = json.load(tfd)
    train_ids, train_mat = load_id_vec_pair(train_data, key_name)

    logging.info("Done Loading...")

    with torch.cuda.device(device):
        # dev_mat: dev_samples, hdim
        # train_mat: train_sample, hdim
        sim = torch.matmul(dev_mat, train_mat.T)
        max_index = torch.argmax(sim, dim=-1)
        max_index_list = max_index.tolist()

    for idx, d_id in enumerate(dev_ids):
        dev_data[d_id]['ret_doc'] = train_data[train_ids[max_index_list[idx]]]['document']
        dev_data[d_id][key_name] = None
        if random.random() < 0.01:
            logging.info(
                f"Original DOC: {dev_data[d_id]['document']}\nMatch: {dev_data[d_id]['ret_doc']}\n\n\n")
    with open(os.path.join(path, 'ret'+dev_fname), 'w') as fd:
        json.dump(dev_data, fd)

    logging.info("DONE.")


if __name__ == "__main__":
    path = '/home/jcxu/data/xsum'
    # # cache xsum dataset to disk
    # store_xsum_to_disk(path)
    # run_bert_for_retrieval(path, 'test.json')
    # run_bert_for_retrieval(path, 'train.json',device='cuda:3')
    # run_bert_for_retrieval(path, 'validation.json')
    assign_closest_neighbor(
        path, train_fname='rep_doc_train.json', dev_fname='rep_doc_validation.json')
    assign_closest_neighbor(
        path, train_fname='rep_doc_train.json', dev_fname='rep_doc_test.json')
