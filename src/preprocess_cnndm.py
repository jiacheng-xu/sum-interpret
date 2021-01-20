import multiprocessing
import pickle
from nltk.tokenize.treebank import TreebankWordDetokenizer
import spacy
nlp = spacy.load("en_core_web_sm")

detok = TreebankWordDetokenizer()


def process_input_doc(inp_doc, max_num_sent=15, max_token_num=50):
    doc = nlp(inp_doc)

    doc_list = []

    for sent in doc.sents:
        toks = [token.text for token in sent]
        toks = toks[:max_token_num]
        out = detok.detokenize(toks)
        doc_list.append(out)
        if len(doc_list) >= max_num_sent:
            break
    return doc_list


def run(idx, data_point, path, split_name):
    print(idx)
    document = data_point['article']
    ref_summary = data_point['highlights']
    out_doc = process_input_doc(document)
    xsum_style_doc = '\n'.join(out_doc)

    out_ref = process_input_doc(ref_summary)
    ref = ' '.join(out_ref)
    idname = f"{split_name}_{idx}"
    # with open(os.path.join(path, idname+'.pkl'), 'wb') as fd:
    #     pickle.dump({
    #         'id': idname,
    #         'document': xsum_style_doc,
    #         'summary': ref
    #     }, fd)
    return {
        'id': idname,
        'document': xsum_style_doc,
        'summary': ref
    }
from multiprocessing import Pool

if __name__ == "__main__":
    from datasets import load_dataset
    split_name = 'train'
    dataset = load_dataset('cnn_dailymail', '3.0.0', split=split_name)
    print(dataset)
    path = '/mnt/data0/jcxu/dataset_cnndm'
    import os
    result = []
    inps = []

    for idx, data_point in enumerate(dataset):
        inps.append([idx, data_point, path, split_name])

    with Pool(multiprocessing.cpu_count()) as p:
        result = p.starmap(run, inps)

    # for idx, data_point in enumerate(dataset):
    #     out = run(idx, data_point, path, split_name)

    with open(os.path.join(path, split_name+'.pkl'), 'wb') as fd:
        pickle.dump(result, fd)
