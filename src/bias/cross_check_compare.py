import json
import os
from pathlib import WindowsPath
from datasets import load_dataset
dataset = load_dataset('cnn_dailymail', '3.0.0', split='validation')
all_ret = []
docs = []
import random
print(len(dataset))
# exit()
for article in dataset:
    retrieve = []
    highlists = article['highlights']
    docs.append(highlists)
    tokens = highlists.split(" ")[:50]
    for idx in range(len(tokens) - 8):
        tks = tokens[idx:idx+7]
        k = "_".join(tks)
        retrieve.append(k)
    retrieve = set(retrieve)
    all_ret.append(retrieve)
print(len(all_ret))
# retrieve = set(retrieve)

dir = '/home/jcxu/sum-interpret/data/cache_copies'

files = os.listdir(dir)
files = [f for f in files if f.startswith('2015')]
random.shuffle(files)
cnt = 0
for jdx,f in enumerate(files):
    if jdx % 100 ==0:
        print(jdx)
    with open(os.path.join(dir, f), 'r') as fd:
        ob = json.load(fd)
    text = ob['text'][:1000]
    tt = text.split(" ")
    candidate = []
    for idx in range(len(tt) - 8):
        tks = tt[idx:idx+7]
        k = "_".join(tks)
        candidate.append(k)
    candidate = set(candidate)
    comp = [idx for idx, ret in enumerate(
        all_ret) if len(candidate.intersection(ret)) > 3]
    if len(comp) > 0:
        cnt += 1
    else:
        continue
    print('-'*20)
    print(text[:500])
    for c in comp:
        print('='*20)
        print(docs[c])
    print('\n')
print(cnt)
print(len(dataset))