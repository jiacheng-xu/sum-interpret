import re


# import pickle
# with open('/mnt/data0/jcxu/dataset_cnndm/validation.pkl', 'rb') as fd:
#     data = pickle.load(fd)
# ids = [ d['id'] for d in data]
# summs = [ d['summary'] for d in data]
# out = [ f"{x}\t{y}" for x,y in zip(ids, summs)]
# out = "\n".join(out)
# with open('/mnt/data0/jcxu/dev_cnndm_sum.log', 'w') as fd:
#     fd.write(out)
# exit()

import nltk
from nltk.corpus import stopwords
stp = stopwords.words('english')
import statistics
p = "/mnt/data0/jcxu/dev_cnndm_sum.log"
q = '/home/jcxu/sum-interpret/train_ref_sum.log'
print("cnndm")
with open(p, 'r') as fd:
    lines = fd.read().splitlines()
    lines = [l for l in lines if l.startswith('vali')]

lines = [l.split('\t')[1] for l in lines]
print(lines[0])
cnndm_string = "\n".join(lines)
print(len(cnndm_string.split(" ")))
# print(f"Num char: {len(s)}")
# cnts = [ s.count(f"{word}") for word in stp]

# print(statistics.mean(cnts))

with open(q, 'r') as fd:
    lines = fd.read().splitlines()
    lines = [l for l in lines]
lines = [l.split('\t')[1] for l in lines]
print(lines[0])
xsum_string = "\n".join(lines)
print(len(xsum_string.split(" ")))
# print(f"Num char: {len(s)}")
# cnts = [ s.count(f"{word}") for word in stp]
# import statistics
# print(statistics.mean(cnts))

# exit()
while True:
    k = input("Input query!")
    cnndm_cnt = cnndm_string.count(k)
    xsum_cnt = xsum_string.count(k)

    toks = k.split(" ")
    toks = toks[:-1]
    newtok = " ".join(toks)
    print(newtok)

    cnndm_prefix_cnt = cnndm_string.count(newtok)
    xsum_prefix_cnt = xsum_string.count(newtok)
    print(f"{xsum_cnt},{xsum_prefix_cnt},{cnndm_cnt},{cnndm_prefix_cnt}")

