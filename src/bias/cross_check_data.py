import json
import os
import random

from newsplease import NewsPlease
file_cnn = '/home/jcxu/sum-interpret/data/CNN_CC_2016'
file_dm = '/home/jcxu/sum-interpret/data/DM_CC_2016'
file_dm='/home/jcxu/sum-interpret/data/CC-MAIN-2015-48-index'
file_dm='/home/jcxu/sum-interpret/data/delta_2014_2015'
file_dm='/home/jcxu/sum-interpret/data/rs.txt'
dir = '/home/jcxu/sum-interpret/data/cache_copies'
from multiprocessing import Process
import multiprocessing
import hashlib

def func(l):
    print(l)
    url = l
    url = url.split(" ")[0]

    hash_object = hashlib.sha1(url.encode()).hexdigest()
    # if os.path.isfile(os.path.join(dir, f"{hash_object}.json")):
    #     return
    try:
        article = NewsPlease.from_url(url)
        date = article.date_publish.year
        maintext = article.maintext
        if len(maintext) < 200:
            return
        output = {'url': url, 'text': maintext, }
        with open(os.path.join(dir, f"{date}_{hash_object}.json"),'w') as fd:
            json.dump(output, fd)
    except:
        return
"""
def func(l):
    l = eval(l)
    url = l['url']
    print(url)
    urlkey = l['urlkey']
    hash_object = hashlib.sha1(urlkey.encode()).hexdigest()
    # if os.path.isfile(os.path.join(dir, f"{hash_object}.json")):
    #     return
    if len(url) < 24:
        return

    try:
        article = NewsPlease.from_url(l['url'])
        date = article.date_publish.year
        maintext = article.maintext
        if len(maintext) < 200:
            return
        
        output = {'url': url, 'text': maintext, 'urlkey': urlkey}
        with open(os.path.join(dir, f"{date}_{hash_object}.json"),'w') as fd:
            json.dump(output, fd)
    except:
        return
"""
if __name__ == "__main__":
    with open(file_dm, 'r') as rfd:
        lines = rfd.read().splitlines()
    random.shuffle(lines)

    total = len(lines)
    count = 0
    pool = multiprocessing.Pool(processes=20)

    pool.map(func,lines)
    pool.close()
    pool.join()
    exit()
    

    for idx, l in enumerate(lines):
        l = eval(l)
        url = l['url']
        print(url)
        if len(url) < 24:
            continue
        # try:
            # year = int(url[19:23])
        # except:
            # continue

        # if year < 2007:
        #     continue
        # else:
        #     count += 1
        article = NewsPlease.from_url(l['url'])
        maintext = article.maintext
        if len(maintext) < 200:
            continue
        urlkey = l['urlkey']
        hash_object = hashlib.sha1(urlkey.encode()).hexdigest()
        output = {'url': url, 'text': maintext, 'urlkey': urlkey}
        with open(os.path.join(dir, f"{hash_object}.json"),'w') as fd:
            json.dump(output, fd)
        
    print(count)
    print(total)
