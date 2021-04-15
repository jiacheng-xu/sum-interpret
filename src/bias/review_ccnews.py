from warcio.archiveiterator import ArchiveIterator
import requests

def print_records(url):
    resp = requests.get(url, stream=True)

    for record in ArchiveIterator(resp.raw, arc2warc=True):
        if record.rec_type == 'warcinfo':
            print(record.raw_stream.read())

        elif record.rec_type == 'response':
            # if record.http_headers.get_header('Content-Type') == 'text/html':
            print(record.rec_headers.get_header('WARC-Target-URI'))
            print(record.content_stream().read())
            print('')

# print_records('https://archive.org/download/ExampleArcAndWarcFiles/IAH-20080430204825-00000-blackbook.warc.gz')

"""
path_to_file = '/mnt/data0/jcxu/CC-NEWS-20160926211809-00000.warc'
with open(path_to_file, 'rb') as stream:
    for record in ArchiveIterator(stream):
        if record.rec_type == 'response':
            print(record.rec_headers.get_header('WARC-Target-URI'))
            print(record.content_stream().read())
"""
import re


def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

def simplify_function(highlights):
    highlights_clean = re.sub(r'[^\w\s]', '', highlights)
    tokens = highlights_clean.split(' ')
    tokens = [ t for t in tokens if len(t) > 1]
    grams = find_ngrams(tokens,4)
    candidate_set = set(grams)
    return candidate_set



if __name__ == "__main__":
    # execute only if run as a script
    from datasets import load_dataset
    path_cnn = '/mnt/data0/jcxu/news-please/cc_download_articles/www.cnn.com'
    dataset = load_dataset("cnn_dailymail",'3.0.0',split='test')
    print()
    all_set = []
    combined_set = set()
    for data in dataset:
        article = data['article']
        highlights = data['highlights']
        # CNN or DM?
        if 'CNN' in article:
            is_cnn = True
        else:
            is_cnn = False
        if not is_cnn:
            continue
        extract_set = simplify_function(highlights)
        all_set.append(extract_set)
        combined_set.update(extract_set)
    
    print("Done preparing data")
    import os
    import json
    files =  os.listdir(path_cnn)
    for f in files:
        with open(os.path.join(path_cnn, f), 'r') as fd:
            exam = json.load(fd)
        description = exam['description']
        maintext = exam['maintext']
        if not description:
            description = ""
        if not maintext:
            maintext = ""
        concat = " ".join([description, maintext])
        cand = simplify_function(concat)
        if len(cand.intersection(combined_set)) >4:
            intersec = cand.intersection(combined_set)
            print(intersec)
            print('='*20)
            for x in all_set:
                if len(cand.intersection(x)) > 2:
                    print(x)
            print('-'*20)
        else:
            print("pass")


        
        
