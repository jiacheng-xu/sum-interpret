from helper import get_sum_data
from util import *
train_data = get_sum_data('xsum', split='train')
dataset = train_data

def read_ref_sum(dataset,q):
    output = []
    for ex in dataset:
        output.append(f"{ex['id']}\t{ex['summary']}")
        if q in ex['summary']:
            print(f"{ex['id']}\t{ex['summary']}")
    return "\n".join(output)

import string  

@dec_print_wrap
def retrieve_train(q):
    logger.info(f"Query: {q}")
    q = q.strip()
    if q in string.punctuation:
        return
    output = []
    max_cnt  = 20
    cnt=0
    for ex in dataset:
        output.append(f"{ex['id']}\t{ex['summary']}")
        if q in ex['summary']:
            logger.info(f"{ex['id']}\t{ex['summary']}")
            cnt += 1
            if cnt >= max_cnt:
                break
        

def write_disk(fname, data: str):
    with open(fname, 'w') as fd:
        fd.write(data)


if __name__ == "__main__":
    data_name = 'xsum'
    q = "A search is"
    print("--------DEV--------")
    dev_data = get_sum_data(data_name)
    dev_str = read_ref_sum(dev_data,q=q)
    print('='*40)
    # write_disk('dev_ref_sum.txt', dev_str)

    # test_data = get_sum_data(data_name, split='test')

    # test_str = read_ref_sum(test_data)
    # write_disk('test_ref_sum.txt', test_str)
    
    print('----TRAIN----')
    train_data = get_sum_data(data_name, split='train')
    train_str = read_ref_sum(train_data,q=q)
    # write_disk('train_ref_sum.txt', train_str)
