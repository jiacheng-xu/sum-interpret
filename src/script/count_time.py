import os
import pickle


def load_pickle(dir, fname) :
    with open(os.path.join(dir, fname), 'rb') as rfd:
        data = pickle.load(rfd)
    return data



dir = '/mnt/data0/jcxu'
path = 'task_occ_xsum_0.5'
# path = 'task_occ_sent_sel_xsum_0.5'

new_dir = os.path.join(dir,path)
files = os.listdir(new_dir)
times = []
import random

for f in files:
    data = load_pickle(new_dir, f)
    time = data['time']
    print(time)
    times.append(time)
import statistics
print(statistics.mean(times))