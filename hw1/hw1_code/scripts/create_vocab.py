import glob
import numpy as np

paths = '../asrs/*.txt'
words_list = []

for path in glob.glob(paths):
    f = open(path, 'r')
    for line in f.readlines():
        words = line.replace('\n','').replace('.', '').replace(',', '').split(' ')
        words_list.extend(words)

words_arr = np.array(words_list)
vocab = np.unique(words_arr)

with open("../vocab", 'w') as f:
    for word in vocab:
        f.write('%s\n' % word)
