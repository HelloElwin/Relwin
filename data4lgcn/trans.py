import numpy as np
import pickle
import os

os.system('rm train.txt test.txt')
os.system('touch train.txt test.txt')

with open('trn_ui', 'rb') as fs:
    A = pickle.load(fs).toarray()
print('=== Loaded train data')

with open('train.txt', 'w') as fs:
    for user, data in enumerate(A):
        print('Processing %d/%d' % (user, len(A)), end='\r')
        cand = np.argwhere(data != 0).flatten()
        raw = str(user)
        for item in cand:
            if data[item] == None: continue
            raw += ' ' + str(item)
        fs.write(raw + '\n')

print('=== Finished translating train data')


with open('tst_int', 'rb') as fs:
    data = pickle.load(fs)
print('=== Loaded test data')

data = np.array(data)
user = np.argwhere(data != None).flatten()
item = data[user]

with open ('test.txt', 'w') as fs:
    for i in range(len(user)):
        fs.write(str(user[i]) + ' ' + str(item[i]) + '\n')
        print('Processing %d/%d' % (i, len(user)), end='\r')

print('=== Finished translating test data')
