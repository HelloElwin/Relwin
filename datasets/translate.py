import numpy as np
import pickle
import os

prefix = './yelp/'
datatype = 'raw'

np.random.seed(19260817)

print('Loading (>﹏<)', end='\r')
with open(prefix + 'trn_' + datatype, 'rb') as fs:
    trn = pickle.load(fs).toarray()

with open(prefix + 'tst_' + datatype, 'rb') as fs:
    tst = pickle.load(fs)

with open(prefix + 'test_time', 'rb') as fs:
    tstTime = pickle.load(fs)

print('=== Loaded data (=ﾟωﾟ)ﾉ')

def translate_for_lgcn():
    with open('train.txt', 'w') as fs:
        for user, data in enumerate(trn):
            print('Processing %d/%d' % (user, len(trn)), end='\r')
            cand = np.argwhere(data != 0).flatten()
            raw = str(user)
            for item in cand:
                if data[item] == None: continue
                raw += ' ' + str(item)
            fs.write(raw + '\n')
    data = np.array(tst)
    user = np.argwhere(data != None).flatten()
    item = data[user]
    with open ('test.txt', 'w') as fs:
        for i in range(len(user)):
            fs.write(str(user[i]) + ' ' + str(item[i]) + '\n')
            print('Processing %d/%d' % (i, len(user)), end='\r')

def translate_for_rechorus():
    with open('train.csv', 'w') as fs:
        fs.write('user_id\titem_id\ttime\n')
        for user, data in enumerate(trn):
            print(f'Processing train set {user}/{len(trn)}', end='\r')
            cand = np.argwhere(data != 0).flatten()
            for item in cand:
                if data[item] == None: continue
                fs.write(str(user+1) + '\t' + str(item+1) + '\t' + str(data[item]) + '\n')
    with open('test.csv', 'w') as fs:
        fs.write('user_id\titem_id\ttime\tneg_items\n')
        users = np.argwhere(np.array(tst) != None).flatten()
        items = np.array(tst)[users]
        for user, item in zip(users, items):
            print(f'Processing test set {user}/{len(trn)}', end='\r')
            neg = np.argwhere(trn[user] == 0).flatten()
            np.random.shuffle(neg)
            while item in neg[:99]:
                np.random.shuffle(neg)
            neg = neg + 1
            fs.write(str(user+1) + '\t' + str(item+1) + '\t' + str(tstTime[user]) + '\t' + str(neg.tolist()[:99]) + '\n')

def translate_for_sasrec():
    with open('yelp.txt', 'w') as fs:
        for user, data in enumerate(trn):
            print(f'Processing train set {user}/{len(trn)}', end='\r')
            cand = np.argwhere(data != 0).flatten().tolist()
            cand = sorted(cand, key=lambda x: data[x])
            for item in cand:
                if data[item] == None: continue
                fs.write(str(user+1) + ' ' + str(item+1) + '\n')

if __name__ == '__main__':
    x = input('[0] LightGCN\n[1] ReChorus\n[2] SASRec\nChoice: ')
    if x == '0': translate_for_lgcn()
    if x == '1': translate_for_rechorus()
    if x == '2': translate_for_sasrec()
    print('=== Finished translating \(≧▽≦)/')
