# encoding:utf-8
import scipy.sparse as sp
import numpy as np
import pickle
import os

dataset = input("Choose a dataset: ")
prefix = './' + dataset + '/'
tarfix = './variants/'
datatype = 'raw'

np.random.seed(19260817)

print('Loading (>﹏<)', end='\r')
with open(prefix + 'trn_' + datatype, 'rb') as fs:
    trn = pickle.load(fs)

with open(prefix + 'tst_' + datatype, 'rb') as fs:
    tst = pickle.load(fs)

with open(prefix + 'test_time', 'rb') as fs:
    tstTime = pickle.load(fs)

print('=== Loaded data (=ﾟωﾟ)ﾉ')

def translate_for_lgcn():
    print('Loading (>﹏<)', end='\r')
    global trn
    trn = trn.toarray()
    with open(tarfix + 'train.txt', 'w') as fs:
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
    with open (tarfix + 'test.txt', 'w') as fs:
        for i in range(len(user)):
            fs.write(str(user[i]) + ' ' + str(item[i]) + '\n')
            print('Processing %d/%d' % (i, len(user)), end='\r')

def translate_for_rechorus():
    print('Loading (>﹏<)', end='\r')
    global trn
    trn = trn.toarray()
    with open(tarfix + 'train.csv', 'w') as fs:
        fs.write('user_id\titem_id\ttime\n')
        for user, data in enumerate(trn):
            print(f'Processing train set {user}/{len(trn)}', end='\r')
            cand = np.argwhere(data != 0).flatten()
            for item in cand:
                if data[item] == None: continue
                fs.write(str(user+1) + '\t' + str(item+1) + '\t' + str(data[item]) + '\n')
    with open(tarfix + 'test.csv', 'w') as fs:
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
    print('Loading (>﹏<)', end='\r')
    global trn
    trn = trn.toarray()
    with open(tarfix + 'yelp.txt', 'w') as fs:
        for user, data in enumerate(trn):
            print(f'Processing train set {user}/{len(trn)}', end='\r')
            cand = np.argwhere(data != 0).flatten().tolist()
            cand = sorted(cand, key=lambda x: data[x])
            for item in cand:
                if data[item] == None: continue
                fs.write(str(user+1) + ' ' + str(item+1) + '\n')
                
def translate_for_elwin_sasrec():
    global trn
    global tst
    with open(tarfix + 'trn', 'wb') as fs:
        seq = []
        for user in range(trn.shape[0]):
            data = trn[user].toarray().flatten()
            print(f'Processing train set {user}/{trn.shape[0]}', end='\r')
            cand = np.argwhere(data != 0).flatten().tolist()
            cand = sorted(cand, key=lambda x: data[x])
            cand = (np.array(cand) + 1).tolist()
            seq.append(cand)
        pickle.dump(seq, fs)
    with open(tarfix + 'tst', 'wb') as fs:
        tst = np.array(tst)
        tst[np.argwhere(tst!=None)] += 1
        pickle.dump(tst.tolist(), fs)
    with open(tarfix + 'trn_raw', 'wb') as fs:
        trn = sp.coo_matrix(trn)
        row = trn.row
        col = trn.col
        dat = trn.data
        col = col + 1
        trn = sp.csr_matrix((dat, (row, col)))
        pickle.dump(trn, fs)
    print(f'=== shape of translated trn_raw: {trn.shape}')

def translate_for_bert4rec():
    global trn
    global tst
    with open(tarfix + 'trn', 'wb') as fs:
        seq = []
        for user in range(trn.shape[0]):
            data = trn[user].toarray().flatten()
            print(f'Processing train set {user}/{trn.shape[0]}', end='\r')
            cand = np.argwhere(data != 0).flatten().tolist()
            cand = sorted(cand, key=lambda x: data[x])
            cand = (np.array(cand) + 1).tolist()
            seq.append(cand)
        pickle.dump(seq, fs, protocol=2)
    with open(tarfix + 'tst', 'wb') as fs:
        tst = np.array(tst)
        tst[np.argwhere(tst!=None)] += 1
        pickle.dump(tst.tolist(), fs, protocol=2)
    max_item_id = 0
    for data in seq:
        max_item_id = max(max_item_id, np.max(data))
    print(f'=== Max user id: {len(seq)}, Max item id: {max_item_id}')

if __name__ == '__main__':
    print(f'=== shape of trn_raw: {trn.shape}')
    x = input('[0] LightGCN\n[1] ReChorus\n[2] SASRec\n[3] Elwin\'s SASRec\n[4] Bert4Rec\nChoice: ')
    if x == '0': translate_for_lgcn()
    if x == '1': translate_for_rechorus()
    if x == '2': translate_for_sasrec()
    if x == '3': translate_for_elwin_sasrec()
    if x == '4': translate_for_bert4rec()
    print('=== Finished translating \(≧▽≦)/')
