# encoding=utf-8
########################
# adapted from: akaxlh #
########################
import numpy as np
import json
import pickle
from TimeLogger import log
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import time
import copy

np.random.seed(19260817) # (=ﾟωﾟ)ﾉ

time_len = []
sequ_len = []

def transTime(date):
    timeArr = time.strptime(date, '%Y-%m-%d %H:%M:%S')
    return time.mktime(timeArr) # 用秒数表示的时间, 零点时间: 1970-01-01 8:00:00

def mapping(infile):
    uName2id = dict()
    iName2id = dict()
    uId, iId = (0, 0)
    interaction = list()
    with open(infile, 'r', encoding='utf-8') as fs:
        for line in fs:
            data  = json.loads(line.strip())
            uName = data['user_id'] # 一个用户的加密字符串
            iName = data['business_id'] # 一个商品的加密字符串
            time_stamp = transTime(data['date']) # 一个浮点数
            if uName not in uName2id:
                uName2id[uName] = uId
                interaction.append(dict())
                uId += 1
            if iName not in iName2id:
                iName2id[iName] = iId
                iId += 1
            usr = uName2id[uName]
            itm = iName2id[iName]
            interaction[usr][itm] = time_stamp # 不考虑一个商品多次交互？
    return interaction, uId, iId

# sparse for ours: 15, 10, 5
# original dense: 30, 20, 15

def checkFunc1(cnt):
    return cnt >= 15
def checkFunc2(cnt):
    return cnt >= 10 
def checkFunc3(cnt):
    return cnt >= 5

def filter(interaction, usrnum, itmnum, ucheckFunc, icheckFunc, filterItem=True):
    # get keep set
    usrKeep = set()
    itmKeep = set()
    itmCnt = [0] * itmnum # 商品被交互次数
    for usr in range(usrnum):
        data = interaction[usr]
        usrCnt = 0 # 当前用户交互量
        for col in data:
            itmCnt[col] += 1
            usrCnt += 1
        if ucheckFunc(usrCnt):
            usrKeep.add(usr)
    for itm in range(itmnum):
        if not filterItem or icheckFunc(itmCnt[itm]):
            itmKeep.add(itm)

    # filter data
    retint = list()
    usrid = 0
    itmid = 0
    itmId = dict()
    for row in range(usrnum):
        if row not in usrKeep: continue
        usr = usrid
        usrid += 1
        retint.append(dict())
        data = interaction[row]
        for col in data:
            if col not in itmKeep: continue
            if col not in itmId:
                itmId[col] = itmid
                itmid += 1
            itm = itmId[col]
            retint[usr][itm] = data[col]
    return retint, usrid, itmid

def division_by_interaction_without_test(trn, usrnum, itmnum, l1, l2):
    ret_trn = [[], [], []]
    ret_num = [0, 0, 0]
    for idx, data in enumerate(trn):
        item_cnt = 0
        l, r = 1e31, 0
        for itm in data:
            if data[itm] != None:
                l = min(l, data[itm])
                r = max(r, data[itm])
            item_cnt += 1
        time_len.append(r - l)
        sequ_len.append(item_cnt)
        if item_cnt < l1:
            ret_trn[0].append(copy.deepcopy(data))
            ret_num[0] += 1
        elif item_cnt > l2:
            ret_trn[2].append(copy.deepcopy(data))
            ret_num[2] += 1
        else:
            ret_trn[1].append(copy.deepcopy(data))
            ret_num[1] += 1
    return ret_trn, ret_num

def division_by_interaction(trn, tst, usrnum, itmnum, l1, l2):
    ret_trn = [[], [], []]
    ret_tst = [[], [], []]
    ret_num = [0, 0, 0]
    for idx, data in enumerate(trn):
        item_cnt = 0
        l, r = 1e31, 0
        for itm in data:
            if data[itm] != None:
                l = min(l, data[itm])
                r = max(r, data[itm])
            item_cnt += 1
        time_len.append(r - l)
        sequ_len.append(item_cnt)
        if item_cnt < l1:
            ret_trn[0].append(copy.deepcopy(data))
            ret_tst[0].append(tst[idx])
            ret_num[0] += 1
        elif item_cnt > l2:
            ret_trn[2].append(copy.deepcopy(data))
            ret_tst[2].append(tst[idx])
            ret_num[2] += 1
        else:
            ret_trn[1].append(copy.deepcopy(data))
            ret_tst[1].append(tst[idx])
            ret_num[1] += 1
    return ret_trn, ret_tst, ret_num

def augmentation_from_raw(trn, tst, usrnum, l1, l2):
    rettrn = copy.deepcopy(trn)
    rettst = copy.deepcopy(tst)
    retusr = copy.deepcopy(usrnum)
    for seq in range(1, 2):
        for idx, data in enumerate(trn[seq]):
            itmcnt = 0
            if seq == 1: 
                items = []
                for itm in data:
                    if data[itm] == None: continue
                    items.append((itm, data[itm]))
                items = sorted(items, key=lambda x: x[1])[:l1 - np.random.randint(5) - 1] # 加random
                data = {}
                for item, time in items:
                    data[item] = time
                rettrn[seq].append(copy.deepcopy(data))
                rettst[seq].append(None)
                retusr[seq] += 1
            else:
                if np.random.randn() > 0.2:
                    items = []
                    for itm in data:
                        if data[itm] == None: continue
                        items.append((itm, data[itm]))
                    items = sorted(items, key=lambda x: x[1])[:l2 - np.random.randint(10)]
                    data = {}
                    for item, time in items:
                        data[item] = time
                    rettrn[seq].append(copy.deepcopy(data))
                    rettst[seq].append(None)
                    retusr[seq] += 1
                elif np.random.randn() > -0.2:
                    items = []
                    for itm in data:
                        if data[itm] == None: continue
                        items.append((itm, data[itm]))
                    items = sorted(items, key=lambda x: x[1])[:l1 - np.random.randint(5) - 1]
                    data = {}
                    for item, time in items:
                        data[item] = time
                    rettrn[seq].append(copy.deepcopy(data))
                    rettst[seq].append(None)
                    retusr[seq] += 1
    return rettrn, rettst, retusr

def augmentation(trnInt, usrnum, l1, l2):
    retint = []
    augcnt = [0, 0, 0]
    for idx, data in enumerate(trnInt):
        retint.append(copy.deepcopy(data))
        itmcnt = 0
        for itm in data: itmcnt += 1
        if itmcnt < l1: seq = 0
        elif itmcnt > l2: seq = 2
        else: seq = 1
        if seq == 1: 
            items = []
            for itm in data:
                items.append((itm, data[itm]))
            items = sorted(items, key=lambda x: x[1])[:l1 - np.random.randint(5) - 1] # 加random
            data = {}
            for item, time in items:
                data[item] = time
            retint.append(copy.deepcopy(data))
            augcnt[0] += 1
        else:
            if np.random.randn() > 0.2:
                items = []
                for itm in data:
                    items.append((itm, data[itm]))
                items = sorted(items, key=lambda x: x[1])[:l2 - np.random.randint(10)]
                data = {}
                for item, time in items:
                    data[item] = time
                retint.append(copy.deepcopy(data))
                augcnt[1] += 1
            elif np.random.randn() > -0.2:
                items = []
                for itm in data:
                    items.append((itm, data[itm]))
                items = sorted(items, key=lambda x: x[1])[:l1 - np.random.randint(5) - 1]
                data = {}
                for item, time in items:
                    data[item] = time
                retint.append(copy.deepcopy(data))
                augcnt[0] += 1
    usrnum += sum(augcnt)
    return retint, usrnum, augcnt

def split(interaction, usrnum, itmnum):
    pickNum = 10000
    usrPerm = np.random.permutation(usrnum)
    pickUsr = usrPerm[:pickNum]

    tstInt = [None] * usrnum
    exception = 0
    for usr in pickUsr:
        temp = list()
        data = interaction[usr]
        for itm in data:
            temp.append((itm, data[itm]))
        if len(temp) == 0:
            exception += 1
            continue
        temp.sort(key=lambda x: x[1])
        tstInt[usr] = temp[-1][0]
        interaction[usr][tstInt[usr]] = None
    # print('Exception:', exception, 'Total test samples:', np.sum(np.array(tstInt)!=None))
    return interaction, tstInt

def split_div(interaction, usrnum, itmnum):
    testNum = 10000
    tstInt = list()
    tstTime = []

    for seq in range(len(interaction)):
        pickNum = int(np.ceil(testNum * usrnum[seq] / np.sum(usrnum)))
        usrPerm = np.random.permutation(usrnum[seq])
        pickUsr = usrPerm[:pickNum]

        tstInt.append([None] * usrnum[seq])
        tstTime.append([None] * usrnum[seq])
        exception = 0
        for usr in pickUsr:
            temp = list()
            data = interaction[seq][usr]
            for itm in data:
                if data[itm] == None: continue
                temp.append((itm, data[itm]))
            if len(temp) <= 2:
                exception += 1
                continue
            temp.sort(key=lambda x: x[1]) # 从小到大
            tstInt[seq][usr] = temp[-1][0]
            tstTime[seq][usr] = interaction[seq][usr][tstInt[seq][usr]]
            interaction[seq][usr][tstInt[seq][usr]] = None

    with open(prefix + 'test_time', 'wb') as fs:
        a = []
        for i in tstTime: a = a + i
        pickle.dump(a, fs)
    log('Saved 「test_time」\(≧▽≦)/')

    # print('Exception:', exception, 'Total test samples:', np.sum(np.array(tstInt)!=None))

    return interaction, tstInt            

def dict_to_sparse(interaction, usrnum, itmnum):
    r, c, d = [list(), list(), list()]
    for usr in range(usrnum):
        if interaction[usr] == None:
            continue
        data = interaction[usr]
        for col in data:
            if data[col] != None:
                r.append(usr)
                c.append(col)
                d.append(data[col])
    retint = csr_matrix((d, (r, c)), shape=(usrnum, itmnum))
    return retint

def sparse_to_dict(interaction):
    retint = []
    A = interaction.toarray()
    usrnum, itmnum = A.shape
    for idx, row in enumerate(A):
        print('Loading %d/%d' % (idx + 1, usrnum), end='\r')
        data = {}
        items = np.argwhere(row != 0).flatten()
        for itm in items:
            data[itm] = row[itm]
        retint.append(copy.deepcopy(data))
    return retint, usrnum, itmnum

def load_from_int(path):
    with open(path, 'rb') as fs:
        trn_int = pickle.load(fs)
    trn_int, usrnum, itmnum = sparse_to_dict(trn_int)
    return trn_int, usrnum, itmnum
            
def load_from_yelp_raw(prefix):
    trn_int, usrnum, itmnum = mapping(prefix + 'yelp_review')
    log('Id Mapped, usr %d, itm %d' % (usrnum, itmnum))

    checkFuncs = [checkFunc1, checkFunc2, checkFunc3]
    for i in range(3):
        filterItem = True if i < 2 else False
        trn_int, usrnum, itmnum = filter(trn_int, usrnum, itmnum, checkFuncs[i], checkFuncs[i], filterItem)
        print('Filter', i, 'times:', usrnum, itmnum)
    log('Sparse Samples Filtered, User:%d, Item:%d' % (usrnum, itmnum))

    with open(prefix + 'trn_int', 'wb') as fs:
        pickle.dump(dict_to_sparse(trn_int, usrnum, itmnum), fs)
    log('Saved 「trn_int」\(≧▽≦)/')

    return trn_int, usrnum, itmnum

if __name__ == '__main__':

    l1, l2 = 15, 35
    dataset = input('Choose a dataset: ')
    load_from_raw = False
    prefix = './' + dataset + '/'
    seqName = ['short', 'medium', 'long']
    if dataset == 'yelp': l1, l2 = 15, 35
    if dataset == 'gowalla': l1, l2 = 10, 30
    log(f'Start! (=ﾟωﾟ)ﾉ dataset={prefix} l1={l1} l2={l2}')

    if load_from_raw:
        log('Loading from raw data')
        if dataset == 'yelp':
            trn_int, usrnum, itmnum = load_from_yelp_raw(prefix)
        log('Loaded ' + dataset + ' raw data')

    trn_int, usrnum, itmnum = load_from_int(prefix + 'trn_int')
    log('Loaded Data from trn_int')

    trn_raw_div,  usrnum_div = division_by_interaction_without_test(trn_int, usrnum, itmnum, l1, l2)
    log('Divided Sequences for Raw Data: Short:%d, Medium:%d, Long:%d' % (usrnum_div[0], usrnum_div[1], usrnum_div[2]))

    trn_raw_div, tst_raw_div = split_div(trn_raw_div, usrnum_div, itmnum)
    log('Raw Data Splited')

    trn_raw = []
    for i in range(3):
        trn_raw.append(dict_to_sparse(trn_raw_div[i], usrnum_div[i], itmnum))
    with open(prefix + 'trn_raw', 'wb') as fs:
        pickle.dump(sp.vstack(trn_raw), fs)
    with open(prefix + 'tst_raw', 'wb') as fs:
        pickle.dump(tst_raw_div[0] + tst_raw_div[1] + tst_raw_div[2], fs)
    log('Saved 「trn_raw」「tst_raw」\(≧▽≦)/')

    with open(prefix + 'tst_raw_div', 'wb') as fs:
        pickle.dump(tst_raw_div, fs)
    for seq in range(3):
        with open(prefix + 'trn_raw_' + seqName[seq], 'wb') as fs:
            pickle.dump(dict_to_sparse(trn_raw_div[seq], usrnum_div[seq], itmnum), fs)
    log('Saved 「trn_raw_short」「trn_raw_medium」「trn_raw_long」\(≧▽≦)/')

    ##### Starting The Augmentation #####

    trn_aug_div, tst_aug_div, usrnum_div = augmentation_from_raw(trn_raw_div, tst_raw_div, usrnum_div, l1, l2)
    
    # trn_aug, usrnum, aug_cnt = augmentation(trn_int, usrnum, l1, l2)
    # log('Dataset Augmented (short+%d, medium+%d, long+%d)' % (aug_cnt[0], aug_cnt[1], aug_cnt[2]))

    # l1, l2 = 15, 35
    # trn_aug_div,  usrnum_div = division_by_interaction_without_test(trn_aug, usrnum, itmnum, l1, l2)
    # log('Divided Sequences for Augmented Data: Short:%d, Medium:%d, Long:%d' % (usrnum_div[0], usrnum_div[1], usrnum_div[2]))

    # trn_aug_div, tst_aug_div = split_div(trn_aug_div, usrnum_div, itmnum)
    # log('Augmented Data Splited')

    trn_aug = []
    for i in range(3):
        trn_aug.append(dict_to_sparse(trn_aug_div[i], usrnum_div[i], itmnum))
    with open(prefix + 'trn_aug', 'wb') as fs:
        pickle.dump(sp.vstack(trn_aug), fs)
    with open(prefix + 'tst_aug', 'wb') as fs:
        pickle.dump(tst_aug_div[0] + tst_aug_div[1] + tst_aug_div[2], fs)
    log('Saved 「trn_aug」「tst_aug」\(≧▽≦)/')

    with open(prefix + 'tst_aug_div', 'wb') as fs:
        pickle.dump(tst_aug_div, fs)
    for seq in range(3):
        with open(prefix + 'trn_aug_' + seqName[seq], 'wb') as fs:
            pickle.dump(dict_to_sparse(trn_aug_div[seq], usrnum_div[seq], itmnum), fs)
    log('Saved 「trn_aug_short」「trn_aug_medium」「trn_aug_long」\(≧▽≦)/')


    np.save(prefix + 'statistics.npy', np.array([time_len, sequ_len]))

    log('Saved Statistical Data')
