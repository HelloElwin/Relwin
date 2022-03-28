########################
# adapted from: akaxlh #
########################
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from params import args
import scipy.sparse as sp
from utils.TimeLogger import log
import tensorflow as tf

def sparse_transpose(mat):
    coomat = sp.coo_matrix(mat)
    return csr_matrix(coomat.transpose())

def negative_sample(temLabel, sampSize, nodeNum):
    negset = [None] * sampSize
    cur = 0
    while cur < sampSize:
        rdmItm = np.random.choice(nodeNum)
        if temLabel[rdmItm] == 0:
            negset[cur] = rdmItm
            cur += 1
    return negset

def sparse_to_tensor(mat, mask=False, norm=False):
    shape = [mat.shape[0], mat.shape[1]]
    coomat = sp.coo_matrix(mat)
    indices = np.array(list(map(list, zip(coomat.row, coomat.col))), dtype=np.int32)
    data = coomat.data.astype(np.float32)

    if norm:
        rowD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=1) + 1e-8) + 1e-8)))
        colD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=0) + 1e-8) + 1e-8)))
        for i in range(len(data)):
            row = indices[i, 0]
            col = indices[i, 1]
            data[i] = data[i] * rowD[row] * colD[col]

    # half mask
    if mask:
        spMask = (np.random.uniform(size=data.shape) > 0.5) * 1.0
        data = data * spMask

    if indices.shape[0] == 0:
        indices = np.array([[0, 0]], dtype=np.int32)
        data = np.array([0.0], np.float32)

    return tf.sparse.SparseTensor(indices, data, shape)

class DataHandler:
    def __init__(self):
        self.prefix = './datasets/' + args.data + '/'

    def load_data(self):

        # interaction matrix
        with open(self.prefix + 'trn_raw', 'rb') as fs:
            self.trn = pickle.load(fs).astype(np.float32).toarray()

        # train data
        with open(self.prefix + 'trn', 'rb') as fs:
            self.seqs = np.array(pickle.load(fs))

        # test data
        with open(self.prefix + 'tst', 'rb') as fs:
            self.tst = np.array(pickle.load(fs))
        self.tstUsrs = np.argwhere(self.tst != None).flatten()

        #finished
        args.user, args.item = self.trn.shape
        log('Loaded User: %d Item: %d' % (args.user, args.item))
