import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from params import args
import utils.TimeLogger as logger
from utils.TimeLogger import log
import utils.NNLayers as NNs
from utils.NNLayers import FC, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from handler import negative_sample, sparse_transpose, DataHandler, sparse_to_tensor
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle

def set_random_seed():
    tf.random.set_random_seed(19260817)
    np.random.seed(19260817)

class Recommender:
    def __init__(self, sess, handler):
        self.sess = sess
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'HR', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepare_model()
        log('Model Prepared')

        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            init = tf.global_variables_initializer()
            self.sess.run(init)
            log('Variables Inited')
        for ep in range(stloc, args.epoch):
            test = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, test))
            if test:
                reses = self.testEpoch()
                log(self.makePrint('Test', ep, reses, test))
                self.saveHistory()
        
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        self.saveHistory()

    def predict(self):
        uids = self.uids
        iids = self.iids

        ulat = tf.nn.embedding_lookup(self.ulat, uids)
        ilat = tf.nn.embedding_lookup(self.ilat, iids)

        similarity = tf.reduce_sum(ulat * ilat, axis=-1)

        return similarity

    def calculate_ssl_loss(self):
        ssl_loss = 0

        ulat1 = tf.nn.embedding_lookup(self.ulat_ssl1, self.uids)
        ulat2 = tf.nn.embedding_lookup(self.ulat_ssl2, self.uids)
        ilat1 = tf.nn.embedding_lookup(self.ilat_ssl1, self.iids)
        ilat2 = tf.nn.embedding_lookup(self.ilat_ssl2, self.iids)

        ulat1 = tf.nn.l2_normalize(ulat1, 1)
        ulat2 = tf.nn.l2_normalize(ulat2, 1)
        ilat1 = tf.nn.l2_normalize(ilat1, 1)
        ilat2 = tf.nn.l2_normalize(ilat2, 1)

        # ulat0 = tf.transpose(tf.nn.l2_normalize(self.ulat_ssl2, 1))
        # ilat0 = tf.transpose(tf.nn.l2_normalize(self.ilat_ssl2, 1))

        ulat0 = tf.transpose(ulat2)
        ilat0 = tf.transpose(ilat2)

        pos_score = tf.exp(tf.reduce_sum(ulat1 * ulat2, axis=-1) / args.ssl_temp)
        ttl_score = tf.reduce_sum(tf.exp(ulat1 @ ulat0 / args.ssl_temp), axis=-1)
        ssl_loss += -tf.reduce_sum(tf.log(pos_score / (ttl_score + 1e-8) + 1e-8))

        pos_score = tf.exp(tf.reduce_sum(ilat1 * ilat2, axis=-1) / args.ssl_temp)
        ttl_score = tf.reduce_sum(tf.exp(ilat1 @ ilat0 / args.ssl_temp), axis=-1)
        ssl_loss += -tf.reduce_sum(tf.log(pos_score / (ttl_score + 1e-8) + 1e-8))

        return ssl_loss

    def prepare_model(self):
        self.actFunc = 'leakyRelu'

        adj = self.handler.trnMat
        self.A0 = sparse_to_tensor(adj, norm=True)
        self.A1 = sparse_to_tensor(sparse_transpose(adj), norm=True)

        self.uids = tf.placeholder(name='uids', dtype=tf.int32, shape=[None])
        self.iids = tf.placeholder(name='iids', dtype=tf.int32, shape=[None])

        self.ssl_user_mask = tf.placeholder(name='ssl_user_data', dtype=tf.int32, shape=[None])
        self.ssl_item_mask = tf.placeholder(name='ssl_item_data', dtype=tf.int32, shape=[None])
        
        self.define_model()

        self.preds = self.predict()
        sampNum = tf.shape(self.uids)[0] // 2
        posPred = tf.slice(self.preds, [0], [sampNum])
        negPred = tf.slice(self.preds, [sampNum], [-1])

        self.preLoss = tf.reduce_mean(tf.maximum(0.0, 1.0 - (posPred - negPred)))
        self.regLoss = args.reg * NNs.regularization()
        self.sslLoss = args.ssl_reg * self.calculate_ssl_loss()
        self.loss = self.preLoss + self.regLoss + self.sslLoss

        global_step = tf.Variable(0, trainable=False)
        learning_method = tf.train.exponential_decay(args.lr, global_step, args.decay_step, args.decay, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_method).minimize(self.loss, global_step=global_step)

    def define_model(self):

        uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim], reg=True)
        iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim], reg=True)

        self.ulat = 0
        self.ilat = 0
        self.ulat_ssl1 = 0
        self.ulat_ssl2 = 0
        self.ilat_ssl1 = 0
        self.ilat_ssl2 = 0

        self.generate_ssl_graphs()

        ulats = [uEmbed0]
        ilats = [iEmbed0]
        for i in range(args.gnn_layer):
            ulat = self.message_propagate(self.A0, ilats[-1])
            ilat = self.message_propagate(self.A1, ulats[-1])
            ulats.append(ulat)
            ilats.append(ilat)
            self.ulat += ulat / (i + 1)
            self.ilat += ilat / (i + 1)

        ulats = [uEmbed0]
        ilats = [iEmbed0]
        for i in range(args.gnn_layer):
            ulat = self.message_propagate(self.A10, ilats[-1])
            ilat = self.message_propagate(self.A11, ulats[-1])
            ulats.append(ulat)
            ilats.append(ilat)
            self.ulat_ssl1 += ulat / (i + 1)
            self.ilat_ssl1 += ilat / (i + 1)

        ulats = [uEmbed0]
        ilats = [iEmbed0]
        for i in range(args.gnn_layer):
            ulat = self.message_propagate(self.A20, ilats[-1])
            ilat = self.message_propagate(self.A21, ulats[-1])
            ulats.append(ulat)
            ilats.append(ilat)
            self.ulat_ssl2 += ulat / (i + 1)
            self.ilat_ssl2 += ilat / (i + 1)

    def message_propagate(self, A, lat):
        return tf.sparse.sparse_dense_matmul(A, lat)

    def generate_ssl_graphs(self):
        self.A10 = self.sparse_dropout(self.A0, self.ssl_user_mask)
        self.A20 = self.sparse_dropout(self.A0, self.ssl_user_mask)
        self.A11 = self.sparse_dropout(self.A1, self.ssl_item_mask)
        self.A21 = self.sparse_dropout(self.A1, self.ssl_item_mask)

    def sparse_dropout(self, A, mask):
        ret = tf.sparse_retain(A, mask)
        return ret

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * args.sampNum
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = negative_sample(temLabel[i], sampNum, args.item)
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur+temlen//2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur+temlen//2] = negloc
                cur += 1
        uLocs = uLocs[:cur] + uLocs[temlen//2: temlen//2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen//2: temlen//2 + cur]
        return uLocs, iLocs

    def trainEpoch(self):
        num = args.user
        sfIds = np.random.permutation(num)[:args.trnNum]
        epochLoss, epochPreLoss = [0] * 2
        num = len(sfIds)
        steps = int(np.ceil(num / args.batch))

        ssl_user_mask = args.ssl_keep_rate + np.random.uniform(size=self.A0.values.shape)
        ssl_item_mask = args.ssl_keep_rate + np.random.uniform(size=self.A1.values.shape)

        # print("=====", ssl_user_mask.shape)

        for i in range(steps):
            st = i * args.batch
            ed = min((i+1) * args.batch, num)
            batIds = sfIds[st: ed]

            uLocs, iLocs = self.sampleTrainBatch(batIds, self.handler.trnMat)

            feed_dict = {}
            feed_dict[self.uids] = uLocs
            feed_dict[self.iids] = iLocs
            feed_dict[self.ssl_user_mask] = ssl_user_mask
            feed_dict[self.ssl_item_mask] = ssl_item_mask

            target = [self.optimizer, self.preLoss, self.regLoss, self.sslLoss, self.loss]
            res = self.sess.run(target, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

            preLoss, regLoss, sslLoss, loss = res[1:]

            epochLoss += loss
            epochPreLoss += preLoss
            # print("\n", sslLoss, "\n")
            log('Step %d/%d: loss = %.2f, sslLoss = %.2f, regLoss = %.2f         ' % (i, steps, loss, sslLoss, regLoss), save=False, oneline=True)

        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps

        return ret

    def sampleTestBatch(self, batIds, labelMat):
        batch = len(batIds)
        temTst = self.handler.tstInt[batIds]
        temLabel = labelMat[batIds].toarray()
        temlen = batch * 100
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        tstLocs = [None] * batch
        cur = 0
        for i in range(batch):
            posloc = temTst[i]
            if args.pop_neg: 
                negset = []
                item_pop = self.handler.item_pop[np.argwhere(temLabel[i]==0).flatten()]
                candidates = np.argwhere(temLabel[i]==0).flatten()
                item_pop /= sum(item_pop)
                while len(negset) < 99:
                    attempt = np.random.choice(candidates, 99, replace=False, p=item_pop)
                    for item in attempt:
                        if item not in negset:
                            negset.append(item)
                negset = np.array(negset[:99])
            else:
                candidates = np.reshape(np.argwhere(temLabel[i]==0), [-1])
                negset = np.random.permutation(candidates)[:99]
            locset = np.concatenate((negset, np.array([posloc])))
            tstLocs[i] = locset
            for j in range(100):
                uLocs[cur] = batIds[i]
                iLocs[cur] = locset[j]
                cur += 1
        return uLocs, iLocs, temTst, tstLocs

    def testEpoch(self):
        epochHit, epochNdcg = [0] * 2
        ids = self.handler.tstUsrs
        num = len(ids)
        tstBat = args.batch
        steps = int(np.ceil(num / tstBat))
        for i in range(steps):
            st = i * tstBat
            ed = min((i+1) * tstBat, num)
            batIds = ids[st: ed]
            feed_dict = {}
            uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds, self.handler.trnMat)
            feed_dict[self.uids] = uLocs
            feed_dict[self.iids] = iLocs
            preds = self.sess.run(self.preds, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            hit, ndcg = self.calcRes(np.reshape(preds, [ed-st, 100]), temTst, tstLocs)
            epochHit += hit
            epochNdcg += ndcg
            log('Steps %d/%d: hit = %d, ndcg = %d          ' % (i, steps, hit, ndcg), save=False, oneline=True)
        ret = dict()
        ret['HR'] = epochHit / num
        ret['NDCG'] = epochNdcg / num
        return ret

    def calcRes(self, preds, temTst, tstLocs):
        hit = 0
        ndcg = 0
        for j in range(preds.shape[0]):
            predvals = list(zip(preds[j], tstLocs[j]))
            predvals.sort(key=lambda x: x[0], reverse=True)
            shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
            if temTst[j] in shoot:
                hit += 1
                ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
        return hit, ndcg
    
    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        saver = tf.train.Saver()
        saver.save(self.sess, 'Models/' + args.save_path)
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        saver = tf.train.Saver()
        saver.restore(sess, 'Models/' + args.load_model)
        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded') 

if __name__ == '__main__':
    logger.saveDefault = True
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    set_random_seed()

    log('Start! (=ﾟωﾟ)ﾉ')
    handler = DataHandler()
    handler.load_data()
    log('Loading Data!')

    with tf.Session(config=config) as sess:
        recom = Recommender(sess, handler)
        recom.run()
