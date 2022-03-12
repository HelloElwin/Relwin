########################
# adapted from: akaxlh #
########################
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
        self.prepareModel()
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
            # if ep % args.tstEpoch == 0:
                # self.saveHistory()
            # print()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        # self.saveHistory()

    def predict(self):
        uids = self.uids
        iids = self.iids

        ulat = tf.nn.embedding_lookup(self.ulat, uids)
        ilat = tf.nn.embedding_lookup(self.ilat, iids)

        similarity = tf.reduce_sum(ulat * ilat, axis=-1)

        return similarity

    def prepareModel(self):
        self.actFunc = 'leakyRelu'

        adj = self.handler.trnMat
        self.A0 = sparse_to_tensor(adj, norm=True)
        self.A1 = sparse_to_tensor(sparse_transpose(adj), norm=True)

        self.uids = tf.placeholder(name='uids', dtype=tf.int32, shape=[None])
        self.iids = tf.placeholder(name='iids', dtype=tf.int32, shape=[None])
        
        self.defineModel()

        self.preds = self.predict()
        sampNum = tf.shape(self.uids)[0] // 2
        posPred = tf.slice(self.preds, [0], [sampNum])
        negPred = tf.slice(self.preds, [sampNum], [-1])

        self.preLoss = tf.reduce_mean(tf.maximum(0.0, 1.0 - (posPred - negPred)))
        self.regLoss = args.reg * NNs.regularization()
        self.loss = self.preLoss + self.regLoss

        global_step = tf.Variable(0, trainable=False)
        learning_method = tf.train.exponential_decay(args.lr, global_step, args.decay_step, args.decay, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_method).minimize(self.loss, global_step=global_step)

    def defineModel(self):

        uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim], reg=True)
        iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim], reg=True)

        self.ulat = 0
        self.ilat = 0

        ulats = [uEmbed0]
        ilats = [iEmbed0]

        for i in range(args.gnn_layer):
            ulat = self.message_propagate(self.A0, ilats[-1])
            ilat = self.message_propagate(self.A1, ulats[-1])
            ulats.append(ulat)
            ilats.append(ilat)
            self.ulat += ulat / (i + 1)
            self.ilat += ilat / (i + 1)

    def message_propagate(self, A, lat):
        return tf.sparse.sparse_dense_matmul(A, lat)

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

        for i in range(steps):
            st = i * args.batch
            ed = min((i+1) * args.batch, num)
            batIds = sfIds[st: ed]

            uLocs, iLocs = self.sampleTrainBatch(batIds, self.handler.trnMat)

            feed_dict = {}
            feed_dict[self.uids] = uLocs
            feed_dict[self.iids] = iLocs

            target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
            res = self.sess.run(target, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

            preLoss, regLoss, loss = res[1:]

            epochLoss += loss
            epochPreLoss += preLoss
            log('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, regLoss), save=False, oneline=True)

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
            negset = np.reshape(np.argwhere(temLabel[i]==0), [-1])
            rdnNegSet = np.random.permutation(negset)[:99]
            locset = np.concatenate((rdnNegSet, np.array([posloc])))
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

    log('Start! (=ﾟωﾟ)ﾉ')
    handler = DataHandler()
    handler.load_data()
    log('Loading Data!')

    with tf.Session(config=config) as sess:
        recom = Recommender(sess, handler)
        recom.run()
