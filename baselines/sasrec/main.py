import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from params import args
import utils.TimeLogger as logger
from utils.TimeLogger import log
import utils.NNLayers as NNs
from utils.NNLayers import FC, Dropout, Bias, getParam, defineParam, defineRandomNameParam
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
            
        reses = self.test()
        log(self.makePrint('Test', 0, reses, True))

        for ep in range(stloc, args.epoch):
            test = (ep % args.tstEpoch == 0)
            reses = self.train()
            log(self.makePrint('Train', ep, reses, test))
            if test:
                reses = self.test()
                log(self.makePrint('Test', ep, reses, test))
                self.saveHistory()
        self.saveHistory()
        reses = self.test()
        log(self.makePrint('Test', args.epoch, reses, True))
        sys.stdout.flush()

    def prepare_model(self):
        self.actFunc = 'leakyRelu'
        self.is_training = tf.placeholder(name='is_training', dtype=tf.bool, shape=[])

        self.seqs = self.handler.seqs # (num_user, length_of_user_interactions)

        self.seq = tf.placeholder(name='seq', dtype=tf.int32, shape=[None]) # (? * num_his)
        self.pos = tf.placeholder(name='pos', dtype=tf.int32, shape=[None]) # (? * num_his)
        self.neg = tf.placeholder(name='neg', dtype=tf.int32, shape=[None]) # (? * num_his)

        self.define_model()
        self.pos_score, self.neg_score = self.predict() # (?, num_his), (?, num_his)
        self.loss = self.calculate_loss(self.pos_score, self.neg_score)

        global_step = tf.Variable(0, trainable=False)
        learning_method = tf.train.exponential_decay(args.lr, global_step, args.decay_step, args.decay, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_method).minimize(self.loss, global_step=global_step)

    def predict(self):
        poss = tf.reshape(self.pos, [-1, args.num_his])
        negs = tf.reshape(self.neg, [-1, args.num_his])

        seq_emb = self.seq_emb # (? * num_his, latdim)
        pos_emb = tf.nn.embedding_lookup(self.item_emb, self.pos) # (? * num_his, latdim)
        neg_emb = tf.nn.embedding_lookup(self.item_emb, self.neg) # (? * num_his, latdim)

        pos_score = tf.reduce_sum(seq_emb * pos_emb, axis=-1) # (? * num_his)
        neg_score = tf.reduce_sum(seq_emb * neg_emb, axis=-1) # (? * num_his) 

        neg_emb = tf.reshape(neg_emb, [-1, 100, args.latdim]) # (?, num_test, latdim)
        seq_emb = tf.reshape(seq_emb, [-1, args.num_his, args.latdim]) # (?, num_his, latdim)
        seq_emb = tf.slice(seq_emb, [0, args.num_his - 1, 0], [-1, -1, -1]) # (?, 1, latdim)

        self.pred = tf.reduce_sum(seq_emb * neg_emb, axis=-1) # for test only (?, 100)
        # print(self.pred.shape)

        return pos_score, neg_score

    def calculate_loss(self, pos, neg):
        is_tar = tf.cast(self.pos != 0, tf.float32)

        self.pred_loss = -tf.log(tf.sigmoid(pos) + 1e-8) * is_tar - tf.log(1 - tf.sigmoid(neg) + 1e-8) * is_tar
        self.pred_loss = tf.reduce_sum(self.pred_loss) / tf.reduce_sum(is_tar)
        self.regu_loss = args.reg * NNs.regularization() 
        
        loss = self.pred_loss + self.regu_loss

        return loss
        
    def define_model(self):
        self.item_emb = NNs.defineParam('item_emb', [args.item + 1, args.latdim], reg=True) # 0 for padding
        self.posi_emb = NNs.defineParam('posi_emb', [args.num_his,  args.latdim], reg=True)

        item_emb0 = tf.nn.embedding_lookup(self.item_emb, self.seq)
        item_emb0 = tf.reshape(item_emb0, [-1, args.num_his, args.latdim]) + self.posi_emb
        item_emb0 = tf.reshape(item_emb0, [-1, args.latdim])
        layer_emb = [item_emb0]

        # print(layer_emb[0].shape,'<<<<<<<<')
        for i in range(args.num_layers):
            new_embed = tf.reshape(layer_emb[-1], [-1, args.num_his, args.latdim])
            new_embed = NNs.self_attention(NNs.layer_norm(new_embed), args.num_his, args.latdim, args.num_heads, causality=True, dropout=args.dropout, is_training=self.is_training)
            new_embed = tf.reshape(new_embed, [-1, args.latdim])
            new_embed = NNs.activate(NNs.FC(new_embed, args.latdim, bias=True, reg=True, bias_reg=True), 'relu')
            new_embed = NNs.FC(new_embed, args.latdim, bias=True, reg=True, bias_reg=True)
            layer_emb = layer_emb + [new_embed + layer_emb[-1]]
        
        self.seq_emb = tf.reshape(layer_emb[-1], [-1, args.latdim])

    def sample_train(self, batch_uids):
        batch_rows = self.handler.trn[batch_uids]
        batch_seqs = self.handler.seqs[batch_uids] # an array of lists
        batch_size = len(batch_uids)

        seq, pos, neg = [], [], []

        for i in range(batch_size):
            if len(batch_seqs[i]) <= args.num_his:
                padding = args.num_his - (len(batch_seqs[i]) - 1)
                cut_seq = [0] * padding + batch_seqs[i][:-1]
                pos_seq = cut_seq[1:] + [batch_seqs[i][-1]]
                neg_seq = np.random.permutation(np.argwhere(batch_rows[i]==0).flatten())[:args.num_his].tolist()
                pos_seq[padding - 1] = 0
            else:
                cut_cut = np.random.randint(len(batch_seqs[i]) - args.num_his)
                cut_seq = batch_seqs[i][cut_cut:cut_cut + args.num_his]
                pos_seq = cut_seq[1:] + [batch_seqs[i][-1]]
                neg_seq = np.random.permutation(np.argwhere(batch_rows[i]==0).flatten())[:args.num_his].tolist()

            seq += cut_seq
            pos += pos_seq
            neg += neg_seq

        return seq, pos, neg

    def train(self):
        uids = np.random.permutation(args.user)[:args.trnNum]
        epochLoss, epochPreLoss = [0] * 2
        steps = int(np.ceil(args.trnNum / args.batch))

        for i in range(steps):
            st = i * args.batch
            ed = min((i + 1) * args.batch, args.trnNum)
            batch_uids = uids[st: ed]

            seq, pos, neg = self.sample_train(batch_uids)

            feed_dict = {}
            feed_dict[self.seq] = seq 
            feed_dict[self.pos] = pos
            feed_dict[self.neg] = neg
            feed_dict[self.is_training] = True

            target = [self.optimizer, self.pred_loss, self.regu_loss, self.loss]
            res = self.sess.run(target, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

            preLoss, regLoss, loss = res[1:]

            epochLoss += loss
            epochPreLoss += preLoss
            # print("\n", sslLoss, "\n")
            log('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, regLoss), save=False, oneline=True)

        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps

        return ret

    def sample_test(self, batch_uids):
        batch_seqs = self.handler.seqs[batch_uids]
        batch_rows = self.handler.trn[batch_uids]
        batch_test = self.handler.tst[batch_uids]
        batch_size = len(batch_uids)

        seq, pos, neg = [], [], []

        for i in range(batch_size):
            if len(batch_seqs[i]) < args.num_his:
                padding = args.num_his - len(batch_seqs[i])
                cut_seq = [0] * padding + batch_seqs[i]
                # neg_seq = np.random.permutation(np.argwhere(batch_rows[i]==0).flatten())[:99].tolist() + [batch_test[i]]
            else:
                cut_seq = batch_seqs[i][-args.num_his:]
                # neg_seq = np.random.permutation(np.argwhere(batch_rows[i]==0).flatten())[:99].tolist() + [batch_test[i]] 

            if args.pop_neg:
                negset = []
                item_pop = self.handler.item_pop[np.argwhere(batch_rows[i]==0).flatten()]
                candidates = np.argwhere(batch_rows[i]==0).flatten()
                item_pop /= sum(item_pop)
                while len(negset) < 99:
                    attempt = np.random.choice(candidates, 99, replace=False, p=item_pop)
                    for item in attempt:
                        if item not in negset:
                            negset.append(item)
                negset = np.array(negset[:99])
            else:
                candidates = np.reshape(np.argwhere(batch_rows[i]==0), [-1])
                negset = np.random.permutation(candidates)[:99]

            neg_seq = negset.tolist() + [batch_test[i]]

            seq += cut_seq
            neg += neg_seq

        return seq, pos, neg, batch_test

    def test(self):
        epochHit, epochNdcg = [0] * 2
        uids = self.handler.tstUsrs
        num_test = len(uids)
        steps = int(np.ceil(num_test / args.batch))
        for i in range(steps):
            st = i * args.batch
            ed = min((i + 1) * args.batch, num_test)
            batch_uids = uids[st: ed]
            seq, pos, neg, temTst = self.sample_test(batch_uids)
            feed_dict = {}
            feed_dict[self.seq] = seq 
            feed_dict[self.pos] = pos 
            feed_dict[self.neg] = neg 
            feed_dict[self.is_training] = False
            pred = self.sess.run(self.pred, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            hit, ndcg = self.calculate_metrics(np.reshape(pred, [-1, 100]), temTst, np.reshape(np.array(neg), [-1, 100]))
            epochHit += hit
            epochNdcg += ndcg
            log('Steps %d/%d: hit = %d, ndcg = %d          ' % (i, steps, hit, ndcg), save=False, oneline=True)
        ret = dict()
        ret['HR'] = epochHit / num_test
        ret['NDCG'] = epochNdcg / num_test
        return ret

    def calculate_metrics(self, preds, temTst, tstLocs):
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
