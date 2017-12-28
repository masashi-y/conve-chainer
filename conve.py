#!/usr/bin/env python
import argparse
from collections import defaultdict

import numpy as np

import logging
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import chainer.optimizers as O
from chainer import reporter
from chainer import training
from chainer.training import extensions

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
log_path = 'log'
file_handler = logging.FileHandler(log_path)
fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)

class ConvE(chainer.Chain):

    def __init__(self, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = 200

        with self.init_scope():
            self.emb_e = L.EmbedID(
                    num_entities, self.embedding_dim, initialW=I.GlorotNormal())
            self.emb_rel = L.EmbedID(
                    num_relations, self.embedding_dim, initialW=I.GlorotNormal())
            self.conv1 = L.Convolution2D(1, 32, 3, stride=1, pad=0)
            self.bias = L.EmbedID(num_entities, 1)
            self.fc = L.Linear(10368, self.embedding_dim)
            self.bn0 = L.BatchNormalization(1)
            self.bn1 = L.BatchNormalization(32)
            self.bn2 = L.BatchNormalization(self.embedding_dim)

    def loss_fun(self, probs, Y):
        losses = Y * F.log(probs) + (1 - Y) * F.log(1 - probs)
        loss = - F.average(losses)
        return loss

    def __call__(self, e1, rel, e2, Y, flt):
        """
        Input:
            e1, rel, e2: ids for each entity and relation (shape : (batchsize,) )
            Y: whether true (1) or negative (0) sample (shape : (batchsize,) )
        Output:
            loss ( float )
        """
        if chainer.config.train:
            probs = self.forward(e1, rel, e2)
            loss = self.loss_fun(probs, Y)
            reporter.report({'loss': loss}, self)
            return loss
        else:
            assert flt is not None
            batch_size, = e1.shape
            probs_all = self.forward(e1, rel, e2)
            rank_all = self.xp.argsort(-probs_all.data)
            probs_all_flt = probs_all * flt
            rank_all_flt = self.xp.argsort(-probs_all_flt.data)
            mrr = 0.
            mrr_flt = 0.
            for i in range(batch_size):
                rank = self.xp.where(rank_all[i] == e2[i])[0][0] + 1
                mrr += 1. / rank
                rank_flt = self.xp.where(rank_all_flt[i] == e2[i])[0][0] + 1
                mrr_flt += 1. / rank_flt
            mrr /= float(batch_size)
            mrr_flt /= float(batch_size)
            probs = probs_all[self.xp.arange(batch_size), e2]
            loss = self.loss_fun(probs, Y)
            reporter.report({'loss': loss, 'mrr': mrr, 'mrr(flt)': mrr_flt}, self)
            return loss

    def forward(self, e1, rel, e2):
        """
        Input:
            e1, rel, e2: ids for each entity and relation (shape : (batchsize,) )
        Output:
            score (shape : (batchsize,) )
        """
        batch_size, = e1.shape
        e1_embedded = self.emb_e(e1).reshape(batch_size, 1, 10, 20)
        rel_embedded = self.emb_rel(rel).reshape(batch_size, 1, 10, 20)

        stacked_inputs = F.concat([e1_embedded, rel_embedded], axis=2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = F.dropout(stacked_inputs, 0.2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        x = F.dropout(x, 0.3)
        x = self.bn2(x)
        x = F.relu(x)
        if chainer.config.train:
            e2_embedded = self.emb_e(e2)
            bias = self.bias(e2).reshape((-1,))
            x *= e2_embedded
            x = F.sum(x, axis=1) + bias
            pred = F.sigmoid(x)
            return pred
        else:
            x = F.matmul(x, self.emb_e.W, transb=True)
            x, bias = F.broadcast(x, self.bias.W.T)
            x += bias
            pred = F.sigmoid(x)
            return pred


class Vocab(object):
    def __init__(self):
        self.id2word = []
        self.word2id = {}

    def add(self, word):
        if word not in self.id2word:
            self.word2id[word] = len(self.id2word)
            self.id2word.append(word)
        return self.word2id[word]

    def __len__(self):
        return len(self.id2word)

    def __getitem__(self, word):
        return self.word2id[word]

    @classmethod
    def load(cls, vocab_path):
        v = Vocab()
        with open(vocab_path) as f:
            for word in f:
                v.add(word.strip())
        return v


class TripletDataset(chainer.dataset.DatasetMixin):
    def __init__(self, ent_vocab, rel_vocab, path, negative, flt_graph=None):
        self.path = path
        logger.info("creating TripletDataset for: {}".format(self.path))
        self.negative = negative
        self.entities = ent_vocab
        self.relations = rel_vocab
        self.data = []
        if flt_graph is not None:
            logger.info("filtered on")
            self.filtered = True
            self.graph = flt_graph
        else:
            logger.info("filtered off")
            self.filtered = False
            self.graph = defaultdict(list)
        self.load_from_path()

    def __len__(self):
        return len(self.data)

    def load_from_path(self):
        logger.info("start loading dataset")
        for line in open(self.path):
            e1, rel, e2 = line.strip().split("\t")
            id_e1 = self.entities[e1]
            id_e2 = self.entities[e2]
            id_rel = self.relations[rel]
            self.data.append((id_e1, id_rel, id_e2))
            self.graph[id_e1, id_rel].append(id_e2)
        logger.info("done")
        self.num_entities = len(self.entities)
        self.num_relations = len(self.relations)
        logger.info("num entities: {}".format(self.num_entities))
        logger.info("num relations: {}".format(self.num_relations))

    def get_example(self, i):
        triplet = self.data[i]
        triplets = np.asarray(triplet * (1 + self.negative), 'i').reshape((-1, 3))
        neg_ents = np.random.randint(0, self.num_entities, size=self.negative)
        head_or_tail = 2 * np.random.randint(0, 2, size=self.negative)
        triplets[np.arange(1, self.negative+1), head_or_tail] = neg_ents
        e1, rel, e2 = zip(*triplets)
        Y = np.zeros(1 + self.negative, 'i')
        Y[0] = 1
        if self.filtered:
            e1_id, rel_id, e2_id = triplet
            flt = np.ones(self.num_entities, 'f')
            flt[self.graph[e1_id, rel_id]] = 0.
            flt[e2_id] = 1.
        else:
            flt = None
        return e1, rel, e2, Y, flt


def convert(batch, device):
    e1, rel, e2, Y, flt = zip(*batch)
    e1  = np.concatenate(e1)
    rel = np.concatenate(rel)
    e2  = np.concatenate(e2)
    Y   = np.concatenate(Y)
    flt = np.vstack(flt) if flt[0] is not None else None
    if device >= 0:
        e1  = cuda.to_gpu(e1)
        rel = cuda.to_gpu(rel)
        e2  = cuda.to_gpu(e2)
        Y   = cuda.to_gpu(Y)
        if flt is not None:
            flt = cuda.to_gpu(flt)
    return e1, rel, e2, Y, flt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='Path to training triplet list file')
    parser.add_argument('val', help='Path to validation triplet list file')
    parser.add_argument('ent_vocab', help='Path to entity vocab')
    parser.add_argument('rel_vocab', help='Path to relation vocab')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', type=int, default=1000,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--negative-size', default=10, type=int,
                        help='number of negative samples')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    parser.add_argument('--val-iter', type=int, default=1000,
                        help='validation iteration')
    args = parser.parse_args()

    logger.info('train: {}'.format(args.train))
    logger.info('val: {}'.format(args.val))
    logger.info('gpu: {}'.format(args.gpu))
    logger.info('batchsize: {}'.format(args.batchsize))
    logger.info('epoch: {}'.format(args.epoch))
    logger.info('negative-size: {}'.format(args.negative_size))
    logger.info('out: {}'.format(args.out))

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()

    ent_vocab = Vocab.load(args.ent_vocab)
    rel_vocab = Vocab.load(args.rel_vocab)

    train = TripletDataset(ent_vocab, rel_vocab, args.train, args.negative_size)
    val = TripletDataset(ent_vocab, rel_vocab, args.val, 0, train.graph)

    model = ConvE(train.num_entities, train.num_relations)

    if args.gpu >= 0:
        model.to_gpu()

    # Set up an optimizer
    optimizer = O.Adam() # (alpha=0.003)
    optimizer.setup(model)

    # Set up an iterator
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(val, args.batchsize, repeat=False)

    # Set up an updater
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)

    # Set up a trainer
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    val_interval = args.val_iter, 'iteration'
    log_interval = 10, 'iteration'

    trainer.extend(extensions.Evaluator(val_iter, model,
        converter=convert, device=args.gpu), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss',
        'validation/main/loss', 'validation/main/mrr', 'validation/main/mrr(flt)']), trigger=log_interval)
    trainer.extend(extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    main()
