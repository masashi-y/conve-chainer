#!/usr/bin/env python
import argparse
from collections import defaultdict

import numpy as np
import os, sys
from tqdm import tqdm
import logging
import chainer
from chainer import cuda
from chainer import serializers
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import chainer.optimizers as O
from chainer import reporter
from chainer import training
from chainer.training import extensions

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

IGNORE = -1
UNK = "<unk>"


def concat_arrays(arrays, length=None):
    if length is None:
        length = max(len(a) for a in arrays)
    res = -np.ones((len(arrays), length), 'i')
    for i, array in enumerate(arrays):
        for j, v in enumerate(array):
            res[i, j] = v
    return res


def load_pretrained_embeddings(filepath):
    io = open(filepath)
    dim = len(io.readline().split())
    io.seek(0)
    nvocab = sum(1 for line in io)
    io.seek(0)
    res = np.empty((nvocab, dim), dtype=np.float32)
    for i, line in enumerate(io):
        line = line.strip()
        if len(line) == 0: continue
        res[i] = line.split()
    io.close()
    return res

class Evaluator(object):
    def __init__(self):
        self.total = 0
        self.mrr = 0.
        self.mrr_flt = 0.
        self.hits1 = 0.
        self.hits3 = 0.
        self.hits10 = 0.

    def process_batch(self, xp, probs, e2, flt):
        """
        Input:
            probs: probability matrix (shape: (batch_size, num_entities) )
            e2: gold e2's (shape: (batch_size,) )
            flt: {0,1} filters for filtered scores (shape: (batch_size, num_entities) )
        Output:
            MRR, filtered MRR, filtered HITs (1, 3, 10)
        """
        batch_size, = e2.shape
        rank_all = xp.argsort(-probs.data)
        probs_flt = probs * flt
        rank_all_flt = xp.argsort(-probs_flt.data)
        for i in range(batch_size):
            rank = xp.where(rank_all[i] == e2[i])[0][0] + 1
            self.mrr += 1. / int(rank)
            rank_flt = xp.where(rank_all_flt[i] == e2[i])[0][0] + 1
            self.mrr_flt += 1. / int(rank_flt)
            if rank_flt <= 1:
                self.hits1 += 1
            if rank_flt <= 3:
                self.hits3 += 1
            if rank_flt <= 10:
                self.hits10 += 1
        self.total += batch_size

    def results(self):
        total = float(self.total)
        print(total)
        mrr = self.mrr / total
        mrr_flt = self.mrr_flt / total
        hits1 = self.hits1 / total
        hits3 = self.hits3 / total
        hits10 = self.hits10 / total
        return {'mrr': mrr,
                'mrr(flt)': mrr_flt,
                'hits1(flt)': hits1,
                'hits3(flt)': hits3,
                'hits10(flt)': hits10}


class BaseModel(object):
    def binary_cross_entropy(self, probs, Y):
        """
        Input:
            probs: probability matrix in any shape
            Y {0, 1} matrix in the same shape as probs
        Output:
            scalar loss
        """
        losses = Y * F.log(probs + 1e-6) + (1 - Y) * F.log(1 - probs + 1e-6)
        loss = - F.average(losses)
        return loss

    def __call__(self, e1, char_e1, rel, e2, char_e2, Y, flt):
        probs = self.forward(char_e1, rel, char_e2)
        loss = self.binary_cross_entropy(probs, Y)
        reporter.report({'loss': loss}, self)
        return loss


class ComplEx(chainer.Chain, BaseModel):
    def __init__(self, num_chars, num_entities, num_relations,
                char_dim, embedding_dim=200, validation=None):
        super(ComplEx, self).__init__()
        self.char_dim = char_dim
        self.num_chars = num_chars
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.validation = concat_arrays(validation)
        if self.validation is not None:
            self.validation = cuda.to_gpu(self.validation)

        with self.init_scope():
            self.emb_char = L.EmbedID(num_chars, char_dim,
                    initialW=I.GlorotNormal(), ignore_label=IGNORE)
            self.conv_char=L.Convolution2D(1, embedding_dim * 2, # for real & img
                        (3, char_dim), stride=1, pad=(1, 0))
            self.emb_rel_real = L.EmbedID(
                    num_relations, embedding_dim, initialW=I.GlorotNormal())
            self.emb_rel_img = L.EmbedID(
                    num_relations, embedding_dim, initialW=I.GlorotNormal())

    def embed_fast(self, char_e1, char_e2):
        batch_e1, seq_len = char_e1.shape
        batch_e2, _ = char_e2.shape
        assert char_e1.shape == char_e2.shape
        batchsize = batch_e1 + batch_e2
        e1, e2 = F.split_axis(F.reshape(
            F.max_pooling_2d( # (batchsize, embedding_dim * 4, 1, 1)
            self.conv_char( # (batchsize, embedding_dim * 4, seqlen, 1)
                F.expand_dims(
                self.emb_char( # (batchsize, seq_len, char_dim)
                    F.concat([char_e1, char_e2], axis=0)), 1)),
                ksize=(seq_len, 1)),
            (batchsize, self.embedding_dim*2)),
            [batch_e1], 0)
        return e1, e2

    def embed_separate(self, e):
        batchsize, seq_len = e.shape
        return F.reshape(
                F.max_pooling_2d( # (batchsize, embedding_dim * 4, 1, 1)
            self.conv_char( # (batchsize, embedding_dim * 4, seqlen, 1)
                F.expand_dims(
                self.emb_char( # (batchsize, seq_len, char_dim)
                    e), 1)),
                ksize=(seq_len, 1)),
            (batchsize, self.embedding_dim*2))

    def forward(self, char_e1, rel, char_e2, validation=False):
        """ train """
        if validation:
            e1 = self.embed_separate(char_e1)
            e2 = self.embed_separate(self.validation)
            assert e2.shape[0] == self.num_entities
        else:
            e1, e2 = self.embed_fast(char_e1, char_e2)

        e1_real, e1_img = F.split_axis(e1, 2, 1)
        e2_real, e2_img = F.split_axis(e2, 2, 1)
        rel_real = self.emb_rel_real(rel)
        rel_img = self.emb_rel_img(rel)

        e1_real = F.dropout(e1_real, 0.2)
        e1_img = F.dropout(e1_img, 0.2)
        e2_real = F.dropout(e2_real, 0.2)
        e2_img = F.dropout(e2_img, 0.2)
        rel_real = F.dropout(rel_real, 0.2)
        rel_img = F.dropout(rel_img, 0.2)

        if not validation:
            realrealreal = e1_real * rel_real * e2_real
            realimgimg = e1_real * rel_img * e2_img
            imgrealimg = e1_img * rel_real * e2_img
            imgimgreal = e1_img * rel_img * e2_real
            pred = realrealreal + realimgimg + imgrealimg - imgimgreal
            pred = F.sigmoid(F.sum(pred, 1))
            return pred
        else:
            realrealreal = F.matmul(e1_real * rel_real, e2_real, transb=True)
            realimgimg = F.matmul(e1_real * rel_img, e2_img, transb=True)
            imgrealimg = F.matmul(e1_img * rel_real, e2_img, transb=True)
            imgimgreal = F.matmul(e1_img * rel_img, e2_real, transb=True)
            pred = realrealreal + realimgimg + imgrealimg - imgimgreal
            pred = F.sigmoid(pred)
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
    def __init__(self, char_vocab, ent_vocab, rel_vocab, path, negative, validation=False):
        self.path = path
        logger.info("creating CharacterTripletDataset for: {}".format(self.path))
        self.chars = char_vocab
        assert UNK in self.chars.word2id
        self.unk = self.chars[UNK]
        self.entities = ent_vocab
        self.relations = rel_vocab
        self.data = []
        self.graph = defaultdict(list)
        self.negative = negative
        self.validation = validation
        self.load_from_path()

    def __len__(self):
        return len(self.data)

    def embed(self, entity):
        return [self.chars[c] for c in entity]

    def load_from_path(self):
        logger.info("start loading dataset")
        for line in open(self.path):
            e1, rel, e2 = line.strip().split("\t")
            e1 = self.entities[e1]
            e2 = self.entities[e2]
            rel = self.relations[rel]
            self.data.append((e1, rel, e2))
            self.graph[e1, rel].append(e2)
        logger.info("done")
        self.num_entities = len(self.entities)
        self.num_relations = len(self.relations)
        self.num_chars = len(self.chars)
        self.e2s = [self.embed(e) for e in self.entities.id2word]
        logger.info("num samples: {}".format(len(self)))
        logger.info("num entities: {}".format(self.num_entities))
        logger.info("num relations: {}".format(self.num_relations))

    def get_example(self, i):
        e1_id, rel_id, e2_id = self.data[i]
        triplets = np.asarray((e1_id, rel_id, e2_id) * (1 + self.negative), 'i').reshape((-1, 3))
        neg_ents = np.random.randint(0, self.num_entities, size=self.negative)
        head_or_tail = 2 * np.random.randint(0, 2, size=self.negative)
        triplets[np.arange(1, self.negative+1), head_or_tail] = neg_ents
        e1, rel, e2 = zip(*triplets)
        Y = np.zeros(1 + self.negative, 'i')
        char_e1 = [self.e2s[e] for e in e1]
        char_e2 = [self.e2s[e] for e in e2]
        Y[0] = 1

        if self.validation:
            char_e2 = None
            flt = np.ones(self.num_entities, 'f')
            flt[self.graph[e1_id, rel_id]] = 0.
            flt[e2_id] = 1.
        else:
            flt = None
        return e1, char_e1, rel, e2, char_e2, Y, flt


def convert(batch, device):
    e1, char_e1, rel, e2, char_e2, Y, flt = zip(*batch)
    e1  = np.concatenate(e1)
    rel = np.concatenate(rel)
    e2  = np.concatenate(e2)
    if char_e2[0] is not None:
        length = max(max(len(e) for ce in char_e1 for e in ce),
                max(len(e) for ce in char_e2 for e in ce))
    else:
        char_e2 = None
        length = max(len(e) for ce in char_e1 for e in ce)
    char_e1 = concat_arrays([e for ce in char_e1 for e in ce], length=length)
    char_e2 = concat_arrays([e for ce in char_e2 for e in ce],
                length=length) if char_e2 is not None else None
    Y   = np.concatenate(Y)
    flt = np.vstack(flt) if flt[0] is not None else None
    if device >= 0:
        e1  = cuda.to_gpu(e1)
        rel = cuda.to_gpu(rel)
        e2  = cuda.to_gpu(e2)
        char_e1  = cuda.to_gpu(char_e1)
        char_e2  = cuda.to_gpu(char_e2) \
                if char_e2 is not None else None
        Y   = cuda.to_gpu(Y)
        if flt is not None:
            flt = cuda.to_gpu(flt)
    return e1, char_e1, rel, e2, char_e1, Y, flt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='Path to training triplet list file')
    parser.add_argument('val', help='Path to validation triplet list file')
    parser.add_argument('char_vocab', help='Path to character vocab')
    parser.add_argument('ent_vocab', help='Path to entity vocab')
    parser.add_argument('rel_vocab', help='Path to relation vocab')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', type=int, default=1000,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--char-dim', '-c', default=30, type=int,
                        help='the dimension of character embedding')
    parser.add_argument('--negative-size', default=10, type=int,
                        help='number of negative samples')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    parser.add_argument('--val-iter', type=int, default=1000,
                        help='validation iteration')
    parser.add_argument('--init-model', default=None,
                        help='initialize model with saved one')
    parser.add_argument('--expand-graph', action='store_true')
    parser.add_argument('--fast-eval', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.mkdir(args.out)
    log_path = os.path.join(args.out, 'loginfo')
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

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

    char_vocab = Vocab.load(args.char_vocab)
    ent_vocab = Vocab.load(args.ent_vocab)
    rel_vocab = Vocab.load(args.rel_vocab)

    train = TripletDataset(char_vocab, ent_vocab, rel_vocab, args.train, args.negative_size)
    val = TripletDataset(char_vocab, ent_vocab, rel_vocab, args.val, 0, validation=True)

    model = ComplEx(train.num_chars, train.num_entities,
            train.num_relations, args.char_dim, 200, validation=train.e2s)

    if args.init_model:
        logger.info("initialize model with: {}".format(args.init_model))
        serializers.load_npz(args.init_model, model)
    if args.gpu >= 0:
        model.to_gpu()

    optimizer = O.Adam() # (alpha=0.003)
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(val, args.batchsize, repeat=False)

    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    val_interval = args.val_iter, 'iteration'
    log_interval = 100, 'iteration'

    @chainer.training.make_extension()
    def evaluate(trainer):
        evaluator = Evaluator()
        if hasattr(val_iter, 'reset'):
            val_iter.reset()
        for batch in val_iter:
            _, char_e1, rel, e2, _, _, flt = convert(batch, args.gpu)
            probs = model.forward(char_e1, rel, None, validation=True)
            evaluator.process_batch(model.xp, probs, e2, flt)
        metrics = evaluator.results()
        print(metrics, file=sys.stderr)

    trainer.extend(evaluate, trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss']), trigger=log_interval)
    trainer.extend(extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    main()
