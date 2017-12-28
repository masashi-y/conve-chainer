#!/usr/bin/env python

import argparse
import dill
import sys
import numpy as np
import chainer
from conve import Vocab, ComplEx, ConvE


models = {'complex': ComplEx, 'conve': ConvE}

def run(model, dataset, ent_vocab, rel_vocab, lemma2id, topn=100):
    pairs = [(sid, rid, oid) for s, o in dataset
                for sid in lemma2id[s]
                    for oid in lemma2id[o]
                        for rid in range(len(rel_vocab))]

    if len(pairs) == 0:
        print("NO ENTRIES FOR:", dataset, file=sys.stderr)
        return
    samples = np.asarray(pairs)
    e1, rel, e2 = map(np.asarray, zip(*pairs))
    with chainer.using_config('train', False):
        res = model.forward(e1, rel, e2)
        print(res.shape)
    res = res.data
    rank = np.argsort(-res)

    print(dataset, file=sys.stderr)
    for i, (score, (sid, rid, oid)) in enumerate(
                                zip(res[rank], samples[rank]), 1):
        print("{: >3}: {}\t{}\t{}\t{:.3f}".format(
            i,
            ent_vocab.id2word[sid],
            ent_vocab.id2word[oid],
            rel_vocab.id2word[rid],
            score))
        if i >= topn:
            break
    print()


def clean_ent(ent):
    ent = ent[2:]
    items = ent.split("_")
    syn_id = items[-1]
    cat = items[-2]
    ent = "_".join(items[:-2])
    return ent


def compress(args):
    from collections import defaultdict
    from nltk.corpus import wordnet as wn
    ent_vocab = Vocab.load(args.ent)
    rel_vocab = Vocab.load(args.rel)
    model = models[args.model](len(ent_vocab), len(rel_vocab))
    chainer.serializers.load_npz(args.modelfile, model)
    lemma2id = defaultdict(list)
    id2lemma = []
    for i, w in enumerate(ent_vocab.id2word):
        syn = wn.synset(w)
        for w in syn.lemma_names():
            lemma2id[w].append(i)
            id2lemma.append(w)
    with open(args.out, "wb") as f:
        dill.dump([ent_vocab, rel_vocab, model, lemma2id, id2lemma], f)


def prompt():
    def parse(pair, bracket=True):
        if bracket:
            assert pair[0] == "(" and pair[-1] == ")"
            pair = pair[1:-1]
        s, o = pair.split(",")
        return s.strip(), o.strip()
    res = []
    input_line = input(">> ").strip()
    inp = input_line
    if inp.startswith("S "):
        inp = inp[1:].strip()
        return ("S", [parse(inp, bracket=False)])
    try:
        while len(inp) > 0:
            idx = inp.find(")")
            if idx == -1:
                res.append(parse(inp, bracket=False))
                break
            else:
                res.append(parse(inp[:idx+1]))
                inp = inp[idx+1:].strip(", ")
    except:
        print(("PARSER ERROR ON INPUT : {}\n"
              "input line should be either of:\n"
              "  * HEAD, TAIL\n"
              "  * (HEAD, TAIL) [(HEAD, TAIL) ...]\n"
              "  * (HEAD, TAIL), [(HEAD, TAIL) ...]\n"
              ).format(input_line), file=sys.stderr)
        return None
    return "C", res


def main(args):
    with open(args.model, "rb") as f:
        ent_vocab, rel_vocab, model, lemma2id, id2lemma = dill.load(f)

    while True:
        try:
            command, dataset = prompt()
            if command == "C" and \
                dataset is not None and len(dataset) > 0:
                run(model, dataset, ent_vocab,
                        rel_vocab, lemma2id, topn=args.topn)
            if command == "S":
                raise "not implemented yet"
        except KeyboardInterrupt:
            print()


if __name__ == '__main__':
    p = argparse.ArgumentParser('Link prediction models')
    p.set_defaults(func=lambda _: p.print_help())
    ps = p.add_subparsers()

    p1 = ps.add_parser("create")
    p1.add_argument('--ent', type=str, help='entity list')
    p1.add_argument('--rel', type=str, help='relation list')
    p1.add_argument('--model', type=str, choices=models.keys())
    p1.add_argument('--modelfile', type=str, help='trained model path')
    p1.add_argument('--out', type=str, help='output file', default="model.config")
    p1.set_defaults(func=compress)

    p2 = ps.add_parser("run")
    p2.add_argument('--model', type=str, help='trained model config file')
    p2.add_argument('--topn', type=int, help='show top N results', default=7)
    p2.set_defaults(func=main)

    args = p.parse_args()
    args.func(args)
