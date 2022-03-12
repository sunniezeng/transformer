# encoding: utf-8

import tensorflow as tf
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from transformer import parameters as params
import jieba as jb


def create_vocabs(file, vocab_file):
    word_counts = {
        "<pad>": 1000000000,
        "<unknow>": 999999999,
        "<start>": 999999998,
        "<end>": 999999997
    }
    with tf.gfile.Open(file, mode='r') as reader:
        for line in reader:
            words = list(jb.cut(line.strip('\n')))
            for word in words:
                if word != '\t':
                    if word in word_counts:
                        word_counts[word] += 1
                    else:
                        word_counts[word] = 1
        word_counts = sorted(word_counts.items(), key=lambda d: d[1], reverse=True)
    with tf.gfile.Open(vocab_file, mode="w") as writer:
        for word_ct in word_counts:
            writer.write("{}\t{}\n".format(word_ct[0], word_ct[1]))


def load_vocabs(vocab_file, min_count):
    word2idx = {}
    idx2word = {}
    with tf.gfile.Open(vocab_file, mode="r") as reader:
        idx = 0
        for line in reader:
            word_ct = line.strip("\n").split("\t")
            if int(word_ct[1]) >= min_count:
                word2idx[word_ct[0]] = idx
                idx2word[idx] = word_ct[0]
                idx += 1
    return word2idx, idx2word


def create_inputs(file, word2idx, max_len, num_records):
    with tf.gfile.Open(file, mode="r") as reader:
        data = []
        ct = 0
        for line in reader:
            if ct >= num_records: break
            record = np.zeros(shape=[max_len], dtype=np.int32)
            words = list(jb.cut(line.strip('\n')))
            words.insert(0, "<start>")
            words.append("<end>")
            words = words[:max_len]
            for i in range(len(words)):
                record[i] = word2idx.get(words[i], 1)
            data.append(record)
            ct += 1
    return np.array(data)

def load_data():
    try:
        source_word2idx, source_idx2word = load_vocabs(params.source_vocab_file, params.min_count)
        target_word2idx, target_idx2word = load_vocabs(params.target_vocab_file, params.min_count)
    except:
        create_vocabs(params.source_file, params.source_vocab_file)
        create_vocabs(params.target_file, params.target_vocab_file)
        source_word2idx, source_idx2word = load_vocabs(params.source_vocab_file, params.min_count)
        target_word2idx, target_idx2word = load_vocabs(params.target_vocab_file, params.min_count)
    source_inputs = create_inputs(params.source_file, source_word2idx, params.max_len, 1)
    target_inputs = create_inputs(params.target_file, target_word2idx, params.max_len, 1)
    source_vocab_size = len(source_word2idx.keys())
    target_vocab_size = len(target_word2idx.keys())
    return source_inputs, target_inputs, source_vocab_size, target_vocab_size, source_idx2word, target_idx2word


def get_batches(source_inputs, target_inputs, shuffle=True):
    idx = np.arange(source_inputs.shape[0])
    if shuffle:
        np.random.shuffle(idx)
    batches = [
        idx[range(i * params.batch_size, (i + 1) * params.batch_size)] for i in range(int(len(idx) / params.batch_size))
    ]
    for i in batches:
        yield (source_inputs[i], target_inputs[i])

