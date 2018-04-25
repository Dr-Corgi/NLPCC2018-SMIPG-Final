# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import bilstm_model.data_utils as data_utils
import bilstm_model.multi_task_model as multi_task_model
import bilstm_model.multi_task_model_context as multi_task_model_context

import subprocess
import stat

# tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
# tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9,
#                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 128, "word embedding size")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 10000, "max vocab Size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 10000, "max tag vocab Size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit)")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("max_training_steps", 30000,
                            "Max training steps.")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "Max size of test set.")
tf.app.flags.DEFINE_boolean("use_attention", True,
                            "Use attention based RNN")
tf.app.flags.DEFINE_integer("max_sequence_length", 50,
                            "Max sequence length.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                          "dropout keep cell input and output prob.")
tf.app.flags.DEFINE_boolean("bidirectional_rnn", True,
                            "Use birectional RNN")
tf.app.flags.DEFINE_string("task", "joint", "Options: joint; intent; tagging")
FLAGS = tf.app.flags.FLAGS

if FLAGS.max_sequence_length == 0:
    print('Please indicate max sequence length. Exit')
    exit()

if FLAGS.task is None:
    print('Please indicate task to run.' +
          'Available options: intent; tagging; joint')
    exit()

task = dict({'intent': 0, 'tagging': 0, 'joint': 0})
if FLAGS.task == 'intent':
    task['intent'] = 1
elif FLAGS.task == 'tagging':
    task['tagging'] = 1
elif FLAGS.task == 'joint':
    task['intent'] = 1
    task['tagging'] = 1
    task['joint'] = 1

_buckets = [(FLAGS.max_sequence_length, FLAGS.max_sequence_length)]


# _buckets = [(3, 10), (10, 25)]

# metrics function using conlleval.pl
def conlleval(p, g, w, filename):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out[:-1])  # remove the ending \n on last line
    f.close()

    return get_perf(filename)


def save_class_file(p, g, filename):
    out = ''
    for sl, sp in zip(g, p):
        out += sl + ' ' + sp + '\n'
    f = open(filename, 'w')
    f.writelines(out[:-1])  # remove the ending \n on last line
    f.close()

def get_perf(filename):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = os.path.dirname(os.path.realpath(__file__)) + '/conlleval.pl'
    os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(["perl",
                             _conlleval],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(b''.join(open(filename, 'rb').readlines()))
    for line in stdout.decode().split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    return {'p': precision, 'r': recall, 'f1': f1score}


def read_data(source_path, target_path, label_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
      source_path: path to the files with token-ids for the word sequence.
      target_path: path to the file with token-ids for the tag sequence;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      label_path: path to the file with token-ids for the intent label
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target, label) tuple read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1];source, target, label are lists of token-ids
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            with tf.gfile.GFile(label_path, mode="r") as label_file:
                source = source_file.readline()
                target = target_file.readline()
                label = label_file.readline()
                counter = 0
                while source and target and label and (not max_size \
                                                       or counter < max_size):
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading data line %d" % counter)
                        sys.stdout.flush()
                    source_ids = [int(x) for x in source.split()]
                    target_ids = [int(x) for x in target.split()]
                    label_ids = [int(x) for x in label.split()]
                    #          target_ids.append(data_utils.EOS_ID)
                    for bucket_id, (source_size, target_size) in enumerate(_buckets):
                        if len(source_ids) < source_size and len(target_ids) < target_size:
                            data_set[bucket_id].append([source_ids, target_ids, label_ids])
                            break
                    source = source_file.readline()
                    target = target_file.readline()
                    label = label_file.readline()
    return data_set  # 3 outputs in each unit: source_ids, target_ids, label_ids


def read_data_context(source_path, target_path, label_path, context_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
      source_path: path to the files with token-ids for the word sequence.
      target_path: path to the file with token-ids for the tag sequence;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      label_path: path to the file with token-ids for the intent label
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target, label) tuple read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1];source, target, label are lists of token-ids
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            with tf.gfile.GFile(label_path, mode="r") as label_file:
                with tf.gfile.GFile(context_path, mode="r") as context_file:
                    source = source_file.readline()
                    target = target_file.readline()
                    label = label_file.readline()
                    context = context_file.readline()
                    counter = 0
                    while source and target and label and context and (not max_size \
                                                           or counter < max_size):
                        counter += 1
                        if counter % 100000 == 0:
                            print("  reading data line %d" % counter)
                            sys.stdout.flush()
                        source_ids = [int(x) for x in source.split()]
                        target_ids = [int(x) for x in target.split()]
                        label_ids = [int(x) for x in label.split()]
                        context_ids = [int(x) for x in context.split()]
                        #          target_ids.append(data_utils.EOS_ID)
                        for bucket_id, (source_size, target_size) in enumerate(_buckets):
                            if len(source_ids) < source_size and len(target_ids) < target_size:
                                data_set[bucket_id].append([source_ids, target_ids, label_ids, context_ids])
                                break
                        source = source_file.readline()
                        target = target_file.readline()
                        label = label_file.readline()
                        context = context_file.readline()
    return data_set  # 3 outputs in each unit: source_ids, target_ids, label_ids


def create_model_context(session,
                 source_vocab_size,
                 target_vocab_size,
                 label_vocab_size,
                 context_vocab_size):
    """Create model and initialize or load parameters in session."""
    with tf.variable_scope("model", reuse=None):
        model_train = multi_task_model_context.MultiTaskModelContext(
            source_vocab_size,
            target_vocab_size,
            label_vocab_size,
            context_vocab_size,
            _buckets,
            FLAGS.word_embedding_size,
            FLAGS.size, FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            use_lstm=True,
            forward_only=False,
            use_attention=FLAGS.use_attention,
            bidirectional_rnn=FLAGS.bidirectional_rnn,
            task=task)
    with tf.variable_scope("model", reuse=True):
        model_test = multi_task_model_context.MultiTaskModelContext(
            source_vocab_size,
            target_vocab_size,
            label_vocab_size,
            context_vocab_size,
            _buckets,
            FLAGS.word_embedding_size,
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            use_lstm=True,
            forward_only=True,
            use_attention=FLAGS.use_attention,
            bidirectional_rnn=FLAGS.bidirectional_rnn,
            task=task)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model_train.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model_train, model_test


def create_model(session,
                 source_vocab_size,
                 target_vocab_size,
                 label_vocab_size):
    """Create model and initialize or load parameters in session."""
    with tf.variable_scope("model", reuse=None):
        model_train = multi_task_model.MultiTaskModel(
            source_vocab_size,
            target_vocab_size,
            label_vocab_size,
            _buckets,
            FLAGS.word_embedding_size,
            FLAGS.size, FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            use_lstm=True,
            forward_only=False,
            use_attention=FLAGS.use_attention,
            bidirectional_rnn=FLAGS.bidirectional_rnn,
            task=task)
    with tf.variable_scope("model", reuse=True):
        model_test = multi_task_model.MultiTaskModel(
            source_vocab_size,
            target_vocab_size,
            label_vocab_size,
            _buckets,
            FLAGS.word_embedding_size,
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            use_lstm=True,
            forward_only=True,
            use_attention=FLAGS.use_attention,
            bidirectional_rnn=FLAGS.bidirectional_rnn,
            task=task)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model_train.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model_train, model_test
