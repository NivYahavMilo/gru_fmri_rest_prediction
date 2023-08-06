'''
model: LSTM
task: predict clip (15 way classifier)
data: all runs used together
input to model: clip time series/seq
output: label time series
'''
import time

import numpy as np
import pandas as pd

'''
ml
'''
import torch
import torch.nn as nn
from models import GRUClassifier

'''
Helpers
'''
from utils import _info
from cc_utils import _lstm_test_acc
from dataloader import _get_clip_seq as _get_seq

K_SEED = 330


def train(df: pd.DataFrame, args):
    '''
    test subject results
    view only for best cross-val parameters
    '''
    _info('test mode')

    # get X-y from df
    subject_list = df['Subject'].unique()
    train_list = subject_list[:args.train_size]
    test_list = subject_list[args.train_size:]

    features = [ii for ii in df.columns if 'feat' in ii]
    k_feat = len(features)
    print('number of classes = %d' % (args.k_class))

    # length of each clip
    clip_time = np.zeros(args.k_class)
    for ii in range(args.k_class):
        class_df = df[df['y'] == ii]
        clip_time[ii] = np.max(np.unique(class_df['timepoint'])) + 1
    clip_time = clip_time.astype(int)  # df saves float
    _info('seq lengths = %s' % clip_time)

    # results dict init
    results = {}

    # mean accuracy across time
    results['train'] = np.zeros(len(test_list))
    results['val'] = np.zeros(len(test_list))

    # per class temporal accuracy
    results['t_train'] = {}
    results['t_test'] = {}
    for ii in range(args.k_class):
        results['t_train'][ii] = np.zeros(
            (len(test_list), clip_time[ii]))
        results['t_test'][ii] = np.zeros(
            (len(test_list), clip_time[ii]))
    '''
    init model
    '''
    model = GRUClassifier(k_feat, args.k_hidden, args.k_layers, args.k_class, bi_lstm=True)
    model.to(args.device)
    print(model)

    lossfn = nn.CrossEntropyLoss(ignore_index=-100)
    # if input is cuda, loss function is auto cuda
    opt = torch.optim.Adam(model.parameters())

    # get train, val sequences
    X_train, train_len, y_train = _get_seq(df, train_list, args)
    X_test, test_len, y_test = _get_seq(df, test_list, args)

    max_length = torch.max(train_len)
    '''
    train classifier
    '''
    permutation = torch.randperm(X_train.size()[0])
    losses = np.zeros(args.num_epochs)
    #
    then = time.time()

    for epoch in range(args.num_epochs):
        for i in range(0, X_train.size()[0], args.batch_size):
            indices = permutation[i:i + args.batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            batch_x_len = train_len[indices]

            y_pred = model(batch_x, batch_x_len, max_length)
            loss = lossfn(y_pred.view(-1, args.k_class), batch_y.view(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()

        losses[epoch] = loss

    _info(losses)
    #
    print('--- train time =  %0.4f seconds ---' % (time.time() - then))

    '''
    evaluate
    '''
    a, a_t, c_mtx = _lstm_test_acc(model, X_test, y_test, test_len, max_length, clip_time, len(test_list))
    results['test'] = a
    print('sacc = %0.3f' % np.mean(a))
    for ii in range(args.k_class):
        results['t_test'][ii] = a_t[ii]
    results['test_conf_mtx'] = c_mtx

    return results
