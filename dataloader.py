import os

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

K_RUNS = 4


def _get_clip_seq(df, subject_list, args):
    '''
    return:
    X: input seq (batch_size x time x feat_size)
    y: label seq (batch_size x time)
    X_len: len of each seq (batch_size x 1)
    batch_size <-> number of sequences
    time <-> max length after padding
    '''
    features = [ii for ii in df.columns if 'feat' in ii]

    X = []
    y = []
    for subject in subject_list:
        for i_class in range(args.k_class):

            if i_class == 0:  # split test-retest into 4
                seqs = df[(df['Subject'] == subject) &
                          (df['y'] == 0)][features].values
                label_seqs = df[(df['Subject'] == subject) &
                                (df['y'] == 0)]['y'].values

                k_time = int(seqs.shape[0] / K_RUNS)
                for i_run in range(K_RUNS):
                    seq = seqs[i_run * k_time:(i_run + 1) * k_time, :]
                    label_seq = label_seqs[i_run * k_time:(i_run + 1) * k_time]

                    X.append(torch.FloatTensor(seq))
                    y.append(torch.LongTensor(label_seq))
            else:
                seq = df[(df['Subject'] == subject) &
                         (df['y'] == i_class)][features].values
                label_seq = df[(df['Subject'] == subject) &
                               (df['y'] == i_class)]['y'].values

                X.append(torch.FloatTensor(seq))
                y.append(torch.LongTensor(label_seq))

    X_len = torch.LongTensor([len(seq) for seq in X])

    # pad sequences
    X = pad_sequence(X, batch_first=True, padding_value=0)
    y = pad_sequence(y, batch_first=True, padding_value=-100)

    return X.to(args.device), X_len.to(args.device), y.to(args.device)


def _clip_class_df(args):
    df = pd.DataFrame()
    data_path = args.train_data
    for subject in os.listdir(data_path):
        subject_df = pd.read_pickle(os.path.join(data_path, subject, f'{args.roi}.pkl'))
        df = pd.concat([df, subject_df])

    df['Subject'] = df['Subject'].astype(int)

    return df
