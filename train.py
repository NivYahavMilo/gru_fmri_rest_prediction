import os
import pickle

import torch

import config
from clip_lstm import train
from dataloader import _clip_class_df
from train_utils import RoiTrainingParams, NetworkTrainingParams, TrainingMode


def run(level: TrainingMode, train_config: dict):
    res_path = os.path.join(RES_DIR, f'{args.train_data_mode}-training_{args.test_data_mode}-testing_{args.roi}.pkl')
    if not os.path.isfile(res_path):
        df = _clip_class_df(args)
        results = {'test_mode': train(df, args)}
        with open(res_path, 'wb') as f:
            pickle.dump(results, f)



if __name__ == '__main__':
    training_config = {
        'train_data_mode':'clip',
        'test_data_mode':'clip',
        'train_data':config.SUBNET_DATA_DF.format(mode='TASK'),
        'test_data':config.SUBNET_DATA_DF.format(mode='TASK'),
        'roi':'RH_Vis_18',
        'k_class':15,
        'mode':'video',
        'k_hidden':128,
        'k_layers':2,
        'batch_size':16,
        'num_epochs':50,
        'train_size':100,
        'device': ''
    }

    run(train_config=training_config)

    print('finished!')
