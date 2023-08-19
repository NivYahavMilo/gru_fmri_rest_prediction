import os
import pickle

import torch

import config
from clip_lstm import train, train_clip_predict_rest
from dataloader import _clip_class_df_roi, _clip_class_df_net
from train_utils import BrainRegion


def run(train_config: dict, region: BrainRegion):
    res_name = f'{train_config["train_data_mode"]}-training_{train_config["test_data_mode"]}-testing_{train_config["area"]}.pkl'
    res_path = os.path.join(config.RESULTS_DIR, res_name)
    if not os.path.isfile(res_path):

        if train_config.get('train_data_mode') != train_config.get('test_data_mode'):

            if region == BrainRegion.NETWORK:

                clip_df = _clip_class_df_net(train_config, data_path=train_config['train_data'])
                rest_df = _clip_class_df_net(train_config, data_path=train_config['test_data'])

            else:
                clip_df = _clip_class_df_roi(train_config, data_path=train_config['train_data'])
                rest_df = _clip_class_df_roi(train_config, data_path=train_config['test_data'])

            results = {'test_mode': train_clip_predict_rest(clip_df, rest_df, train_config)}

        else:
            if region == BrainRegion.ROI:
                df = _clip_class_df_roi(train_config)
            else:
                df = _clip_class_df_net(train_config)
            results = {'test_mode': train(df, train_config)}

        with open(res_path, 'wb') as f:
            pickle.dump(results, f)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for roi in ['RH_Vis_18', 'RH_DorsAttn_Post_2', 'RH_Default_pCunPCC_1']:
        training_config = {
            'train_data_mode': 'clip',
            'test_data_mode': 'rest',
            'train_data': config.SUBNET_DATA_DF.format(mode='TASK'),
            'test_data': config.SUBNET_DATA_DF.format(mode='REST'),
            'area': roi,
            'k_class': 15,
            'mode': 'video',
            'k_hidden': 128,
            'k_layers': 2,
            'bi_lstm': True,
            'batch_size': 16,
            'num_epochs': 50,
            'train_size': 100,
            'device': torch.device(device),

        }

        run(train_config=training_config, region=BrainRegion.ROI)


    for net in ['DMN', 'Visual', 'DorsalAttention']:
        training_config = {
            'train_data_mode': 'clip',
            'test_data_mode': 'rest',
            'train_data': config.NETWORK_DATA_DF.format(mode='TASK'),
            'test_data': config.NETWORK_DATA_DF.format(mode='REST'),
            'area': net,
            'k_class': 15,
            'mode': 'video',
            'k_hidden': 128,
            'k_layers': 2,
            'bi_lstm': True,
            'batch_size': 16,
            'num_epochs': 50,
            'train_size': 100,
            'device': torch.device(device),
        }

        run(train_config=training_config, region=BrainRegion.NETWORK)



    print('finished!')
