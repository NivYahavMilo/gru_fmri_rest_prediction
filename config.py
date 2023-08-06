import os

""" Repository paths """
ROOT_PATH = os.path.abspath(os.path.curdir)
DATA_CENTER = os.path.join(ROOT_PATH, 'data_center')


""" External Data Source path """
RAW_DATA = os.path.join(r'E:', 'S1200', '7T_{mode}')
DATA_DRIVE_E = os.path.join(r'E:', 'parcelled_data_niv')
SUBNET_DATA_DF = os.path.join(DATA_DRIVE_E, 'Schaefer2018_SUBNET_{mode}_DF')

""" results directory """
RES_DIR = 'results/clip_lstm'
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)
