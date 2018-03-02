import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint
import seaborn as sns
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from collections  import OrderedDict

# unsw_nb15_1 = "../datasets/UNSW-NB15/UNSW-NB15_1.csv"
# unsw_nb15_2 = "../datasets/UNSW-NB15/UNSW-NB15_2.csv"
# unsw_nb15_3 = "../datasets/UNSW-NB15/UNSW-NB15_3.csv"
# unsw_nb15_4 = "../datasets/UNSW-NB15/UNSW-NB15_4.csv"

unsw_nb15_1 = "../datasets/UNSW-NB15/df1-time-adjusted.csv"
unsw_nb15_2 = "../datasets/UNSW-NB15/df2-time-adjusted.csv"
unsw_nb15_3 = "../datasets/UNSW-NB15/df3-time-adjusted.csv"
unsw_nb15_4 = "../datasets/UNSW-NB15/df4-time-adjusted.csv"

all_data = "../datasets/UNSW-NB15/all-time-adjusted.csv"

random_state = 55

seed = np.random.seed(random_state)

class ModalData:

    def __init__(self):
        self.rs = random_state
        self.load()

    def load(self):
        cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
                'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
                'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt',
                'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
                'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
                'ct_src_dport_ltm',
                'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label']
        self.df1 = pd.read_csv(unsw_nb15_1, header=None, names=cols)
        self.df2 = pd.read_csv(unsw_nb15_2, header=None, names=cols)
        self.df3 = pd.read_csv(unsw_nb15_3, header=None, names=cols)
        self.df4 = pd.read_csv(unsw_nb15_4, header=None, names=cols)
        # self.dataset = pd.read_csv(all_data)
        # self.dataset = pd.concat([self.df1, self.df2, self.df3, self.df4], axis=0, ignore_index=True)
        # self.dataset = self.df1


    def print_head(self):
        with pd.option_context('display.max_columns', 50, 'display.precision', 1):
            print(self.dataset.head(5))


    def write_to_csv(self):
        self.dataset.to_csv("../datasets/UNSW-NB15/all-time-adjusted.csv", header=False, index=False)

    def df1_remove_attacks(self):
        # self.df1 = self.df1[self.df1['label'] == 0]
        self.df1_normal = self.df1[186788:]
        self.df1_attack = self.df1[:186787]

    def df2_trim_attacks(self):
        self.df2_normal = self.df2[:387246]
        self.df2_attack = self.df2[387247:]
        # print(attack)


    def ts_dur(self):
        self.normal = pd.concat([self.df1_normal, self.df2_normal])
        self.attack = pd.concat([self.df1_attack, self.df2_attack, self.df3, self.df4])
        # self.dataset = self.df2
        agg_dict = {
            'ltime': 'count',
            'dur': 'mean',
            'sbytes': 'mean',
            'dbytes': 'mean',
            'sttl' : 'mean',
            'dttl' : 'mean',
            'sloss' : 'mean',
            'dloss' : 'mean',
            'sload' : 'mean',
            'dload' : 'mean',
            'spkts' : 'mean',
            'dpkts' : 'mean',
            'swin' : 'mean',
            'dwin' : 'mean',
            'trans_depth' : 'mean',
            'res_bdy_len' : 'mean',
            'sjit' : 'mean',
            'djit' : 'mean',
            'sintpkt' : 'mean',
            'dintpkt' : 'mean',
            'tcprtt' : 'mean',
            'synack' : 'mean',
            'ackdat' : 'mean',
            'is_sm_ips_ports' : 'count',
            'is_ftp_login' : 'count',
            'ct_state_ttl' : 'mean',
            'ct_srv_src' : 'mean',
            'ct_srv_dst' : 'mean',
            'ct_dst_ltm' : 'mean',
            'ct_src_ltm' : 'mean',
            'ct_src_dport_ltm' : 'mean',
            'ct_dst_sport_ltm' : 'mean',
            'ct_dst_src_ltm' : 'mean'
        }
        rename_dict = {
            'ltime': 'count'
        }
        self.normal = self.normal[list(agg_dict)].groupby('ltime').agg(OrderedDict(agg_dict.items())).rename(columns=rename_dict)
        self.attack = self.attack[list(agg_dict)].groupby('ltime').agg(OrderedDict(agg_dict.items())).rename(columns=rename_dict)
        # ts['normal'] = ts['count'] - ts['attack']
        # with pd.option_context('display.max_columns', 50, 'display.precision', 1):
        #     print(self.normal.head)

    def write_ts_to_csv(self):
        self.normal.to_csv("../datasets/derived/normal.csv", header=True, index=False)
        self.attack.to_csv("../datasets/derived/attack.csv", header=True, index=False)



data = ModalData()
# data.vis_labels()
# data.print_head()
# data.verify_time_sequence()
# data.print_time_gaps()
# data.write_to_csv()
data.df1_remove_attacks()
data.df2_trim_attacks()
data.ts_dur()
data.write_ts_to_csv()
