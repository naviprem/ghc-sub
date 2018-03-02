import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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


unsw_nb15_1 = "../datasets/UNSW-NB15/UNSW-NB15_1.csv"
unsw_nb15_2 = "../datasets/UNSW-NB15/UNSW-NB15_2.csv"
unsw_nb15_3 = "../datasets/UNSW-NB15/UNSW-NB15_3.csv"
unsw_nb15_4 = "../datasets/UNSW-NB15/UNSW-NB15_4.csv"
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
                'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm',
                'ct_src_dport_ltm',
                'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label']
        # self.df1 = pd.read_csv(unsw_nb15_1, header=None, names=cols)
        self.df2 = pd.read_csv(unsw_nb15_2, header=None, names=cols)
        # self.df3 = pd.read_csv(unsw_nb15_3, header=None, names=cols)
        # self.df4 = pd.read_csv(unsw_nb15_4, header=None, names=cols)
        # self.dataset = pd.concat([self.df1, self.df2, self.df3, self.df4], axis=0, ignore_index=True)


    def vis_labels(self):
        labels = pd.DataFrame(self.dataset['label'], columns=['label'])
        print(labels.describe())
        with_markers = pd.concat([labels, labels.index.to_series().map(lambda row: int((row + 1) / 1000)),
                                  labels.index.to_series().map(lambda row: int((row + 1) % 1000))], axis=1)
        with_markers.columns = ['label', 'x', 'y']
        print(with_markers.head)
        attack = with_markers[with_markers['label'] == 1]
        normal = with_markers[with_markers['label'] == 0]
        print("Total number of normal records: ", len(normal))
        print("Total number of attack records: ", len(attack))

        plt.figure(figsize=(250, 100))
        plt.scatter(x=attack.loc[:, 'x'], y=attack.loc[:, 'y'],
                    facecolors='r', edgecolors='r', s=10, alpha=0.5)
        plt.scatter(x=normal.loc[:, 'x'], y=normal.loc[:, 'y'],
                    facecolors='b', edgecolors='b', s=10, alpha=0.5)
        plt.savefig("../results/labels.png")
        plt.show()

    def print_head(self):
        with pd.option_context('display.max_columns', 50, 'display.precision', 1):
            print(self.df4.head(5))



    def verify_time_sequence(self):

        ts = self.df3['ltime']
        unique_time_ct = 1
        prev_t = ts[0]
        for t in ts[1:]:
            if prev_t > t:
                print("{}, {}".format(prev_t, t))
            elif prev_t < t:
                prev_t = t
                unique_time_ct += 1
        print("Unique timestamp count: {}". format(unique_time_ct))

    def ts_dur(self):
        agg_dict = {
            'ltime': 'count',
            'dur' : 'mean',
            'sbytes' : 'mean',
            'dbytes': 'mean'
        }
        ts = self.df2[list(agg_dict)].groupby('ltime').agg(OrderedDict(agg_dict.items()))

        with pd.option_context('display.max_columns', 50, 'display.precision', 1):
            print(ts.head)
        ts.plot()
        plt.show()


    def print_time_gaps(self):

        ts = self.dataset['ltime']

        prev_t = ts[0]
        for t in ts[1:]:
            if t == prev_t or t == (prev_t + 1):
                prev_t = t
            else:
                print("expected: {}, actual: {}, difference: {}".format(prev_t + 1, t, t - (prev_t + 1)))
                prev_t = t


    def adjust_time_gap(self):
        # self.dataset = self.dataset.apply(lambda x: self.adjust_time(x) if x["ltime"] >= 1421972725 else x, axis=1)
        self.dataset["ltime"] = self.dataset["ltime"] - 2246283
        self.dataset["stime"] = self.dataset["stime"] - 2246283


    def adjust_time(self, row):
        row["ltime"] -= 2246283
        row["stime"] -= 2246283
        return row

data = ModalData()
# data.vis_labels()
# data.print_head()
# data.verify_time_sequence()
data.ts_dur()


# expected: 1421972725, actual: 1424219008, difference: 2246283
# first attack record in df2 = 387247 (index)
# last attack record in df1 = 186787 (index)

# last index if normal data = 37381