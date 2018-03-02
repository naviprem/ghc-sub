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


normal = "../datasets/derived/normal.csv"
attack = "../datasets/derived/attack.csv"

random_state = 55

seed = np.random.seed(random_state)

class ModalData:

    def __init__(self):
        self.rs = random_state
        self.load()

    def load(self):
        self.normal = pd.read_csv(normal)
        self.attack = pd.read_csv(attack)
        self.dataset = pd.concat([self.normal, self.attack], axis=0, ignore_index=True)


    def print_head(self):
        with pd.option_context('display.max_columns', 50, 'display.precision', 1):
            print(self.dataset.head)
        # print(self.dataset.columns.values)



    def log_transform(self):
        self.dataset = self.dataset.apply(lambda x: np.log(x + 1))

    def min_max_scaler(self):
        scaler = MinMaxScaler()
        # print("has infinity: {}".format(np.isfinite(self.dataset)))
        # print("has nans: {}".format(np.isnan(self.dataset)))
        # np.nan_to_num(self.dataset)
        self.dataset = pd.DataFrame(scaler.fit_transform(self.dataset))


    def pca(self, c):
        pca = PCA(n_components=c, random_state=self.rs)
        pca.fit(self.dataset)
        reduced_dataset = pca.transform(self.dataset)
        self.dataset = pd.DataFrame(reduced_dataset)

        # dimensions = ['Dimension {}'.format(i) for i in range(1, len(pca.components_) + 1)]
        # components = pd.DataFrame(np.round(pca.components_, 4), columns=pd.DataFrame(self.dataset).columns.values)
        # components.index = dimensions
        # ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
        # variance_ratios = pd.DataFrame(np.round(ratios, 4), columns=['Explained Variance'])
        # variance_ratios.index = dimensions
        # variance = pd.concat([variance_ratios, components], axis=1)
        # # with pd.option_context('display.max_rows', None, 'display.max_columns', 50, 'display.precision', 4):
        # #     print(variance)
        # variance.to_csv("../results/explained_variance.csv")

        # self.pca_data = pd.DataFrame(np.round(reduced_X_train, 4), columns=variance.index.values)
        # pca_label_data = pd.concat([self.pca_data, self.y_train], axis=1)
        # pca_attack_data = pca_label_data[pca_label_data['label'] == 1]
        # pca_normal_data = pca_label_data[pca_label_data['label'] == 0]
        # plt.scatter(x=pca_attack_data.loc[:, 'Dimension 1'], y=pca_attack_data.loc[:, 'Dimension 2'],
        #        facecolors='r', edgecolors='r', s=70, alpha=0.5)
        # plt.scatter(x=pca_normal_data.loc[:, 'Dimension 1'], y=pca_normal_data.loc[:, 'Dimension 2'],
        #             facecolors='b', edgecolors='b', s=70, alpha=0.5)
        # plt.show()

    def visualize(self):
        cols = self.dataset.columns.values
        l = len(cols)
        fig, ax = plt.subplots(nrows=l, figsize=(250, l*5))

        for i, col in enumerate(cols):
            ax[i].plot(self.dataset[col])

        plt.savefig("../results/all.png")
        plt.show()


data = ModalData()

data.log_transform()
data.min_max_scaler()
data.print_head()
data.pca(15)
data.visualize()

