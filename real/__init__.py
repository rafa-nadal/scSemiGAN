import scipy.sparse
import scipy.io
import numpy as np
from sklearn.preprocessing import scale, minmax_scale
import pandas as pd
import scanpy as sc
import real.data_Preprocess
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE

class DataSampler(object):
    def __init__(self):
        self.total_size = 2000
        self.train_size = 2000
        self.test_size = 0
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_gene_mtx_liver()

    def _load_gene_mtx_zeisel(self):
        self.total_size = 3005
        self.train_size = 3005

        data = pd.read_csv("data/Zeisel/data.csv")

        data = data[data.columns[1:]].values

        indicator = np.where(data > 0, 1, 0)
        sum_gene = np.sum(indicator, axis=0).flatten()
        sum_cell = np.sum(indicator, axis=1).flatten()

        data = data[sum_cell > 0, :]
        data = data[:, sum_gene >= 10]

        print(data.shape)

        var_gene = np.var(data, axis=0)
        index = np.argsort(var_gene)[-2000:]
        data = data[:, index]

        print(data.shape)

        data = minmax_scale(data, feature_range=(0, 1), axis=1, copy=False)

        label = pd.read_csv("data/Zeisel/label.csv")
        label = label[label.columns[1:]].values

        label = label.astype("int32")
        label = label - 1
        print(label.shape)
        print(label.dtype)

        label = label[sum_cell > 0]
        print(label)
        print(label.shape)
        print(type(label))

        print(np.unique(label))

        np.random.seed(0)
        index = np.random.permutation(np.arange(self.total_size))

        data_train = data[index[0:self.train_size], :]
        data_test = data[index[self.train_size:], :]

        labels_train = label[index[0:self.train_size]]
        labels_test = label[index[self.train_size:]]

        return data_train, data_test, labels_train, labels_test

    def _load_gene_mtx_liver(self):
        self.total_size = 8444
        self.train_size = 8444

        data = pd.read_csv("data/real/Liver_GSE115469_Tutorial/GSE115469_Data.csv")
        print(data)
        print(data.shape)

        data = data[data.columns[1:]].values
        print(data.shape)
        data = np.transpose(data)
        print(data.shape)

        indicator = np.where(data > 0, 1, 0)
        sum_gene = np.sum(indicator, axis=0).flatten()
        sum_cell = np.sum(indicator, axis=1).flatten()

        data = data[sum_cell > 0, :]
        data = data[:, sum_gene >= 10]

        print(data.shape)

        var_gene = np.var(data, axis=0)
        index = np.argsort(var_gene)[-2000:]
        data = data[:, index]

        print(data.shape)

        data = minmax_scale(data, feature_range=(0, 1), axis=1, copy=False)

        label = pd.read_csv("data/real/Liver_GSE115469_Tutorial/GSE115469_CellClusterType.txt", sep='\t')
        print(label)
        print(label.shape)        

        label = label[sum_cell > 0]
        label = label["CellType"]
        print(label)
        print(label.shape)
        print(label.dtype)

        cell_type, cell_label = np.unique(label, return_inverse=True)
        print(cell_type)
        print(cell_label)

        np.random.seed(0)
        index = np.random.permutation(np.arange(self.total_size))

        data_train = data[index[0:self.train_size], :]
        data_test = data[index[self.train_size:], :]

        labels_train = cell_label[index[0:self.train_size]]
        labels_test = cell_label[index[self.train_size:]]

        return data_train, data_test, labels_train, labels_test

    def _load_gene_mtx_QStrachea(self):
      self.total_size = 1350
      self.train_size = 1350

      data, label, cell_type, cell_class, obs, var = data_Preprocess.read_cell_for_h5(
          "data/real/Quake_Smart-seq2_Trachea/data.h5")

      indicator = np.where(data > 0, 1, 0)
      sum_gene = np.sum(indicator, axis=0).flatten()
      sum_cell = np.sum(indicator, axis=1).flatten()

      data = data[sum_cell > 0, :]
      data = data[:, sum_gene >= 10]

      print(data.shape)

      var_gene = np.var(data, axis=0)
      index = np.argsort(var_gene)[-2000:]
      data = data[:, index]

      print(data.shape)

      data = minmax_scale(data, feature_range=(0, 1), axis=1, copy=False)

      label = label.astype("int32")
      print(label)
      print(label.dtype)

      label = label[sum_cell > 0]

      print(label)
      print(label.shape)
      print(type(label))

      print(np.unique(label))

      np.random.seed(0)
      index = np.random.permutation(np.arange(self.total_size))

      data_train = data[index[0:self.train_size], :]
      data_test = data[index[self.train_size:], :]

      labels_train = label[index[0:self.train_size]]
      labels_test = label[index[self.train_size:]]

      return data_train, data_test, labels_train, labels_test


    def train(self, batch_size, label=False):
        index = np.random.randint(low=0, high=self.train_size, size=batch_size)

        if label:
            return self.X_train[index, :], self.y_train[index].flatten()
        else:
            return self.X_train[index, :]

    
    def supervised_test(self, ratio):

        matrix_final = self.X_train[0:int(self.train_size * ratio), :]
        label_final = self.y_train[0:int(self.train_size * ratio)].flatten()

        matrix_final_left = self.X_train[int(self.train_size * ratio):, :]
        label_final_left = self.y_train[int(self.train_size * ratio):].flatten()

        return matrix_final, label_final, matrix_final_left, label_final_left

    
if __name__ == '__main__':
    xs = DataSampler()
    X_train, X_test, y_train, y_test = xs._load_gene_mtx_zeisel()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)

   




