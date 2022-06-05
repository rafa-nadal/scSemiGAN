import scipy.sparse
import scipy.io
import numpy as np
from sklearn.preprocessing import scale, minmax_scale
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE

class DataSampler(object):
    def __init__(self):
        self.total_size = 2000
        self.train_size = 2000
        self.test_size = 0
        self.X_train, self.X_test = self._load_gene_mtx("data/simulate/gene_cell/sim_65.csv", "data/simulate/label/label_65.csv")
        self.y_train, self.y_test = self._load_labels("data/simulate/gene_cell/sim_65.csv", "data/simulate/label/label_65.csv")


    def _load_gene_mtx(self, mat, label):
        data, _ = self.read_sim(mat, label)
        data = minmax_scale(data, feature_range=(0, 1), axis=1, copy=False)

        np.random.seed(0)
        index = np.random.permutation(np.arange(self.total_size))
        data_train = data[index[0:self.train_size], :]
        data_test = data[index[self.train_size:], :]

        return data_train, data_test


    def _load_labels(self, mat, label):
        _, label = self.read_sim(mat, label)

        np.random.seed(0)
        index = np.random.permutation(np.arange(self.total_size))
        labels_train = label[index[0:self.train_size]]
        labels_test = label[index[self.train_size:]]

        return labels_train, labels_test

      
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


    def read_sim(self, mat, label):
        gene_cell = pd.read_csv(mat)
        gene_cell = gene_cell[gene_cell.columns[1:]].values
        cell_gene = gene_cell.transpose()

        label_cell = pd.read_csv(label, header=None)
        label_cell = label_cell[label_cell.columns[1:]].values
        for i in range(label_cell.shape[0]):
            for j in range(label_cell.shape[1]):
                label_cell[i][j] = label_cell[i][j][5]
        label_cell = label_cell.astype("int32")
        label_cell = label_cell - 1

        return cell_gene, label_cell
    
    
if __name__ == '__main__':
    xs = DataSampler()
    

   




