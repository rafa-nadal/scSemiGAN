import os
import time
import dateutil.tz
import datetime
import argparse
import importlib
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import umap

import util


class scSemiGAN(object):
    def __init__(self, g_net, d_net, enc_net, x_sampler, z_sampler, data, model,
                 num_classes, dim_gen, batch_size, beta_cycle_gen, beta_cycle_label):
        self.model = model
        self.data = data
        self.g_net = g_net
        self.d_net = d_net
        self.enc_net = enc_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.num_classes = num_classes
        self.dim_gen = dim_gen
        self.batch_size = batch_size
        self.beta_cycle_gen = beta_cycle_gen
        self.beta_cycle_label = beta_cycle_label

        self.x_dim = self.d_net.x_dim
        self.z_dim = self.g_net.z_dim

        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.x_sup = tf.placeholder(tf.float32, [None, self.x_dim], name='x_sup')
        self.x_sup_label = tf.placeholder(tf.int32, [None, self.num_classes], name='x_sup_label')
        self.x_sup_label_value = tf.placeholder(tf.int32, [200], name='x_sup_label_value')

        self.z_gen = self.z[:, 0:self.dim_gen]
        self.z_hot = self.z[:, self.dim_gen:]

        self.x_ = self.g_net(self.z, reuse=False)
        self.z_enc_gen, self.z_enc_label, self.z_enc_logits = self.enc_net(self.x_, reuse=False)
        self.z_infer_gen, self.z_infer_label, self.z_infer_logits = self.enc_net(self.x)

        self.d = self.d_net(self.x, reuse=False)
        self.d_ = self.d_net(self.x_)

        self.z_infer_gen_sup, self.z_infer_label_sup, self.z_infer_logits_sup = self.enc_net(self.x_sup)

        self.g_loss = tf.reduce_mean(self.d_) + \
                      self.beta_cycle_gen * tf.reduce_mean(tf.square(self.z_gen - self.z_enc_gen)) + \
                      self.beta_cycle_label * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.z_enc_logits, labels=self.z_hot))

        self.g_loss_med_reverse = self.beta_cycle_label * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.z_infer_logits_sup, labels=self.x_sup_label))
 
        self.g_loss += self.g_loss_med_reverse

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.x_
        d_hat = self.d_net(x_hat)

        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0))        

        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_) + ddx

        self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(self.d_loss, var_list=self.d_net.vars)
        self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(self.g_loss, var_list=[self.g_net.vars, self.enc_net.vars])

        self.saver = tf.train.Saver()

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

    def train(self, num_batches=10000):

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

        self.sess.run(tf.global_variables_initializer())

        start_time = time.time()
        print('Training {} on {}, z = {} dimension, beta_n = {}, beta_c = {}'.format(self.model, self.data, self.z_dim, self.beta_cycle_gen, self.beta_cycle_label))

        sup_x, sup_label, matrix_left, label_left = self.x_sampler.supervised_test(ratio=0.1)

        sup_label = [int(i) for i in sup_label]
        sup_label_onehot = np.eye(self.num_classes)[sup_label]

        batch_size = self.batch_size
        
        for t in range(0, num_batches):

            d_iters = 5

            for _ in range(0, d_iters):
                bx = self.x_sampler.train(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim, self.num_classes)
                self.sess.run(self.d_adam, feed_dict={self.x: bx, self.z: bz})

            bz = self.z_sampler(batch_size, self.z_dim, self.num_classes)
            self.sess.run(self.g_adam, feed_dict={self.z: bz, self.x_sup: sup_x, self.x_sup_label: sup_label_onehot, self.x_sup_label_value: sup_label})

            if (t + 1) % 100 == 0:
                bx = self.x_sampler.train(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim, self.num_classes)

                nmi, ari, acc, f1_score_micro, f1_score_macro, f1_score_weighed = self.recon_enc(timestamp, matrix_left, label_left)
                
            if (t + 1) % 1000 == 0:
                nmi, ari, acc, f1_score_micro, f1_score_macro, f1_score_weighed = self.recon_enc_vis(timestamp, matrix_left, label_left)


    def recon_enc(self, timestamp, matrix_left, label_left):

        data_recon = matrix_left
        label_recon = label_left

        num_pts_to_plot = data_recon.shape[0]
        recon_batch_size = self.batch_size
        latent = np.zeros(shape=(num_pts_to_plot, self.z_dim))

        for b in range(int(np.ceil(num_pts_to_plot * 1.0 / recon_batch_size))):

            if (b + 1) * recon_batch_size > num_pts_to_plot:
                pt_indx = np.arange(b * recon_batch_size, num_pts_to_plot)
            else:
                pt_indx = np.arange(b * recon_batch_size, (b + 1) * recon_batch_size)
            xtrue = data_recon[pt_indx, :]

            zhats_gen, zhats_label = self.sess.run([self.z_infer_gen, self.z_infer_label], feed_dict={self.x: xtrue})

            latent[pt_indx, :] = np.concatenate((zhats_gen, zhats_label), axis=1)

        nmi, ari, acc, f1_score_micro, f1_score_macro, f1_score_weighed = self._eval_cluster(latent[:, self.dim_gen:], label_recon, timestamp)

        return nmi, ari, acc, f1_score_micro, f1_score_macro, f1_score_weighed


    def recon_enc_vis(self, timestamp, matrix_left, label_left):

        data_recon = matrix_left
        label_recon = label_left

        num_pts_to_plot = data_recon.shape[0]
        recon_batch_size = self.batch_size
        latent = np.zeros(shape=(num_pts_to_plot, self.z_dim))

        for b in range(int(np.ceil(num_pts_to_plot * 1.0 / recon_batch_size))):

            if (b + 1) * recon_batch_size > num_pts_to_plot:
                pt_indx = np.arange(b * recon_batch_size, num_pts_to_plot)
            else:
                pt_indx = np.arange(b * recon_batch_size, (b + 1) * recon_batch_size)
            xtrue = data_recon[pt_indx, :]

            zhats_gen, zhats_label = self.sess.run([self.z_infer_gen, self.z_infer_label], feed_dict={self.x: xtrue})

            latent[pt_indx, :] = np.concatenate((zhats_gen, zhats_label), axis=1)

        nmi, ari, acc, f1_score_micro, f1_score_macro, f1_score_weighed = self._eval_cluster(latent[:, self.dim_gen:], label_recon, timestamp)
        
        self.visualize(latent, label_recon)

        return nmi, ari, acc, f1_score_micro, f1_score_macro, f1_score_weighed


    def _eval_cluster(self, latent_rep, labels_true, timestamp):
        labels_pred = np.argmax(latent_rep, axis=1)

        accuracy = accuracy_score(labels_true, labels_pred)
        f1_score_micro = f1_score(labels_true, labels_pred, average="micro")
        f1_score_macro = f1_score(labels_true, labels_pred, average="macro")
        f1_score_weighed = f1_score(labels_true, labels_pred, average="weighted")

        ari = adjusted_rand_score(labels_true, labels_pred)
        nmi = normalized_mutual_info_score(labels_true, labels_pred)


        print('Data = {}, Model = {}, z_dim = {}, beta_label = {}, beta_gen = {} '
              .format(self.data, self.model, self.z_dim, self.beta_cycle_label, self.beta_cycle_gen))
        print(' #Points = {}, K = {}, NMI = {}, ARI = {},  Accuracy = {},  F1_score_micro = {},  F1_score_macro = {}, F1_score_weighed = {}'
              .format(latent_rep.shape[0], self.num_classes, nmi, ari, accuracy, f1_score_micro, f1_score_macro, f1_score_weighed))

        return nmi, ari, accuracy, f1_score_micro, f1_score_macro, f1_score_weighed


    def visualize(self, feature, labels):
        # reducer = umap.UMAP(random_state=501)
        # reduced_data = reducer.fit_transform(feature)

        tsne = TSNE(n_components=2, init='pca', random_state=501)
        reduced_data = tsne.fit_transform(feature)

        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=10, c=labels, cmap="tab20")
        plt.savefig("one.png")
        plt.show()   
        

    def save(self, timestamp):

        checkpoint_dir = 'checkpoint_dir/{}/{}_{}_{}_z{}_cyc{}_gen{}'.format(self.data, timestamp, self.model, self.sampler, self.z_dim, self.beta_cycle_label, self.beta_cycle_gen)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'))


    def load(self, pre_trained=False, timestamp=''):

        if pre_trained:
            print('Loading Pre-trained Model...')
            checkpoint_dir = 'pre_trained_models/{}/{}_{}_z{}_cyc{}_gen{}'.format(self.data, self.model, self.z_dim, self.beta_cycle_label, self.beta_cycle_gen)
        else:
            if timestamp == '':
                print('Best Timestamp not provided. Abort !')
                checkpoint_dir = ''
            else:
                checkpoint_dir = 'checkpoint_dir/{}/{}_{}_{}_z{}_cyc{}_gen{}'.format(self.data, timestamp, self.model, self.z_dim, self.beta_cycle_label, self.beta_cycle_gen)

        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'))
        print('Restored model weights.')


if __name__ == '__main__':
    tf.set_random_seed(0)
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='sim')
    parser.add_argument('--model', type=str, default='clus_wgan')
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--dz', type=int, default=10)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--beta_n', type=float, default=1)
    parser.add_argument('--beta_c', type=float, default=1)
    parser.add_argument('--timestamp', type=str, default='')
    parser.add_argument('--train', type=str, default='True')
    args = parser.parse_args()
    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data + '.' + args.model)

    num_classes = args.K
    dim_gen = args.dz
    batch_size = args.bs
    beta_cycle_gen = args.beta_n
    beta_cycle_label = args.beta_c
    timestamp = args.timestamp

    z_dim = dim_gen + num_classes
    d_net = model.Discriminator()
    g_net = model.Generator(z_dim=z_dim)
    enc_net = model.Encoder(z_dim=z_dim, dim_gen=dim_gen)
    xs = data.DataSampler()
    zs = util.sample_Z

    scsemigan = scSemiGAN(g_net, d_net, enc_net, xs, zs, args.data, args.model,
                     num_classes, dim_gen, batch_size, beta_cycle_gen, beta_cycle_label)

    if args.train == 'True':
        scsemigan.train()
    else:

        print('Attempting to Restore Model ...')
        if timestamp == '':
            scsemigan.load(pre_trained=True)
            timestamp = 'pre-trained'
        else:
            scsemigan.load(pre_trained=False, timestamp=timestamp)

        scsemigan.recon_enc(timestamp, val=False)
