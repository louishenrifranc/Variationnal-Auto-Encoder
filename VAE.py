import lasagne
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer, get_all_params, get_output
from GaussianLatentLayer import GaussianPropLayer
from lasagne import nonlinearities
from lasagne.init import GlorotUniform
import theano.tensor as T
import numpy as np
import theano
import urllib
import os
import time
import sys
import gzip


def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)

        data = data.reshape(-1, 784)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]


class VAE(object):
    def __init__(self,
                 batch_size=32,
                 nb_features=784,
                 hidden_size=300,
                 dropout_hid=0.2,
                 latent_space=128):
        self.BATCH_SIZE = batch_size
        self.INPUT_SIZE = nb_features
        self.HIDDEN_SIZE = hidden_size
        self.LATENT_SPACE_SIZE = latent_space
        self.NUM_EPOCH = 100
        self.e = 1e-6
        model = {}

        # Input variable
        X_input = T.fmatrix('input')

        # Build the model of the Variationnal Auto Encoder
        # Topology : (1 hidden layer for the encoder/decoder) + 1 dense layer for getting the mean/stdd vectors
        # Reparametrization trick implemented as a Lasagne layer
        # ---------------
        # 1. Input Layer
        model['l_in'] = InputLayer(input_var=X_input,
                                   shape=(self.BATCH_SIZE, self.INPUT_SIZE))

        # 2. Hidden layer of the encoder
        model['l_hid_enc'] = DenseLayer(model['l_in'],
                                        self.HIDDEN_SIZE,
                                        nonlinearity=nonlinearities.tanh)

        # 2.bis Dropout layer on the hidden neurons of the encoder
        model['l_enc_drop'] = DropoutLayer(model['l_hid_enc'], p=dropout_hid)

        # 3.a Dense layer to output the mean of the gaussian distribution
        model['l_mu'] = DenseLayer(model['l_enc_drop'],
                                   self.LATENT_SPACE_SIZE,
                                   nonlinearity=nonlinearities.linear)

        # 3.a Dense layer to output the variance of the gaussian distribution
        model['l_sttdev'] = DenseLayer(model['l_enc_drop'],
                                       self.LATENT_SPACE_SIZE,
                                       nonlinearity=nonlinearities.linear)
        # 4. Sample from the latent distribution
        model['l_z'] = GaussianPropLayer(model['l_mu'], model['l_sttdev'])

        # 5. Hidden layer of the decoder
        model['l_hid_dec'] = DenseLayer(model['l_z'],
                                        self.HIDDEN_SIZE,
                                        nonlinearity=nonlinearities.tanh)
        # 5.bis Dropout layer on the hidden neurons of the decoder
        model['l_dec_drop'] = DropoutLayer(model['l_hid_dec'], p=dropout_hid)

        # 6. Ouput layer to reconstruct the image
        model['l_out'] = DenseLayer(model['l_dec_drop'],
                                    self.INPUT_SIZE,
                                    nonlinearity=nonlinearities.sigmoid)

        # output image (shape = batch_size, 784) during training
        Y_output = get_output(model['l_out'], deterministic=False)

        # output image (shape = batch_size, 784) during testing
        Y_output_test = get_output(model['l_out'], deterministic=True)

        # Get the mean and stddev vectors
        z_sttdev = get_output(model['l_sttdev'])
        z_mean = get_output(model['l_mu'])

        # KL divergence for gaussian distribution
        # https://arxiv.org/pdf/1312.6114.pdf
        self.latent_loss = 0.5 * T.sum(T.square(z_mean) + T.square(z_sttdev) - T.log(T.square(z_sttdev)) - 1, axis=1)
        # REconsturction
        self.reconstruction_loss = -T.sum(
            X_input * T.log(self.e + Y_output) + (1 - X_input) * T.log(1 - Y_output + self.e), axis=1)
        self.loss = (self.latent_loss + self.reconstruction_loss)

        self.reconstruction_loss_test = -T.sum(
            X_input * T.log(self.e + Y_output_test) + (1 - X_input) * T.log(1 - Y_output_test + self.e), axis=1)
        self.loss_test = (self.latent_loss + self.reconstruction_loss_test)

        self.cost = T.mean(self.loss)
        self.cost_test = -T.mean(self.loss_test)

        all_params = get_all_params(model['l_out'])

        optimizer = lasagne.updates.adadelta(self.cost, all_params)

        self.train_fn = theano.function([X_input], self.cost, updates=optimizer)
        self.test_fn = theano.function([X_input], self.cost_test)
        self.z_mean_fn = theano.function([X_input],[z_sttdev, z_mean])

    def train(self):
        X_train, y_train, X_val, yval, X_test, y_test = load_dataset()
        for epoch in range(self.NUM_EPOCH):
            train_err = 0
            n_train_batches = 0
            start_time = time.time()
            for X_batch_train in iterate_minibatches(X_train, batchsize=self.BATCH_SIZE, shuffle=True):
                train_err += self.train_fn(X_batch_train)
                print(train_err)
                n_train_batches += 1
               # print(self.z_mean_fn(X_batch_train)[0], self.z_mean_fn(X_batch_train)[1])
            val_err = 0
            n_val_batches = 0
            for X_batch_val in iterate_minibatches(X_val, self.BATCH_SIZE, shuffle=False):
                err = self.test_fn(X_batch_val)
                val_err += err
                n_val_batches += 1
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, self.NUM_EPOCH, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / n_train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / n_val_batches))

        test_err = 0
        test_batches = 0
        for X_batch_test in iterate_minibatches(X_test, self.BATCH_SIZE, shuffle=False):
            err = self.test_fn(X_batch_test)
            test_err += err
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))


if __name__ == '__main__':
    vae = VAE()
    vae.train()
