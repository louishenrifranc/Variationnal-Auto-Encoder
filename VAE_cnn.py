import lasagne
from lasagne.layers import DenseLayer, InputLayer, ReshapeLayer, Upscale2DLayer, Conv2DLayer, MaxPool2DLayer, \
    get_all_params, get_output
from GaussianLatentLayer import GaussianPropLayer
from lasagne import nonlinearities, nonlinearities
from lasagne import init
import theano.tensor as T
import numpy as np
import theano
import urllib
import os
import time
import sys
import gzip
import cPickle


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

        data = data.reshape(-1, 1, 28, 28)
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
    X_train, X_val = X_train[:10000, :, :, :], X_train[10000:, :, :, :]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    X_train = np.reshape(X_train, (-1, 1, 28, 28))
    X_test = np.reshape(X_test, (-1, 1, 28, 28))
    X_val = np.reshape(X_val, (-1, 1, 28, 28))
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('X_val shape:', X_val.shape)
    # We just return all the arrays in order, as expected in main().
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
    """
    Variationnal Auto Encoder MNIST
    """

    def __init__(self,
                 batch_size=32,
                 nb_features=784,
                 hidden_size=16,
                 dropout_hid=0.0,
                 latent_space=16):
        self.BATCH_SIZE = batch_size
        self.INPUT_SIZE = nb_features
        self.HIDDEN_SIZE = hidden_size
        self.LATENT_SPACE_SIZE = latent_space
        self.NUM_EPOCH = 100
        self.e = 1e-4
        model = {}

        # Input variable
        X_input = T.ftensor4('input')

        out = {}

        # REMINDER ABOUT CONVOLUTION NEURAL NETWORK
        # input shape (width, height, channel = depth)

        # every neuron in one depth of the output (activation) is using the same weights.  So for every depth,
        # every neuron is calculated using the same filter, its  just the kernel that changes (the position over the input)
        # For example if input shape (10, 10, 3) and conv layer is of form (num_filters=num_channel=16, filter_size=4, nopadding, stride=1), then
        # there will be 16 filters of size  (4, 4, 3)

        # (BATCH_SIZE, 1, 28, 28)
        model['l_in'] = InputLayer(shape=(None, 1, 28, 28),
                                   input_var=X_input)
        # (BATCH_SIZE, 16, 26, 26)
        model['l_conv1'] = Conv2DLayer(
            model['l_in'], num_filters=16, filter_size=3, pad='valid')  # valid : no padding
        # (BATCH_SIZE, 16, 24, 24)
        model['l_conv2'] = Conv2DLayer(
            model['l_in'], num_filters=16, filter_size=3, pad='valid')
        # (BATCH_SIZE, 16, 12, 12)
        model['l_max1'] = MaxPool2DLayer(model['l_conv2'], pool_size=2)
        # (BATCH_SIZE, 32, 10, 10)
        model['l_conv3'] = Conv2DLayer(model['l_max1'], num_filters=32, filter_size=3, pad='valid')
        # (BATCH_SIZE, 32, 5, 5)
        model['l_max2'] = MaxPool2DLayer(model['l_conv3'], pool_size=2)
        # (BATCH_SIZE, 800)
        model['l_resh1'] = ReshapeLayer(model['l_max2'], ([0], -1))
        # (BATCH_SIZE, 128)
        model['l_hid1'] = DenseLayer(
            model['l_resh1'],
            num_units=128)

        model['l_mu'] = DenseLayer(model['l_hid1'],
                                   self.LATENT_SPACE_SIZE,
                                   nonlinearity=lasagne.nonlinearities.linear)

        # (BATCH_SIZE, 16)
        model['l_sttdev'] = DenseLayer(model['l_hid1'],
                                       self.LATENT_SPACE_SIZE,
                                       nonlinearity=lasagne.nonlinearities.linear)

        # (BATCH_SIZE, 16)
        model['l_z'] = GaussianPropLayer(model['l_mu'], model['l_sttdev'])
        # (BATCH_SIZE, 128)
        model['l_hid_d'] = DenseLayer(model['l_z'],
                                      128)
        # (BATCH_SIZE, 800)
        model['l_hid1_d'] = DenseLayer(model['l_hid_d'],
                                       800)
        # (BATCH_SIZE, 32, 5, 5)
        model['l_resh_d'] = ReshapeLayer(model['l_hid1_d'], ([0], 2 * 16, 5, 5))
        # (BATCH_SIZE, 32, 10, 10)
        model['l_ups1_d'] = Upscale2DLayer(model['l_resh_d'], 2,
                                           mode='repeat')  # mode repeat (double the value X times), possibility of dilating the values with 'dilate')
        # (BATCH_SIZE, 16, 12, 12)
        model['l_conv1_d'] = Conv2DLayer(model['l_ups1_d'], num_filters=16, filter_size=3,
                                         pad='full')  # full : adding a pd on both sides of size (filter_size - 1)  so that the first output will be calculated considering only the first input
        # (BATCH_SIZE, 16, 24, 24)
        model['l_ups2_d'] = Upscale2DLayer(model['l_conv1_d'], 2)
        # (BATCH_SIZE, 16, 26, 26)
        model['l_conv2_d'] = Conv2DLayer(model['l_ups2_d'], num_filters=16, filter_size=3, pad='full')
        # (BATCH_SIZE, 1, 28, 28)
        model['l_conv3_d'] = Conv2DLayer(model['l_conv2_d'], num_filters=1, filter_size=3, pad='full',
                                         nonlinearity=lasagne.nonlinearities.sigmoid)
        # (BATCH_SIZE, 784
        model['l_out'] = ReshapeLayer(model['l_conv3_d'], ([0], -1))

        # output image (shape = batch_size, 784) during training
        Y_output = get_output(model['l_out'], deterministic=False)

        # output image (shape = batch_size, 784) during testing
        # Y_output_test = get_output(model['l_out'], deterministic=True)

        # Get the mean and stddev vectors
        z_sttdev = get_output(model['l_sttdev'])
        z_mean = get_output(model['l_mu'])

        # KL divergence for gaussian distribution
        # https://arxiv.org/pdf/1312.6114.pdf
        self.latent_loss = 0.5 * T.sum(T.square(z_mean) + T.square(z_sttdev) - T.log(T.square(z_sttdev) + self.e) - 1,
                                       axis=1)
        # Reconstruction loss
        self.reconstruction_loss = T.nnet.binary_crossentropy(Y_output + self.e,
                                                              X_input.reshape((-1, 784))).sum(axis=1)
        self.loss = (self.latent_loss + self.reconstruction_loss)

        # Reconstruction loss without Dropout (deterministic output)
        # self.reconstruction_loss_test = T.nnet.binary_crossentropy(Y_output_test, X_input.reshape((-1, 784))).sum()
        # self.loss_test = (self.latent_loss + self.reconstruction_loss_test)

        # Cost function
        self.cost = T.mean(self.loss, axis=0)
        # self.cost_test = T.mean(self.loss_test, axis=0)

        # Parameters of the neural network
        all_params = get_all_params(model['l_out'], trainable=True)

        # Adadelta optimizer (take into consideration the previous gradients in a windows size)
        optimizer = lasagne.updates.adadelta(self.cost, all_params)

        self.train_fn = theano.function([X_input], [self.cost], updates=optimizer)
        self.test_fn = theano.function([X_input], [self.cost])

    def train(self):
        X_train, _, X_val, _, X_test, _ = load_dataset()
        train_loss, test_loss = [], []
        for epoch in range(self.NUM_EPOCH):
            train_err = 0
            n_train_batches = 0
            start_time = time.time()
            for X_batch_train in iterate_minibatches(X_train, batchsize=self.BATCH_SIZE, shuffle=True):
                # print(X_batch_train[0][0][1])
                err_train = self.train_fn(X_batch_train)
                train_err += err_train[0]
                n_train_batches += 1

            val_err = 0
            n_val_batches = 0
            for X_batch_val in iterate_minibatches(X_val, self.BATCH_SIZE, shuffle=False):
                err = self.test_fn(X_batch_val)
                val_err += err[0]
                n_val_batches += 1
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, self.NUM_EPOCH, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / n_train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / n_val_batches))
            train_loss.append(train_err / n_train_batches)
            test_loss.append(val_err / n_val_batches)

        test_err = 0
        n_test_batches = 0
        for X_batch_test in iterate_minibatches(X_test, self.BATCH_SIZE, shuffle=False):
            err = self.test_fn(X_batch_test)
            test_err += err
            n_test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / n_test_batches))
        cPickle.dump(train_loss, open('train_loss.p', 'wb'))
        cPickle.dump(test_loss, open('test_loss.p', 'wb'))


if __name__ == '__main__':
    vae = VAE()
    vae.train()
