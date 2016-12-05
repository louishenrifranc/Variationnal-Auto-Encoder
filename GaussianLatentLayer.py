import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from lasagne.layers.base import MergeLayer
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams


class GaussianPropLayer(MergeLayer):
    def __init__(self, z_mean, z_sttdev, **kwargs):
        super(GaussianPropLayer, self).__init__(incomings=[z_mean, z_sttdev], **kwargs)
        assert self.input_shapes[0][1] == self.input_shapes[1][1]

    def get_output_shape_for(self, inputs_shapes):
        return (self.input_shapes[0][0], self.input_shapes[0][1])

    def get_output_for(self, inputs, **kwargs):
        return self.sampler(inputs[0], inputs[1])

    def sampler(self, mean, stddev):
        """

        :param self:
        :param mean:
        :param stddev:
        :return:
        """
        seed = 123
        if "gpu" in theano.config.device:
            self.srng = CURAND_RandomStreams(seed=seed)
        else:
            self.srng = RandomStreams(seed=seed)
        eps = self.srng.normal(stddev.shape)
        z = mean + T.exp(stddev) * eps
        return z
