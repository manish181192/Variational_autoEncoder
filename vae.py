import tensorflow as tf
from ops import conv_with_lrelu, linear

class vae(object):

    def __init__(self):
        self.x = tf.placeholder(dtype= tf.float32, shape=[None, 28,28], name = "input_x")
        self.y = tf.placeholder(dtype= tf.int32, shape=[None], name="targets")
        self.latent_emb_size = 20
        self.mean, self.std_dev = self.encoder()
        self.decoder()

        self.compute_loss()

    def encoder(self):
        with tf.variable_scope('Encoder'):
            ip1 = self.x.reshape([-1, 28,28,1])
            h1 = conv_with_lrelu(ip=ip1, ip_channel=1, out_channel=16, scope='conv1')
            h2 = conv_with_lrelu(ip=h1, ip_channel=16, out_channel=32, scope='conv2')

            h2_flat = tf.reshape(h2, [h2,tf.shape(self.y), -1])

            latent_emb = linear()

            mean =
            std_dev =
        return mean, std_dev

    def decoder(self):


    def compute_loss(self):