import tensorflow as tf
from ops import conv_with_lrelu, linear

class vae(object):

    def __init__(self):
        self.x = tf.placeholder(dtype= tf.float32, shape=[None, 28,28], name = "input_x")
        self.y = tf.placeholder(dtype= tf.int32, shape=[None], name="targets")
        self.latent_emb_size = 20
        self.batch_size = tf.shape(self.y)[0]
        self.mean, self.std_dev = self.encoder()

        samples = tf.random_normal([self.batch_size, self.latent_emb_size], 0, 1, dtype=tf.float32)
        self.pred_latent =self.mean +(samples*self.std_dev)

        self.decoder(ip=self.pred_latent)

        self.compute_loss()

    def encoder(self):
        with tf.variable_scope('Encoder'):
            ip1 = self.x.expand_dims([-1])
            h1 = conv_with_lrelu(ip=ip1, ip_channel=1, out_channel=16, scope='conv1')
            h2 = conv_with_lrelu(ip=h1, ip_channel=16, out_channel=32, scope='conv2')

            h2_flat = tf.reshape(h2, [self.batch_size, -1])

            h_mean = linear(h2_flat, ip_size=tf.shape(h2_flat)[1], out_size=self.latent_emb_size, scope='mean')
            h_std_dev = linear(h2_flat, ip_size=tf.shape(h2_flat)[1], out_size=self.latent_emb_size, scope='std')
        return h_mean, h_std_dev

    def decoder(self, ip):


    def compute_loss(self):