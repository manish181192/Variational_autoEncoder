import tensorflow as tf
from ops import conv_with_lrelu, linear, lrelu, deconv

class vae(object):

    def __init__(self):
        self.ip_image = tf.placeholder(dtype= tf.float32, shape=[None, 28,28], name = "input_x")
        # self.y = tf.placeholder(dtype= tf.int32, shape=[None], name="targets")
        self.latent_emb_size = 20
        self.batch_size = tf.shape(self.ip_image)[0]
        self.mean, self.std_dev = self.encoder()

        #generalization
        samples = tf.random_normal([self.batch_size, self.latent_emb_size], 0, 1, dtype=tf.float32)
        self.pred_latent =self.mean +(samples*self.std_dev)

        self.gen_image = self.decoder(ip=self.pred_latent)

        self.loss = self.compute_loss()

    def encoder(self):
        with tf.variable_scope('Encoder'):
            self.ip1 = tf.expand_dims(self.ip_image, [-1])
            self.h1 = conv_with_lrelu(ip=self.ip1, ip_channel=1, out_channel=16, scope='conv1')
            self.h2 = conv_with_lrelu(ip=self.h1, ip_channel=16, out_channel=32, scope='conv2')

            self.h2_flat = tf.reshape(self.h2, [self.batch_size, -1])

            h_mean = linear(self.h2_flat, ip_size=tf.shape(self.h2_flat)[1], out_size=self.latent_emb_size, scope='mean')
            h_std_dev = linear(self.h2_flat, ip_size=tf.shape(self.h2_flat)[1], out_size=self.latent_emb_size, scope='std')
        return h_mean, h_std_dev

    def decoder(self, ip):
        with tf.variable_scope('encoder'):
            self.gen_linear = linear(ip, ip_size=self.latent_emb_size, out_size=tf.shape(self.h2_flat)[1], scope='gen_linear')
            self.img1 = tf.nn.relu(tf.reshape(self.gen_linear, [self.batch_size,7,7,32]))

            self.gen_h1 = lrelu(deconv(self.img1, [self.batch_size, 14,14,16], scope = 'gen_h1'))
            self.gen_h2 = tf.nn.sigmoid(deconv(self.img1, [self.batch_size, 28, 28, 1], scope='gen_h2'))

        gen_image = self.gen_h2
        return gen_image

    def compute_loss(self):
        # mean square of generated image and original image
        self.generated_loss = tf.reduce_mean(tf.square(self.gen_image - self.ip_image))

        #KL-Divergence between latent variable and unit gaussian

        return 0