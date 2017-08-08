import tensorflow as tf
from ops import conv_with_lrelu, linear, lrelu, deconv

class vae(object):

    def __init__(self):
        self.ip_image_x = tf.placeholder(dtype= tf.float32, shape=[None, 28*28], name = "input_x")
        self.ip_image = tf.reshape(self.ip_image_x, [-1,28,28])
        # self.y = tf.placeholder(dtype= tf.int32, shape=[None], name="targets")
        self.latent_emb_size = 20
        # self.batch_size = tf.shape(self.ip_image)[0]
        self.batch_size = 100
        self.mean, self.std_dev = self.encoder()

        #generalization
        samples = tf.random_normal([self.batch_size, self.latent_emb_size], 0, 1, dtype=tf.float32)
        self.pred_latent =self.mean +(samples*self.std_dev)

        # self.loss = tf.reduce_mean(self.pred_latent)

        self.gen_image = self.decoder(ip=self.pred_latent)

        self.loss = self.compute_loss()

    def encoder(self):
        with tf.variable_scope('Encoder'):
            ip1 = tf.expand_dims(self.ip_image, [-1])
            h1 = conv_with_lrelu(ip=ip1, ip_channel=1, out_channel=16, scope='conv1')
            h2 = conv_with_lrelu(ip=h1, ip_channel=16, out_channel=32, scope='conv2')

            h2_flat = tf.reshape(h2, [self.batch_size, -1])

            h_mean = linear(h2_flat, ip_size=7*7*32, out_size=self.latent_emb_size, scope='mean')
            h_std_dev = linear(h2_flat, ip_size=7*7*32, out_size=self.latent_emb_size, scope='std')
        return h_mean, h_std_dev

    def decoder(self, ip):
        with tf.variable_scope('decoder'):
            gen_linear = linear(ip, ip_size=self.latent_emb_size, out_size=7*7*32, scope='gen_linear')
            img1 = tf.nn.relu(tf.reshape(gen_linear, [self.batch_size,7,7,32]))

            gen_h1 = lrelu(deconv(img1, [self.batch_size, 14,14,16], scope = 'gen_h1'))
            gen_h2 = tf.nn.sigmoid(deconv(gen_h1, [self.batch_size, 28, 28, 1], scope='gen_h2'))

        gen_image = gen_h2[:,:,:,0]
        return gen_image

    def compute_loss(self):
        flat_ip_img = tf.reshape(self.ip_image, [-1, 28*28])
        flat_gen_image = tf.reshape(self.gen_image, [-1, 28*28])
        # mean square of generated image and original image
        self.generation_loss = tf.reduce_mean(tf.square(self.gen_image - self.ip_image))
        # self.generation_loss = -tf.reduce_sum(flat_ip_img * tf.log(1e-8 + flat_gen_image)
        #                                       + (1-flat_ip_img) * tf.log(1e-8 + 1 - flat_gen_image),1)

        #KL-Divergence between latent variable and unit gaussian
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(self.mean) + tf.square(self.std_dev) - tf.log(tf.square(self.std_dev)) - 1, 1)

        total_loss = self.generation_loss+tf.reduce_mean(self.latent_loss)

        return total_loss