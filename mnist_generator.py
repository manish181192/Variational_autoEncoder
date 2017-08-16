import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from vae import vae
import numpy as np
from scipy.misc import imsave
from ops import merge

mnist = input_data.read_data_sets('MNIST_data')
# display(mnist.train.images[0,:])
vae = vae()

opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(vae.loss)

with tf.Session() as sess:
    batch = mnist.train.next_batch(100)
    imsave("results/base.jpg", merge(np.reshape(batch[0], [100, 28, 28])[:64], [8, 8]))

    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch = mnist.train.next_batch(100)[0]

        _, loss, gen_images_train = sess.run([opt, vae.loss, vae.gen_image],feed_dict={vae.ip_image_x: batch})
        print('epoch'+str(i)+' : '+str(loss))
        if i%100 == 0:
            eval_batch = mnist.test.images[:100]
            gen_images_test = sess.run([vae.gen_image], feed_dict={vae.ip_image_x: eval_batch})
            imsave("results/" + str(i) + ".jpg", merge(gen_images_test[0][:64], [8, 8]))
# f = open('gen_images','wb')
#
# for image in gen_images_test:
#     flat_image = np.reshape(image, [-1])
#     f.write(str(flat_image)+'\n')
# f.close()