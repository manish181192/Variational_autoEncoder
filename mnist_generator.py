import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from vae import vae
import png

def display(image):
    rows = 28
    cols = 28
    output_filename = 'temp.png'
    print("writing " + output_filename)
    with open(output_filename, "wb") as h:
        w = png.Writer(cols, rows, greyscale=True)
        data_i = [
            image[(rows * cols + j * cols): (rows * cols + (j + 1) * cols)]
            for j in range(rows)
            ]
        w.write(h, data_i)

mnist = input_data.read_data_sets('MNIST_data')
# display(mnist.train.images[0,:])
vae = vae()

opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(vae.loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        batch = mnist.train.next_batch(100)
        _, loss, gen_images = sess.run([opt, vae.loss, vae.gen_image],feed_dict={vae.ip_image_x: batch[0]})

f = open('gen_images','wb')
f.write(gen_images)
f.close()