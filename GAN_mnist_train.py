import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
import numpy as np
import GAN_mnist
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/GAN_mnist_data', 'Directory to put the training data.')
flags.DEFINE_string('train_dir', '/tmp/GAN_mnist_train', 'Directory to put the training data.')
flags.DEFINE_string('output_dir', '/tmp/GAN_mnist_output', 'Directory to put the output data.')
tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')

def placeholder_inputs():
    # MNIST real data placeholder
    X = tf.placeholder(tf.float32, shape=[None, GAN_mnist.IMAGE_PIXELS])
    # Generator noise input placeholder
    Z = tf.placeholder(tf.float32, shape=[None, 100])
    return X, Z
    
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig
    
def train():
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    with tf.Graph().as_default():
        _ = tf.contrib.framework.get_or_create_global_step()
        X, Z = placeholder_inputs()
        
        G_sample, G_vars = GAN_mnist.inference_generative(Z) # removed G_vars
        with tf.variable_scope("Descriminator_net") as scope:
            D_real, D_vars = GAN_mnist.inference_descriminator(X, 1)
            scope.reuse_variables() # Use the same network parameters twice for the discriminator
            D_fake, _ = GAN_mnist.inference_descriminator(G_sample, 0)
        
        G_loss = GAN_mnist.loss_generator(D_fake)
        D_loss = GAN_mnist.loss_discriminator(D_real, D_fake)
        
        D_train_op, G_train_op = GAN_mnist.train(G_loss, D_loss, G_vars, D_vars)
        
        sess = tf.Session()
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
        sess.run(tf.initialize_all_variables())
        
        for i in range(FLAGS.max_steps):
            X_mb, _ = mnist.train.next_batch(GAN_mnist.BATCH_SIZE)
            _, D_loss_curr = sess.run([D_train_op, D_loss], feed_dict={X: X_mb, Z: np.random.uniform(-1., 1., size=[GAN_mnist.BATCH_SIZE, GAN_mnist.GEN_INPUT])})
            _, G_loss_curr = sess.run([G_train_op, G_loss], feed_dict={Z: np.random.uniform(-1., 1., size=[GAN_mnist.BATCH_SIZE, GAN_mnist.GEN_INPUT])})
        
            if i % 200 == 0:
                format_str = ('step %d, discriminator loss = %.2f, generator loss = %.2f')
                print (format_str % (i, D_loss_curr, G_loss_curr))
                               
                summary_str = sess.run(summary_op, feed_dict={X: X_mb, Z: np.random.uniform(-1., 1., size=[GAN_mnist.BATCH_SIZE, GAN_mnist.GEN_INPUT])})
                summary_writer.add_summary(summary_str, i)
                summary_writer.flush()
                
            if i % 200 == 0:
                samples = sess.run(G_sample, feed_dict={Z: np.random.uniform(-1., 1., size=[16, GAN_mnist.GEN_INPUT])})
                fig = plot(samples)
                plt.savefig(FLAGS.output_dir+'/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                plt.close(fig)

def main(_):
    if tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.DeleteRecursively(FLAGS.output_dir)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    train()

if __name__ == '__main__':
  tf.app.run()