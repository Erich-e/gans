import os

import numpy as np
import tensorflow as tf

# Ensure repeatability of noise
np.random.seed(1)
random_dim = 100

# Load, normalize and flatten mnist data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5 
x_train = x_train.reshape(60000, 784)


# Create the generator
generator = tf.keras.models.Sequential()
generator.add(
    tf.keras.layers.Dense(256,
                         input_dim=random_dim,
                         kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)))
generator.add(tf.keras.layers.LeakyReLU(0.2))

generator.add(tf.keras.layers.Dense(512))
generator.add(tf.keras.layers.LeakyReLU(0.2))

generator.add(tf.keras.layers.Dense(1024))
generator.add(tf.keras.layers.LeakyReLU(0.2))

generator.add(tf.keras.layers.Dense(784, activation='tanh'))

generator.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=0.002, beta_1=0.6))

# Create the discriminator using transfer learning from resnet

resnet_weights = '/some/file/path'

discriminator = tf.keras.models.Sequential()
discriminator.add(
    tf.keras.applications.ResNet50(include_top=False,
                                  pooling='avg',
                                  weights=resnet_weights))

discriminator.add(tf.keras.layers.Dense(2, tf.keras.layers.Softmax))

discriminator.layers[0].trainable = False

discriminator.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=0.002, beta_1=0.9))

# Combine networks together 

discriminator.trainable = False

gan_input = tf.keras.layers.Input((random_dim, ))

x = generator(gan_input)