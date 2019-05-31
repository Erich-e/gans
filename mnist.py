import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Hyperparameters
# np.random.seed(1)
random_dim = 100

epochs = 40
batch_size = 128

# Load, normalize and flatten mnist data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5 
x_train = x_train.reshape(60000, 784)

def plot_images(filename, generator):
    '''
    Plot and save 100 smaple images from the generator
    '''
    noise = np.random.normal(0, 1, size=[100, random_dim])

    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100, 28, 28)

    plt.figure(figsize=(10, 10))
    for i, image in enumerate(generated_images):
        plt.subplot(10, 10, i+1)
        plt.imshow(image, interpolation='nearest', cmap='gray_r')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(filename)


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

generator.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam())

# Create the discriminator using transfer learning from resnet

# resnet_weights = './resnet_weights.h5'

# discriminator = tf.keras.models.Sequential()
# discriminator.add(
#     tf.keras.applications.ResNet50(include_top=False,
#                                   pooling='avg',
#                                   weights=resnet_weights))

# discriminator.add(tf.keras.layers.Dense(1, tf.keras.activations.sigmoid))

# discriminator.layers[0].trainable = False

# discriminator.compile(loss='binary_crossentropy',
#                       optimizer=tf.keras.optimizers.Adam(lr=0.002, beta_1=0.9))

discriminator = tf.keras.models.Sequential()
discriminator.add(
    tf.keras.layers.Dense(256,
                          input_dim=784,
                          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)))
discriminator.add(tf.keras.layers.LeakyReLU(0.2))
discriminator.add(tf.keras.layers.Dropout(0.2))

discriminator.add(tf.keras.layers.Dense(512))
discriminator.add(tf.keras.layers.LeakyReLU(0.2))
discriminator.add(tf.keras.layers.Dropout(0.2))

discriminator.add(tf.keras.layers.Dense(1024))
discriminator.add(tf.keras.layers.LeakyReLU(0.2))
discriminator.add(tf.keras.layers.Dropout(0.2))

discriminator.add(tf.keras.layers.Dense(1, tf.keras.activations.sigmoid))

discriminator.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam())

# Combine networks together 

gan_input = tf.keras.layers.Input((random_dim, ))

x = generator(gan_input)

gan_output = discriminator(x)

gan = tf.keras.models.Model(inputs=gan_input, outputs=gan_output)
gan.compile(loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam())

# Train the network

batch_size = 50
batch_count = x_train.shape[0] / batch_size


for e in xrange(epochs):
    filename = './samples/mnist_{0}.png'.format(e)

    plot_images(filename, generator)

    for i in xrange(batch_count):
        print('Running epoch {0}, batch {1}'.format(e, i))

        # Discriminator training

        # Create real/generated batch of training data
        noise = np.random.normal(0, 1, size=(batch_size, random_dim))
        # Just pick random images, cross-validation too much work
        images_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

        generated_images = generator.predict(noise)
        X = np.concatenate([generated_images, images_batch])

        # Create labels for discriminator data
        y_dis = np.zeros(2 * batch_size)
        # one-sided label smoothing
        y_dis[:batch_size] = 0.9

        discriminator.trainable = True
        discriminator.train_on_batch(X, y_dis)
        discriminator.trainable = False

        # Generator training

        # Make some noise for our input
        noise = np.random.normal(0, 1, size=(batch_size, random_dim))

        # Our goal is for the discriminator is to believe all these images are real
        y_gen = np.ones(batch_size)

        gan.train_on_batch(noise, y_gen)
        

x_valid = (x_valid.astype(np.float32) - 127.5) / 127.5 
x_valid = x_valid.reshape(x_valid.shape[0], 784)

y_valid = np.ones(x_valid.shape[0])

discriminator.evaluate(x_valid, y_valid)