import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Hyperparameters

# Dump hyperparameterish stuff in this for now
class Context(object):
    pass

def load_data(ctx):
    '''
    Load, normalize and flatten mnist data
    '''
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()

    x_train = (x_train.astype(np.float32) - 127.5) / 127.5 
    x_valid = (x_valid.astype(np.float32) - 127.5) / 127.5 

    ctx.x_train = x_train.reshape(x_train.shape[0], 784)
    ctx.x_valid = x_valid.reshape(x_valid.shape[0], 784)
    ctx.y_valid = np.ones(x_valid.shape[0])

def plot_images(ctx, filename):
    '''
    Plot and save 100 sample images from the generator
    '''
    noise = np.random.normal(0, 1, size=[100, ctx.random_dim])

    generated_images = ctx.generator.predict(noise)
    generated_images = generated_images.reshape(100, 28, 28)

    plt.figure(figsize=(10, 10))
    for i, image in enumerate(generated_images):
        plt.subplot(10, 10, i+1)
        plt.imshow(image, interpolation='nearest', cmap='gray_r')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(filename)

def create_generator(ctx):
    generator = tf.keras.models.Sequential()
    generator.add(
        tf.keras.layers.Dense(256,
                            input_dim=ctx.random_dim,
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)))
    generator.add(tf.keras.layers.LeakyReLU(0.2))

    generator.add(tf.keras.layers.Dense(512))
    generator.add(tf.keras.layers.LeakyReLU(0.2))

    generator.add(tf.keras.layers.Dense(1024))
    generator.add(tf.keras.layers.LeakyReLU(0.2))

    generator.add(tf.keras.layers.Dense(784, activation='tanh'))

    generator.compile(loss='binary_crossentropy',
                    optimizer=ctx.opt)

    return generator

def create_dc_generator(ctx):
    '''
    DCGAN architecture defined https://arxiv.org/pdf/1511.06434.pdf
    '''

    generator = tf.keras.models.Sequential()

    # (batch) x random_dim -> (batch) x 16 * filters
    generator.add(tf.keras.layers.Dense(16 * ctx.filters,
                                        activation='tanh',
                                        input_shape=(ctx.random_dim, ),
                                        kernel_initializer=ctx.kernel_initializer))

    # (batch) x 16 * filters -> (batch) x 4 x 4 x filters
    generator.add(tf.keras.layers.Reshape((4, 4, ctx.filters)))

    # (batch) x 4 x 4 x filters -> (batch) x 8 x 8 x filters*8
    generator.add(
        tf.keras.layers.Conv2DTranspose(ctx.filters * 8,
                                        kernel_size=4,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=ctx.kernel_initializer,
                                        data_format='channels_last'))
    generator.add(tf.keras.layers.LeakyReLU(0.2))
    generator.add(tf.keras.layers.BatchNormalization())

    # (batch) x 8 x 8 x filters*8 -> (batch) x 16 x 16 x filters*4
    generator.add(
        tf.keras.layers.Conv2DTranspose(ctx.filters * 4,
                                        kernel_size=4,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=ctx.kernel_initializer,
                                        data_format='channels_last'))
    generator.add(tf.keras.layers.LeakyReLU(0.2))
    generator.add(tf.keras.layers.BatchNormalization())

    # (batch) x 16 x 16 x filters*4 -> (batch) x 32 x 32 x filters*2
    generator.add(
        tf.keras.layers.Conv2DTranspose(ctx.filters * 2,
                                        kernel_size=4,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=ctx.kernel_initializer,
                                        data_format='channels_last'))
    generator.add(tf.keras.layers.LeakyReLU(0.2))
    generator.add(tf.keras.layers.BatchNormalization())

    # (batch) x 32 x 32 x filters*2 -> (batch) x 64 x 64 x filters
    generator.add(
        tf.keras.layers.Conv2DTranspose(ctx.filters,
                                        kernel_size=4,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=ctx.kernel_initializer,
                                        data_format='channels_last'))
    generator.add(tf.keras.layers.LeakyReLU(0.2))
    generator.add(tf.keras.layers.BatchNormalization())

    # (batch) x 64 x 64 x filters -> (batch) x 128 x 128 x channels
    generator.add(
        tf.keras.layers.Conv2DTranspose(ctx.channels,
                                        kernel_size=4,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=ctx.kernel_initializer,
                                        data_format='channels_last'))
    generator.add(tf.keras.layers.LeakyReLU(0.2))

    # (batch) x 128 x 128 x channels -> (batch) x image_dim x image_dim x channels
    padding_amount = ctx.image_dim - (128 % ctx.image_dim)
    compression_ratio = (128 + padding_amount) // ctx.image_dim
    generator.add(
        tf.keras.layers.ZeroPadding2D((padding_amount // 2),
                                      data_format='channels_last'))
    generator.add(
        tf.keras.layers.MaxPooling2D(pool_size=compression_ratio,
                                     data_format='channels_last'))

    # (batch) x image_dim x image_dim x channels -> (batch) x image_dim*image_dim*channels
    generator.add(tf.keras.layers.Flatten(data_format='channels_last'))

    generator.compile(loss='binary_crossentropy',
                      optimizer=ctx.opt)
    
    return generator

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

def create_discriminator(ctx):
    """
    Boilerplate DNN classifier
    """
    discriminator = tf.keras.models.Sequential()
    discriminator.add(
        tf.keras.layers.Dense(1024,
                            input_dim=784,
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)))
    discriminator.add(tf.keras.layers.LeakyReLU(0.2))
    discriminator.add(tf.keras.layers.Dropout(0.3))

    discriminator.add(tf.keras.layers.Dense(512))
    discriminator.add(tf.keras.layers.LeakyReLU(0.2))
    discriminator.add(tf.keras.layers.Dropout(0.3))

    discriminator.add(tf.keras.layers.Dense(256))
    discriminator.add(tf.keras.layers.LeakyReLU(0.2))
    discriminator.add(tf.keras.layers.Dropout(0.3))

    discriminator.add(tf.keras.layers.Dense(1, tf.keras.activations.sigmoid))

    discriminator.compile(loss='binary_crossentropy',
                        optimizer=ctx.opt)
    discriminator.trainable = False

    return discriminator

def create_dc_discriminator(ctx):
    """
    DCGAN architecture defined https://arxiv.org/pdf/1511.06434.pdf
    """
    discriminator = tf.keras.models.Sequential()

    # (batch) x image_dim*image_dim*channels -> (batch) x image_dim x image_dim x channels
    discriminator.add(
        tf.keras.layers.Reshape((ctx.image_dim, ctx.image_dim, ctx.channels)))

    # (batch) x image_dim x image_dim x channels -> (batch) x 128 x 128 x channels
    padding_amount = ctx.image_dim - (128 % ctx.image_dim)
    compression_ratio = (128 + padding_amount) // ctx.image_dim
    discriminator.add(
        tf.keras.layers.UpSampling2D(size=compression_ratio,
                                     data_format='channels_last'))
    discriminator.add(
        tf.keras.layers.ZeroPadding2D((padding_amount // 2),
                                      data_format='channels_last'))

    # (batch) x 128 x 128 x channels -> (batch) x 64 x 64 x filters
    discriminator.add(
        tf.keras.layers.Conv2D(ctx.filters,
                               kernel_size=4,
                               strides=2,
                               padding='same',
                               kernel_initializer=ctx.kernel_initializer,
                               data_format='channels_last'))
    discriminator.add(tf.keras.layers.LeakyReLU(0.2))

    # (batch) x 64 x 64 x filters -> (batch) x 32 x 32 x filters*2
    discriminator.add(
        tf.keras.layers.Conv2D(ctx.filters*2,
                               kernel_size=4,
                               strides=2,
                               padding='same',
                               kernel_initializer=ctx.kernel_initializer,
                               data_format='channels_last'))
    discriminator.add(tf.keras.layers.LeakyReLU(0.2))
    discriminator.add(tf.keras.layers.BatchNormalization())
    
    # (batch) x 32 x 32 x filters*2 -> (batch) x 16 x 16 x filters*4
    discriminator.add(
        tf.keras.layers.Conv2D(ctx.filters*4,
                               kernel_size=4,
                               strides=2,
                               padding='same',
                               kernel_initializer=ctx.kernel_initializer,
                               data_format='channels_last'))
    discriminator.add(tf.keras.layers.LeakyReLU(0.2))
    discriminator.add(tf.keras.layers.BatchNormalization())

    # (batch) x 16 x 16 x filters*4 -> (batch) x 8 x 8 x filters*8
    discriminator.add(
        tf.keras.layers.Conv2D(ctx.filters*8,
                               kernel_size=4,
                               strides=2,
                               padding='same',
                               kernel_initializer=ctx.kernel_initializer,
                               data_format='channels_last'))
    discriminator.add(tf.keras.layers.LeakyReLU(0.2))
    discriminator.add(tf.keras.layers.BatchNormalization())

    # (batch) x 8 x 8 x filters*8 -> (batch) x 4 x 4 x filters*16
    discriminator.add(
        tf.keras.layers.Conv2D(ctx.filters*8,
                               kernel_size=4,
                               strides=2,
                               padding='same',
                               kernel_initializer=ctx.kernel_initializer,
                               data_format='channels_last'))
    discriminator.add(tf.keras.layers.LeakyReLU(0.2))
    discriminator.add(tf.keras.layers.BatchNormalization())

    discriminator.add(
        tf.keras.layers.Dense(1, activation="sigmoid"))

    discriminator.compile(loss='binary_crossentropy',
                          optimizer=ctx.opt)
    discriminator.trainable = False

    return discriminator

def create_GAN(ctx):
    '''
    Combine networks together
    '''
    gan_input = tf.keras.layers.Input((ctx.random_dim, ))

    x = ctx.generator(gan_input)

    gan_output = ctx.discriminator(x)

    gan = tf.keras.models.Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy',
                optimizer=ctx.opt)

    return gan

# Train the network

def train(ctx):
    batch_count = ctx.x_train.shape[0] // ctx.batch_size

    print(batch_count)

    for e in range(ctx.epochs):
        filename = './train_output/mnist_{0:02d}.png'.format(e)

        plot_images(ctx, filename)

        gen_loss = []
        disc_loss = []

        for i in range(batch_count):
            # if i%200 == 0:
            #     print('Running epoch {0}, batch {1}'.format(e, i))

            print('Running epoch {0}, batch {1}'.format(e, i))

            # Discriminator training

            # Create real/generated batch of training data
            noise = np.random.normal(0, 1, size=(ctx.batch_size, ctx.random_dim))
            # Just pick random images, cross-validation too much work
            images_batch = ctx.x_train[np.random.randint(0, ctx.x_train.shape[0], size=ctx.batch_size)]

            generated_images = ctx.generator.predict(noise)
            X = np.concatenate([generated_images, images_batch])

            # Create labels for discriminator data
            y_dis = np.zeros(2 * ctx.batch_size)
            # one-sided label smoothing
            y_dis[ctx.batch_size:] = 0.9

            ctx.discriminator.trainable = True
            disc_loss.append(ctx.discriminator.train_on_batch(X, y_dis))

            # Generator training

            # Make some noise for our input
            noise = np.random.normal(0, 1, size=(ctx.batch_size, ctx.random_dim))

            # Our goal is for the discriminator to believe all these images are real
            y_gen = np.ones(ctx.batch_size)

            ctx.discriminator.trainable = False
            gen_loss.append(ctx.gan.train_on_batch(noise, y_gen))

        gen_loss_avg = sum(gen_loss) / batch_count
        disc_loss_avg = sum(disc_loss) / batch_count

        print('discriminator loss {0}'.format(gen_loss_avg))
        print('generator loss {0}'.format(disc_loss))
        # print('discriminator')
        # discriminator.evaluate(x_valid, y_valid)
        # print('gan')
        # gan.evaluate(noise, y_gen)

if __name__ == '__main__':
    ctx = Context()

    ctx.random_dim = 100
    ctx.image_dim = 28
    ctx.filters = 64
    ctx.channels = 1
    ctx.batch_size = 128
    ctx.epochs = 51
    ctx.opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    ctx.kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.02)

    load_data(ctx)
    ctx.generator = create_dc_generator(ctx)
    ctx.discriminator = create_dc_discriminator(ctx)
    ctx.gan = create_GAN(ctx)

    train(ctx)
