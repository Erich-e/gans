import os

import horovod.tensorflow as hvd
import mnist

hvd.init()
config = tf.ConfigProto()
# Only use the GPU that horovod gives us
config.gpu_options.visible_device_list = str(hvd.local_rank())

ctx = mnist.Context()
ctx.random_dim = 100
ctx.epochs = 51
ctx.batch_size = 128
ctx.opt = hvd.DistributedOptimizer(
    tf.keras.optimizers.Adam(lr=0.01 * hvd.size(), beta_1=0.5))


# Load, normalize and flatten mnist data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5 
x_train = x_train.reshape(60000, 784)

x_valid = (x_valid.astype(np.float32) - 127.5) / 127.5 
x_valid = x_valid.reshape(x_valid.shape[0], 784)

y_valid = np.ones(x_valid.shape[0])

def plot_images(filename, generator):
    '''
    Plot and save 100 sample images from the generator
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
                  optimizer=opt)

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
                      optimizer=opt)
discriminator.trainable = False

# Combine networks together 

gan_input = tf.keras.layers.Input((random_dim, ))

x = generator(gan_input)

gan_output = discriminator(x)

gan = tf.keras.models.Model(inputs=gan_input, outputs=gan_output)
gan.compile(loss='binary_crossentropy',
            optimizer=opt)

# Train the network

batch_size = 50
batch_count = x_train.shape[0] / batch_size

# Synchronize initial values across workers
hvd.BroadcastGlobalVariablesHook(0).on_train_begin()

for e in xrange(epochs):
    filename = './samples/mnist_{0:02d}.png'.format(e)

    plot_images(filename, generator)

    gen_loss = []
    disc_loss = []

    for i in xrange(batch_count):
        if i%200 == 0:
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
        y_dis[batch_size:] = 0.9

        discriminator.trainable = True
        disc_loss.append(discriminator.train_on_batch(X, y_dis))
        discriminator.trainable = False

        # Generator training

        # Make some noise for our input
        noise = np.random.normal(0, 1, size=(batch_size, random_dim))

        # Our goal is for the discriminator to believe all these images are real
        y_gen = np.ones(batch_size)

        gen_loss.append(gan.train_on_batch(noise, y_gen))

    gen_loss_avg = sum(gen_loss) / batch_count
    disc_loss_avg = sum(disc_loss) / batch_count

    print("discriminator loss {0}".format(gen_loss_avg))
    print("generator loss {0}".format(disc_loss))
    # print('discriminator')
    # discriminator.evaluate(x_valid, y_valid)
    # print('gan')
    # gan.evaluate(noise, y_gen)
