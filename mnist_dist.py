import os

import horovod.tensorflow as hvd
import mnist

if __name__ == '__main__':
    hvd.init()
    config = tf.ConfigProto()
    # Only use the GPU that horovod gives us
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Synchronize initial values across workers
    hvd.BroadcastGlobalVariablesHook(0).on_train_begin()

    ctx = mnist.Context()
    ctx.random_dim = 100
    ctx.epochs = 51
    ctx.batch_size = 128
    ctx.opt = hvd.DistributedOptimizer(
        tf.keras.optimizers.Adam(lr=0.01 * hvd.size(), beta_1=0.5))

    mnist.load_data(ctx)
    ctx.generator = mnist.greate_generator(ctx)
    ctx.discriminator = mnist.create_discriminator(ctx)
    ctx.gan = mnist.create_GAN(ctx)

    mnist.train(ctx)
