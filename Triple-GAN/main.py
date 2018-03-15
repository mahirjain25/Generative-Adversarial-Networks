from TripleGAN import TripleGAN

from utils import show_all_variables
from utils import check_folder

import tensorflow as tf


"""main"""
def main():
    # parse arguments

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = TripleGAN(sess, epoch=1, batch_size=10, unlabel_batch_size=125,
                        z_dim=100, dataset_name='cifar10', n=4000, gan_lr = 2e-4, cla_lr = 2e-3,
                        checkpoint_dir='checkpoint', result_dir='results', log_dir='logs')

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        gan.visualize_results(0)
        print(" [*] Testing finished!")

if __name__ == '__main__':
    main()