""" Train model.

--load          checkpoint directory if you want to continue training a model
--task          task to train on {'mnist', 'translated' or 'cluttered'}
--num_glimpses  # glimpses (fixed)
--n_patches     # resolutions extracted for each glimpse

"""

import tensorflow as tf
import numpy as np
import argparse
from   datetime import datetime
import pickle
from RAM    import RAM
from DRAM   import DRAM
from DRAM_loc import DRAMl
from config_dram import Config

from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':

    # ----- parse command line -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--task','-t', type=str, default='cluttered_var',
                        help='Task - ["org","translated","cluttered", "cluttered_var"].')

    parser.add_argument('--model','-m', type=str, default='dram_loc',
                        help='Model - "RAM" or "DRAM".')
    parser.add_argument('--load','-l', type=str, default=None,
                        help='Load model form directory.')
    parser.add_argument('--num_glimpses','-n', type=int, default=4,
                        help='Number of glimpses to take')
    parser.add_argument('--loc_std',type=float, default=0.11,
                        help='Standard deviation of Gaussian sampling.')
    parser.add_argument('--lr_start',type=float, default=1e-03,
                        help='Starting learning rate.')
    parser.add_argument('--n_patches','-np', type=int, default=3,
                        help='Number of patches for each glimpse')
    parser.add_argument('--use_context', default=True, action='store_true',
                    help='Use context network (True) or not (False)')
    parser.add_argument('--convnet', default=True, action='store_true',
                    help='True: glimpse sensor is convnet, False: fully-connected')

    parser.add_argument('--p_labels','-pl', type=float, default=1,
                        help='Fraction of labeled data')
    FLAGS, _ = parser.parse_known_args()

    time_str = datetime.now().strftime('%H%M%S')

    if FLAGS.model == 'ram':
        from config_dram import Config
    elif FLAGS.model == 'dram':
        from config_dram import Config
    elif FLAGS.model == 'dram_loc':
        from config_dram import Config
    else:
        print 'Unknown model {}'.format(FLAGS.model)
        exit()

    # parameters
    config  = Config()
    n_steps = config.step

    # parameters
    config.loc_std      = FLAGS.loc_std
    config.lr_start     = FLAGS.lr_start
    config.use_context  = FLAGS.use_context
    config.convnet      = FLAGS.convnet
    config.num_glimpses = FLAGS.num_glimpses
    config.n_patches    = FLAGS.n_patches
    config.p_labels     = FLAGS.p_labels

    # log directory
    FLAGS.logdir = "./experiments/task={}{}x{}_colordigits={}_colornoise={}_distr{}-{}_model={}_conv={}_n_glimpses={}_fovea={}x{}_std={}_{}_context={}_lr={}-{}_gamma={}_p_labels={}".format(
        FLAGS.task, config.new_size, config.new_size, config.color_digits, config.color_noise, config.distractor_range[0], config.distractor_range[1],
        FLAGS.model, config.convnet, config.num_glimpses, config.glimpse_size,config.glimpse_size,
        config.loc_std, time_str, config.use_context, config.lr_start, config.lr_min, config.gamma_start,config.p_labels)

    print '\n\nFlags: {}\n\n'.format(FLAGS)
    # ------------------------------

    # data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

    # init model
    config.num_glimpses = FLAGS.num_glimpses
    config.n_patches    = FLAGS.n_patches
    config.sensor_size  = config.glimpse_size**2 * config.n_patches
    config.N            = mnist.train.num_examples # number of training examples

    if FLAGS.model == 'ram':
        print '\n\n\nTraining RAM\n\n\n'
        model = RAM(config, logdir=FLAGS.logdir)
    elif FLAGS.model == 'dram':
        print '\n\n\nTraining DRAM\n\n\n'
        model = DRAM(config, logdir=FLAGS.logdir)
    elif FLAGS.model == 'dram_loc':
        print '\n\n\nTraining DRAM with location ground truth\n\n\n'
        model = DRAMl(config, logdir=FLAGS.logdir)
    else:
        print 'Unknown model {}'.format(FLAGS.model)
        exit()

    # load if specified
    if FLAGS.load != None:
        model.load(FLAGS.load)

    # display # parameters
    model.count_params()

    # train
    model.train(mnist, FLAGS.task)

    model.evaluate(data=mnist, task=FLAGS.task)
