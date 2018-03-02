import numpy as np
import tensorflow as tf
from vgg16_train import Vgg16_train
from train import SolverWrapper



def train_net(network,x,y,output_dir,sample_batch=10,max_iters=40000):
    saver=tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sw=SolverWrapper(network,sess,saver,x,y,output_dir,sample_batch)
        print('Solving.............')
        loss_l,iter_l=sw.train_model(sess,max_iters)
        print('done solving')


network=Vgg16_train()
output_dir="VGG16"
data = np.load("vgg_face.npy")
label = np.load("vgg_y.npy")
train_net(network,data,label,"vgg16")
