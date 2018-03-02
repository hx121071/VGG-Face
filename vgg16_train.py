import numpy as np
import tensorflow as tf
from network import Network


class Vgg16_train(Network):

    def __init__(self):
        self.input=[]
        self.data=tf.placeholder(tf.float32,shape=[10,224,224,3])
        self.labels=tf.placeholder(tf.float32,shape=[10])
        self.layers={'data':self.data,'labels':self.labels}
        self.keep_prob=0.5
        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(3,3,64,1,1,name='conv1_1')
             .conv(3,3,64,1,1,name='conv1_2')
             .max_pool(2,2,2,2,name='pool1',padding='VALID')
             .conv(3,3,128,1,1,name='conv2_1')
             .conv(3,3,128,1,1,name='conv2_2')
             .max_pool(2,2,2,2,name='pool2',padding='VALID')
             .conv(3,3,256,1,1,name='conv3_1')
             .conv(3,3,256,1,1,name='conv3_2')
             .conv(3,3,256,1,1,name='conv3_3')
             .max_pool(2,2,2,2,name='pool3',padding='VALID')
             .conv(3,3,512,1,1,name='conv4_1')
             .conv(3,3,512,1,1,name='conv4_2')
             .conv(3,3,512,1,1,name='conv4_3')
             .max_pool(2,2,2,2,name='pool4',padding='VALID')
             .conv(3,3,512,1,1,name='conv5_1')
             .conv(3,3,512,1,1,name='conv5_2')
             .conv(3,3,512,1,1,name='conv5_3')
             .max_pool(2,2,2,2,name='pool5',padding='VALID')
             .reshape(name='reshape6')
             .fc(4096,name='fc6')
             .dropout(self.keep_prob,name='dropout6')
             .fc(4096,name='fc7')
             .dropout(self.keep_prob,name='dropout7')
             .fc(20,name='fc8',relu=False)
             .softmax(name='cls_prob'))
