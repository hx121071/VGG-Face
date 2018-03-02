import numpy as np
import tensorflow as tf
from vgg16_train import Vgg16_train
import os
import sys

class SolverWrapper():

    def __init__(self,net,sess,saver,data,labels,output_dir,sample_batch):

        self.net=net
        self.saver=saver
        self.data=data
        self.labels=labels
        self.data_num=data.shape[0]
        self.output_dir=output_dir
        self.sample_batch=sample_batch


    def snapshot(self,sess,iter):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filename='train_weights_'+'iter{:d}'.format(iter+1)+'.ckpt'
        output_dir=os.path.join(self.output_dir,filename)
        saver.save(sess,output_dir)
        print('Wrote Wrote snapshot to : {:s}'.format(filename))
    def get_sample_data(self,index):
        return self.data[index],self.labels[index]
    def train_model(self,sess,max_iters):

        cls_prob=self.net.get_output('cls_prob')

	# sample_batchX2622

        #loss fanction
        labels=self.net.labels
        labels= tf.cast(labels,tf.int32)
        batch_size = tf.size(labels) # get size of labels : 4
        labels = tf.expand_dims(labels, 1) # 增加一个维度
        indices = tf.expand_dims(tf.range(0, batch_size,1), 1) #生成索引
        print(indices.get_shape(),labels.get_shape())
        concated = tf.concat([indices, labels] , 1) #作为拼接
        onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size,20]), 1.0, 0.0)
        # gt_onehot=np.zeros(cls_prob.shape)
        #
        # gt_onehot[np.arange(cls_prob.shape[0]),gt.eval()]=1
        sf_loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels,logits=cls_prob),axis=1)

        #train_op
        global_step = tf.Variable(0)
        #--------------学习速率的设置（学习速率呈指数下降）---------------------
        learning_rate = tf.train.exponential_decay(1e-2,global_step,decay_steps=self.data.shape[0]/self.sample_batch,decay_rate=0.98,staircase=True)
        train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(sf_loss)

        sess.run(tf.global_variables_initializer())

        for i  in range(max_iters):
            sample_index=np.random.choice(self.data_num,self.sample_batch)
            sample_x,sample_labels=self.get_sample_data(sample_index)

            feed_dict={self.net.data:sample_x,self.net.labels:sample_labels}

            _,loss=sess.run((train_op,sf_loss),feed_dict=feed_dict)

            if i==0 or (i+1)%1000==0:
                print("loss is ",loss)
                self.snapshot(sess,i)
