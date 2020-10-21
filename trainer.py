import tensorflow.compat.v1 as tf
import os 
import numpy as np

from data_utils.data_loader import DataGenerator



class Seg(object):
    '''
    Control the training, evaluation, and inference process.
    Args:
    - sess: tf.session
    - net_name: string, __all__ = ['unet'].
    - lr: float, learning rate.
    - n_epoch: integer, the epoch number
    - channels: integer, the channel number of the input
    - num_classes: integer, the number of class
    - input_shape: tuple of integer, input dim
    - batch_size: integer
    - is_training: bool, control flag
    - weight_path: weight path of pre-trained model
    '''
    def __init__(self,sess,net_name=None,lr=1e-3,n_epoch=40,channels=1,num_classes=2,input_shape=(256,256),
                batch_size=16,is_training=True,weight_path=None): 
        super(Seg,self).__init__()    
        
        self.sess = sess
        self.net_name = net_name
        self.lr = lr
        self.n_epoch = n_epoch
        self.channels = channels
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.is_training = is_training
        
        self.weight_path = weight_path
        self.start_epoch = 0
        self.global_step = 0
        self.loss_threshold = 1.0

        tf.disable_eager_execution()
        self.net = self._get_net()
      

    def trainer(self,train_path,val_path,output_dir=None,log_dir=None,optimizer='Adam',
                loss_fun='binary_dice_loss',metrics='binary_dice',lr_scheduler=None):
     
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        train_iters = len(train_path) // self.batch_size
        val_iters = len(val_path) // self.batch_size
        train_loader = DataGenerator(train_path,
                                    num_classes=self.num_classes,
                                    channels=self.channels,
                                    input_shape=self.input_shape,
                                    shuffle=True
                                    )
        val_loader = DataGenerator(val_path,
                                  num_classes=self.num_classes,
                                  channels=self.channels,
                                  input_shape=self.input_shape,
                                  shuffle=False
                                  )

        self.ckpt_dir = output_dir
        self.saver = tf.train.Saver(tf.global_variables())
        
        # writer
        self.train_writer = tf.summary.FileWriter(os.path.join(log_dir,'train'), self.sess.graph)
        self.val_writer = tf.summary.FileWriter(os.path.join(log_dir,'val'))  
        
        # loss
        self.loss = self._get_loss(loss_fun)
        self.summary_loss = tf.summary.scalar('loss',self.loss)
        global_step = tf.Variable(0, name="global_step")
        # optimizer
        if lr_scheduler is not None:
            lr = tf.train.exponential_decay(learning_rate=self.lr,
                                            global_step=global_step,
                                            decay_steps=train_iters,
                                            decay_rate=0.95,
                                            staircase=True
                                            )
        else:
            lr = self.lr 

        optimizer = self._get_optimizer(optimizer,lr,global_step)

        # metrics
        self.metrics = self._get_metrics(metrics)
        self.summary_metrics = tf.summary.scalar(metrics,self.metrics)
        
        # merged
        merged = tf.summary.merge([self.summary_loss,self.summary_metrics])

        self.sess.run(tf.global_variables_initializer())

        for epoch in range(self.start_epoch,self.n_epoch):
            # train
            train_total_loss = 0. 
            for step in range(train_iters):
                batch_x, batch_y = train_loader(self.batch_size)

                _,train_loss,summary = self.sess.run([optimizer,self.loss,merged],
                                                        feed_dict={self.net.inputs:batch_x,
                                                                   self.net.labels:batch_y})

                print('epoch:{},step:{},train_loss:{:.5f}'.format(epoch,step,train_loss))

                train_total_loss += train_loss
                self.train_writer.add_summary(summary,self.global_step)
                self.global_step += 1
                if np.mod(self.global_step,150) == 0:
                    self.save()
            # val 
            val_total_loss = 0.   
            for step in range(val_iters):
                val_batch_x, val_batch_y = val_loader(self.batch_size)
                val_loss,summary = self.sess.run([self.loss,merged],
                                                 feed_dict={self.net.inputs:val_batch_x,
                                                            self.net.labels:val_batch_y})
                
                print('epoch:{},step:{},val_loss:{:.5f}'.format(epoch,step,val_loss))
                val_total_loss += val_loss
                self.val_writer.add_summary(summary,self.global_step + step)
            
            print('epoch:{},train_loss:{:.5f},val_loss:{:.5f}'
                .format(epoch,(train_total_loss/train_iters),(val_total_loss/val_iters)))
            
            # save
            if (val_total_loss / val_iters) < self.loss_threshold:
                self.save()
                self.loss_threshold = val_total_loss / val_iters
            
        self.save()


    def _get_net(self):
        if self.net_name == 'unet': 
            from model.unet import unet
            net = unet(channels=self.channels, n_class=self.num_classes, input_shape=self.input_shape)
        return net
    

    def _get_loss(self,loss_fun):
        if loss_fun == 'binary_dice_loss':
            loss = binary_dice_loss(logits=self.net.logits,labels=self.net.labels)

        return loss
    

    def _get_optimizer(self,optimizer,lr,global_step):
        
        if optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss,
                                                                          global_step=global_step
                                                                          )
        return optimizer
    

    def _get_metrics(self,metrics):
        if metrics == 'binary_dice':
            metric = binary_dice(logits=self.net.logits,labels=self.net.labels)
        
        return metric
    

    def save(self):
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        ckpt_path = os.path.join(self.ckpt_dir,self.net_name)
        self.saver.save(self.sess,ckpt_path,global_step=self.global_step)



def binary_dice(logits, labels):
    eps = 1e-8
    logits = tf.nn.softmax(logits, axis=-1) # N,H,W,C
    shape = tf.shape(logits)

    y_pred = tf.reshape(logits[...,1], [shape[0], -1])
    y_true = tf.reshape(labels[...,1], [shape[0], -1])

    inter = tf.reduce_sum(y_pred * y_true, axis=1, keepdims=True)
    union = tf.reduce_sum(y_pred + y_true, axis=1, keepdims=True)

    dice = (2 * inter + eps) / (union + eps)
    return tf.reduce_mean(dice)
    

def binary_dice_loss(logits, labels):
    loss = 1 - binary_dice(logits, labels)
    return loss

