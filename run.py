import os
from trainer import Seg
from config import INIT_TRAINER,SETUP_TRAINER,CURRENT_FOLD,DEVICE
import time
import tensorflow.compat.v1 as tf





def get_cross_validation(path_list,fold_num,current_fold):
  
    _len_ = len(path_list) // fold_num
    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(path_list[start_index:])
        train_id.extend(path_list[:start_index])
    else:
        validation_id.extend(path_list[start_index:end_index])
        train_id.extend(path_list[:start_index])
        train_id.extend(path_list[end_index:])
    
    print(len(train_id),len(validation_id))
    return train_id,validation_id 



def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    data_path = './dataset/train'
    path_list = [os.path.join(data_path,case) for case in os.listdir(data_path)]
    train_path,val_path = get_cross_validation(path_list,5,CURRENT_FOLD)
    
    SETUP_TRAINER['train_path']=train_path
    SETUP_TRAINER['val_path']=val_path
    start_time = time.time()
    with tf.Session(config=config) as sess:
        seg = Seg(sess=sess,**INIT_TRAINER)
        seg.trainer(**SETUP_TRAINER)
    print('run time:%.4f'%(time.time()-start_time))





if __name__ == "__main__":
    tf.app.run()
