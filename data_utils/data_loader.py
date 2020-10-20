import h5py
import numpy as np


def hdf5_reader(data_path,key=None):
    '''
    Hdf5 file reader, return numpy array.
    '''
    hdf5_file = h5py.File(data_path,'r')
    image = np.asarray(hdf5_file[key],dtype=np.float32)
    hdf5_file.close()

    return image

# NHWC
class DataGenerator(object):
    def __init__(self,path_list,num_classes=2,channels=1,input_shape=(256,256),shuffle=True):
        
        self.path_list = path_list
        self.num_classes = num_classes
        self.channels = channels
        self.shape = input_shape
        self.shuffle = shuffle
        self.index = -1
    
    def _load_data(self):
        image,label = self._next_data()
        image = np.expand_dims(image,axis=-1)
       
        one_hot_label = np.zeros(self.shape + (self.num_classes,),dtype=np.float32)
        
        for z in range(1, self.num_classes):
            temp = (label==z).astype(np.float32)
            one_hot_label[...,z] = temp
            one_hot_label[...,0] = np.amax(one_hot_label[...,1:],axis=-1) == 0  
        return image,one_hot_label
    
    def _cycle_path_list(self):
        self.index += 1
        if self.index >= len(self.path_list):
            self.index = 0
            if self.shuffle:
                np.random.shuffle(self.path_list)

    def _next_data(self):
        self._cycle_path_list()
        image = hdf5_reader(self.path_list[self.index],'image')
        label = hdf5_reader(self.path_list[self.index],'label')
        return image,label
    
    def __call__(self,batch_size):

        images = np.empty((batch_size,) + self.shape + (self.channels,), dtype=np.float32)
        labels = np.empty((batch_size,) + self.shape + (self.num_classes,),dtype=np.float32)
        
        for i in range(batch_size):
            images[i],labels[i] = self._load_data()
        
        return images,labels
