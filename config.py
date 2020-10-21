
__all__ = ['unet']

NET_NAME = 'unet'
VERSION = 'v1.0'
DEVICE = '1'
CURRENT_FOLD = 1


WEIGHT_PATH = {
    'unet':'./ckpt/{}/'.format(VERSION)
}

INIT_TRAINER = {
    'net_name':NET_NAME,
    'lr':1e-3,
    'n_epoch':40,
    'channels':1,
    'num_classes':2,
    'input_shape':(256,256),
    'batch_size':32,
    'is_training':True,
    'weight_path':None
}


SETUP_TRAINER = {
    'output_dir':'./ckpt/{}'.format(VERSION),
    'log_dir':'./log/{}'.format(VERSION),
    'optimizer':'Adam',
    'loss_fun':'binary_dice_loss',
    'metrics':'binary_dice',
    'lr_scheduler':None
}
