import abc
import torch
import numpy as np
import torchvision
from torch.utils.data import TensorDataset
from copy import deepcopy
from PIL import Image, ImageFont, ImageDraw

class DataSetBase:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def update(self, rec_X, flag):
        ''' Update x '''
        return

    @abc.abstractmethod
    def construct(self, flag, display):
        ''' Get new dataset based on current x '''
        return

def resettable(f):
    import copy
    def __init_and_copy__(self, *args, **kwargs):
        f(self, *args)
        self.__original_dict__ = copy.deepcopy(self.__dict__['X_'])
        def reset(flag, o = self):
            o.__dict__['X_'][flag] = self.__original_dict__[flag]
        self.reset = reset
    return __init_and_copy__

class DataSet(DataSetBase):
    @resettable
    def __init__(self, data, img_mode, pseudo_inverse_init=False):
        self.data = data
        self.img_mode = img_mode
        train_size, test_size, val_size = data['train'][1].shape, data['test'][1].shape, data['validation'][1].shape
        if pseudo_inverse_init and img_mode is not None:
            init = {
              'train': data['train'][2],
              'test': data['test'][2],
              'validation': data['validation'][2]
            }
        else:
            init = {
              'train': torch.ones( * train_size),
              'test': torch.ones( * test_size),
              'validation': torch.ones( * val_size)
            }
        self.X_ = init

    def update(self, rec_X, flag):
        self.X_[flag] = rec_X

    def construct(self, flag='train', display=True):
        batch_size = 100
        gradient_ = []
        Y, targets = self.data[flag][0], self.data[flag][1]
        import time; start = time.time()
        with torch.no_grad():
            def grad_wrapper(x, y):
                return self.img_mode.grad(x, y)
            gradients = torch.cat([grad_wrapper(chunk_x_, chunk_y)
                                    for (chunk_x_, chunk_y) in
                                    zip(torch.split(self.X_[flag], batch_size),
                                    torch.split(Y, batch_size))])
        if display:
            print('============= {} grad estimated in {:.4f} sec ============= \n'.format(flag, time.time() - start), flush=True)
        return TensorDataset(self.X_[flag], gradients, targets)
