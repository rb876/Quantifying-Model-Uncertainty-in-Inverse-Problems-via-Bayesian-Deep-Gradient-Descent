import odl
import abc
import numpy as np
import torch
from odl.contrib import torch as odl_torch

class ImageModalityBase:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def sinogram(self, phantom):
        ''' Compute sinogram and pseudoinverse '''
        return

    @abc.abstractmethod
    def grad(self, x, y):
        ''' Compute grad of the data fitting term '''
        return

class SimpleCT(ImageModalityBase):
    def __init__(self, forward_model):
        self.forward_model = forward_model

    def sinogram(self, phantom):
        clean = self.forward_model.operator(phantom)
        noise = []
        for el in clean:
            scale_factor = torch.mean(torch.abs(el))
            noise.append(clean.data.new(el.size()).normal_(0, 1) * scale_factor * 0.01)
        noisy = clean + torch.stack(noise)
        fbp = self.forward_model.pseudoinverse(noisy)
        return noisy, phantom, fbp

    def grad(self, x, y):
        return self.forward_model.adjoint(self.forward_model.operator(x) - y)

class ForwardModel:
    def __init__(self):
        self.space = None
        self.geometry = None
        self.operator = None
        self.adjoint = None
        self.pseudoinverse = None

    @property
    def space(self):
        return self.__space
    @space.setter
    def space(self, space):
        self.__space = space

    @property
    def geometry(self):
        return self.__geometry
    @geometry.setter
    def geometry(self, geometry):
        self.__geometry = geometry

    @property
    def operator(self):
        return self.__operator
    @operator.setter
    def operator(self, operator):
        self.__operator = operator

    @property
    def adjoint(self):
        return self.__adjoint
    @adjoint.setter
    def adjoint(self, adjoint):
        self.__adjoint = adjoint

    @property
    def pseudoinverse(self):
        return self.__pseudoinverse
    @pseudoinverse.setter
    def pseudoinverse(self, pseudoinverse):
        self.__pseudoinverse = pseudoinverse
