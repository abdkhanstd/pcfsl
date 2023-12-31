# This code is modified from https://github.com/jakesnell/prototypical-networks
# This code is used for prototypical network

import os

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from PIL import Image


class ProtoNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(ProtoNet, self).__init__(model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        z_proto, z_query = self.parse_feature(x, is_feature)

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        scores = self.set_forward(x)


        # Cross-Entropy Loss
        ce_loss = self.loss_fn(scores, y_query)
        reg_lambda=0.01
        # Regularization Loss
        reg_loss = 0.0
        for name, param in self.named_parameters():
            if param.requires_grad:
                reg_loss += torch.norm(param, p=2)  # L2 regularization on all learnable parameters

        total_loss = ce_loss + reg_lambda * reg_loss
        return total_loss


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


if __name__ == '__main__':
    import torchvision.transforms as transforms
    image_path = os.path.join('../filelists\CUB\CUB\images/001.Black_footed_Albatross\Black_Footed_Albatross_0001_796111.jpg')
    img = Image.open(image_path)
    trans = transforms.ToTensor()
    imggg = trans(img)

    xx = np.repeat(range(5), 5)
    print(xx)
    pass
