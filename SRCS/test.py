import os, tqdm, random, pickle

import torch
import torchvision

from torch.autograd import Variable
from torchvision.transforms import CenterCrop, ToTensor, Compose, Lambda, Resize, Grayscale, Pad
from torchvision.datasets import coco
from torchvision import utils

from torch.nn.functional import binary_cross_entropy, relu, nll_loss, cross_entropy, softmax
from torch.nn import Embedding, Conv2d, Sequential, BatchNorm2d, ReLU
import torch.nn as nn
from torch.optim import Adam


from collections import defaultdict, Counter, OrderedDict

import util, models

from tensorboardX import SummaryWriter

from layers import PlainMaskedConv2d, MaskedConv2d

from dataset import DatasetMaker

import numpy as np


def test(arg):
  arg.batch_size = 1
  OUTCN = 1
  if arg.task == 'T3H':
    _, testloader = DatasetMaker(arg)
    C, H, W = 1, 84, 147
    OH, OW  = 720, 1260
    OUTCN   = 2
    krn = arg.kernel_size
    pad = krn // 2


    encoder = models.ImEncoder(in_size=(H, W), zsize=arg.zsize, depth=arg.vae_depth, colors=C)
    decoder = models.ImDecoder(in_size=(OH, OW), zsize=arg.zsize, depth=arg.vae_depth, out_channels=OUTCN)
    pixcnn  = models.LGated((OUTCN, OH, OW), OUTCN, arg.channels, num_layers=arg.num_layers, k=krn, padding=pad)

    encoder = encoder.cuda()
    decoder = decoder.cuda()
    pixcnn  = pixcnn.cuda()

    encoder.load_state_dict({k.replace('module.',''):v for k,v in torch.load('../DAOU/encoder.pt').items()})
    decoder.load_state_dict({k.replace('module.',''):v for k,v in torch.load('../DAOU/decoder.pt').items()})
    pixcnn.load_state_dict({k.replace('module.',''):v for k,v in torch.load('../DAOU/pixcnn.pt').items()})

    with torch.no_grad():
      for i, (input, target) in enumerate(testloader):
        if torch.cuda.is_available():
          input = input.cuda()
        input = Variable(input)

        zs = encoder(input)
        z = util.sample(*zs)
        out = decoder(z)
        out = pixcnn(out,out)
        out = out.cpu()
        out = out.clone().detach().numpy()
        #out = minmaxNorm(out, 0, 100)
        #target = minmaxNorm(target, 0, 100)
        np.save('../DAOU/pred_%s.npy'%(str(i)), out)
        np.save('../DAOU/ture_%s.npy'%(str(i)), target)
        print(i)

def minmaxNorm(dt, minval, maxval):
  normed_dt = dt * (maxval - minval) + minval
  return normed_dt



if __name__ == "__main__":

    options = util.get_argument_parser()

    print('OPTIONS', options)

    test(options)
