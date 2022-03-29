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

SEEDFRAC = 2

def draw_sample(seeds, decoder, pixcnn, zs, seedsize=(0, 0)):

    b, c, h, w = seeds.size()

    sample = seeds.clone()
    if torch.cuda.is_available():
        sample, zs = sample.cuda(), zs.cuda()
    sample, zs = Variable(sample), Variable(zs)

    cond = decoder(zs)

    for i in tqdm.trange(h):
        for j in range(w):

            if i < seedsize[0] and j < seedsize[1]:
                continue

            for channel in range(c):

                result = pixcnn(sample, cond)
                probs = softmax(result[:, :, channel, i, j]).data

                pixel_sample = torch.multinomial(probs, 1).float() / 255.
                sample[:, channel, i, j] = pixel_sample.squeeze()

    return sample

class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=100, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, model2, model3, epoch):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model2, model3, epoch)
        elif score >= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model2, model3, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model2, model3, epoch):

        '''Saves model when validation loss decrease.'''

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.7f} --> {val_loss:.7f}).  Saving model')
        state = {'epoch': epoch, 'model': model}
        torch.save(model.state_dict(), '../DAOU/encoder.pt')
        torch.save(model2.state_dict(),'../DAOU/decoder.pt')
        torch.save(model3.state_dict(),'../DAOU/pixcnn.pt')
        self.val_loss_min = val_loss


def go(arg):

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    early_stopping = EarlyStopping(patience = arg.patience_num, verbose=True)
    ## Load the data
    if arg.task == 'T3H' or arg.task == 'REH':
        trainloader, testloader = DatasetMaker(arg)
        C, H, W = 1, 84, 147
        OH, OW  = 720, 1260

    else:
        raise Exception('Task {} not recognized.'.format(arg.task))

    ## Set up the model
    fm = arg.channels
    krn = arg.kernel_size
    pad = krn // 2

    OUTCN = 2

    if arg.model == 'vae-up':
        """
        Upsampling model. VAE with an encoder and a decoder, generates a conditional vector at every pixel,
        which is then passed to the picelCNN layers.
        """

        encoder = models.ImEncoder(in_size=(H, W), zsize=arg.zsize, depth=arg.vae_depth, colors=C)
        decoder = models.ImDecoder(in_size=(OH, OW), zsize=arg.zsize, depth=arg.vae_depth, out_channels=OUTCN)
        pixcnn  = models.LGated((OUTCN, OH, OW), OUTCN, arg.channels, num_layers=arg.num_layers, k=krn, padding=pad)

        mods = [encoder, decoder, pixcnn]

    elif arg.model == 'vae-straight':
        """
        Model that generates a single latent code for the whole image, and passes it straight to the autoregressive 
        decoder: no upsampling layers or deconvolutions.
        """

        encoder = models.ImEncoder(in_size=(H, W), zsize=arg.zsize, depth=arg.vae_depth, colors=C)
        decoder = util.Lambda(lambda x : x) # identity
        pixcnn  = models.CGated((C, H, W), (arg.zsize,), arg.channels, num_layers=arg.num_layers, k=krn, padding=pad)

        mods = [encoder, decoder, pixcnn]

    else:
        raise Exception('model "{}" not recognized'.format(arg.model))

    if torch.cuda.is_available():
        for m in mods:
            m.cuda()

    print('Constructed network', encoder, decoder, pixcnn)

    #
    sample_zs = torch.randn(12, arg.zsize)
    sample_zs = sample_zs.unsqueeze(1).expand(12, 6, -1).contiguous().view(72, 1, -1).squeeze(1)

    # A sample of 144 square images with 3 channels, of the chosen resolution
    # (144 so we can arrange them in a 12 by 12 grid)
    sample_init_zeros = torch.zeros(72, C, H, W)
    sample_init_seeds = torch.zeros(72, C, H, W)

    sh, sw = H//SEEDFRAC, W//SEEDFRAC

    # Init second half of sample with patches from test set, to seed the sampling
    testbatch = util.readn(testloader, n=12)
    testbatch = testbatch.unsqueeze(1).expand(12, 6, C, H, W).contiguous().view(72, 1, C, H, W).squeeze(1)
    sample_init_seeds[:, :, :sh, :] = testbatch[:, :, :sh, :]

    params = []
    for m in mods:
        params.extend(m.parameters())
    optimizer = Adam(params, lr=arg.lr)

    instances_seen = 0
    criterion = nn.L1Loss(size_average=True)
    for epoch in range(arg.epochs):

        # Train
        err_tr = []

        for m in mods:
            m.train(True)

        for i, (input, target) in enumerate(tqdm.tqdm(trainloader)):
            if arg.limit is not None and i * arg.batch_size > arg.limit:
                break

            # Prepare the input
            b, c, w, h = input.size()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.type(torch.LongTensor).cuda()

            #target = (input.data * 255).long()

            input, target2 = Variable(input), Variable(target)

            # Forward pass
            zs = encoder(input)
            kl_loss = util.kl_loss(*zs)
            z = util.sample(*zs)
            out = decoder(z)
            out = pixcnn(out ,out)
            #rec = pixcnn(out, target2)

            #rec_loss = cross_entropy(out, target, reduce=False).view(b, -1).sum(dim=1)
            #rec_loss = rec_loss * util.LOG2E  # Convert from nats to bits

            rec_loss = criterion(out, target2)

            loss = (rec_loss + kl_loss).mean()

            instances_seen += input.size(0)
            tbw.add_scalar('pixel-models/vae/training/kl-loss',  kl_loss.mean().data.item(), instances_seen)
            tbw.add_scalar('pixel-models/vae/training/rec-loss', rec_loss.mean().data.item(), instances_seen)

            err_tr.append(loss.data.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if epoch % arg.eval_every == 0 and epoch != 0:
            with torch.no_grad():

                # Evaluate
                # - we evaluate on the test set, since this is only a simple reproduction experiment
                #   make sure to split off a validation set if you want to tune hyperparameters for something important

                err_te = []

                for m in mods:
                    m.train(False)

                if not arg.skip_test:

                    for i, (input, target) in enumerate(tqdm.tqdm(testloader)):
                        if arg.limit is not None and i * arg.batch_size > arg.limit:
                            break
                        b, c, w, h = input.size()

                        if torch.cuda.is_available():
                            input = input.cuda()
                            target = target.type(torch.LongTensor).cuda()
                        #target = (input.data * 255).long()
                        input, target = Variable(input), Variable(target)

                        zs = encoder(input)
                        kl_loss = util.kl_loss(*zs)
                        z = util.sample(*zs)
                        out = decoder(z)

                        #rec = pixcnn(input, out)

                        #rec_loss = cross_entropy(rec, target, reduce=False).view(b, -1).sum(dim=1)
                        #rec_loss_bits = rec_loss * util.LOG2E  # Convert from nats to bits
                        rec_loss = criterion(out, target)
                        loss = (rec_loss + kl_loss).mean()
                        err_te.append(loss.data.item())
                        val_loss = (sum(err_te)/len(err_te))**2
                    early_stopping(val_loss, encoder, decoder, pixcnn, epoch)
                    tbw.add_scalar('pixel-models/test-loss', sum(err_te)/len(err_te), epoch)
                    print('epoch={:02}; training loss: {:.7f}; test loss: {:.7f}'.format(
                        epoch, sum(err_tr)/len(err_tr), sum(err_te)/len(err_te)))
                if early_stopping.early_stop:
                    print("Early stop")
                    break

                for m in mods:
                    m.train(False)

                #sample_zeros = draw_sample(sample_init_zeros, decoder, pixcnn, sample_zs, seedsize=(0, 0))
                #sample_seeds = draw_sample(sample_init_seeds, decoder, pixcnn, sample_zs, seedsize=(sh, W))
                #sample = torch.cat([sample_zeros, sample_seeds], dim=0)

                #utils.save_image(sample, 'sample_{:02d}.png'.format(epoch), nrow=12, padding=0)

if __name__ == "__main__":

    options = util.get_argument_parser()

    print('OPTIONS', options)

    go(options)
