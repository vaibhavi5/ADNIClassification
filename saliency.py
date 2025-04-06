import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import nibabel as nib
from models import AlexNet3D_Dropout #vary models here
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, explained_variance_score, mean_squared_error, r2_score
from dataclasses import dataclass
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder

#call : test(dataloader, net, cfg.cuda_avl, cfg.cr)

def test(dataloader, net, cuda_avl, cr):

    net.eval()
    y_pred = np.array([])
    y_true = np.array([])

    # Iterate over dataloader batches
    for _, data in enumerate(dataloader, 0):

        #inputS, inputF, labels = data
        #concat_input = np.float32(np.concatenate((inputS, inputF), axis=0))
        #inputs = torch.from_numpy(concat_input)
        inputs, labels = data

        # Wrap in variable and load batch to gpu
        if cuda_avl:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # Forward pass
        outputs = net(inputs)

        if cr == 'clx':
            _, predicted = torch.max(outputs[0].data, 1)
            y_pred = np.concatenate((y_pred, predicted.cpu().numpy()))
        elif cr == 'reg':
            y_pred = np.concatenate((y_pred, outputs[0].data.cpu().numpy().squeeze()))

        y_true = np.concatenate((y_true, labels.data.cpu().numpy()))

    return y_true, y_pred

def sensitivity_raw(model, im, mask,gradmode, target_class=None, cuda=True, verbose=False, taskmode='clx'):
    # model: pytorch model set to eval()
    # im: 5D image - nSubs x numChannels x X x Y x Z
    # mask: group input data mask - numChannles x X x Y x Z
    # gradmode: 'saliency', 'filtered_saliency', 'imtimes_saliency', 'filtered_imtimes_saliency', 'intermediate'
    # sal_map: gradient [4D image: X X Y X Z X nSubs]
    if cuda:
        im = torch.Tensor(im).cuda()
    else:
        im = torch.Tensor(im)
    im = Variable(im, requires_grad=True)
    # Forward Pass
    output, intermediate = model(im)
    # Predicted labels
    if taskmode == 'clx':
        output = F.softmax(output, dim=1)
        #print(output, output.shape)
    # Backward pass.
    model.zero_grad()
    output_class = output.cpu().max(1)[1].numpy()
    output_class_prob = output.cpu().max(1)[0].detach().numpy()
    if verbose:
        print('Image was classified as', output_class,
              'with probability', output_class_prob)
    # one hot encoding
    one_hot_output = torch.zeros(output.size())
    for i in np.arange(output.size()[0]):
        if target_class is None:
            one_hot_output[i][output_class[i]] = 1
        else:
            one_hot_output[i][target_class[i]] = 1
    if cuda:
        one_hot_output = one_hot_output.cuda()
    # Backward pass
    if 'intermediate' in gradmode:
        output.backward(gradient=one_hot_output, inputs=intermediate)
        # Gradient
        sal_map = intermediate.grad.cpu().numpy() # Remove the subject axis
    else:
        output.backward(gradient=one_hot_output)
        # Gradient
        sal_map = im.grad.cpu().numpy() # Remove the subject axis
    # Gradient
    # sal_map = np.squeeze(im.grad.cpu().numpy(), axis=1) # Removes the channel axis
    # sal_map = im.grad.cpu().numpy().squeeze(axis=0) # Remove the subject axis
    # sal_map = im.grad.cpu().numpy() # Remove the subject axis
    if not 'intermediate' in gradmode:
        if 'imtimes' in gradmode:
            sal_map = sal_map * im.cpu().detach().numpy()
        if 'filtered' in gradmode:
            sal_map = fil_im_5d(sal_map, normalize_method='zscore')
        else:
            # only normalize withouxt filtering
            sal_map = normalize_5D(sal_map, method='zscore')
        # Mask Gradient
        # mask = np.tile(mask[None], (im.shape[0], 1, 1, 1))
        sal_map *= mask
    return sal_map

def normalize_image(X, method='zscore', axis=None):
    if method == 'zscore':
        return zscore(X, axis=axis)
    elif method == 'minmax':
        return minmax(X)
    elif method == None:
        return X

def minmax(X):
    return ((X-X.min())/(X.max()-X.min()))

def normalize_4D(im4D, method='zscore'):
    # 4D normalization (assumes final dimension is subject)
    mim = []
    for i in np.arange(im4D.shape[3]):
        im = im4D[..., i]
        im = normalize_image(im, method=method)
        mim.append(im)
    mim = np.array(mim)
    return mim

def normalize_5D(im5D, method='zscore'):
    mim = np.zeros(im5D.shape)
    for i in range(im5D.shape[0]):
        for j in range(im5D.shape[1]):
            mim[i,j,:,:,:] = normalize_image(im5D[i,j,:,:,:], method=method)
    return mim

def fil_im(smap, normalize_method='zscore'):
    # smap : 5D: nSubs x nCh(1) x X x Y x Z
    s = 2  # sigma gaussian filter
    w = 9  # kernal size gaussian filter
    # truncate gaussian filter at "t#" standard deviations
    t = (((w - 1)/2)-0.5)/s
    fsmap = []
    for i in np.arange(smap.shape[0]):
        im = smap[i]
        im = normalize_image(im, method=normalize_method)
        im = gaussian_filter(im, sigma=s, truncate=t)
        im = normalize_image(im, method=normalize_method)
        fsmap.append(im)
    fsmap = np.array(fsmap)
    return fsmap

def fil_im_5d(smap, normalize_method='zscore'):
    # smap : 5D: nSubs x nCh(1) x X x Y x Z
    s = 2  # sigma gaussian filter
    w = 9  # kernal size gaussian filter
    # truncate gaussian filter at "t#" standard deviations
    t = (((w - 1)/2)-0.5)/s
    fsmap = np.zeros(smap.shape)
    for i in np.arange(smap.shape[0]):
        for j in np.arange(smap.shape[1]):
            im = smap[i,j,:,:,:]
            im = normalize_image(im, method=normalize_method)
            im = gaussian_filter(im, sigma=s, truncate=t)
            im = normalize_image(im, method=normalize_method)
            fsmap[i,j,:,:,:] = im
    return fsmap
