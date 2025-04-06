import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import pearsonr, zscore
from models import AlexNet3D_Dropout #vary models here
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, explained_variance_score, mean_squared_error, r2_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from dataclasses import dataclass
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


@dataclass
class Config:
    iter: int = 0  # slurmTaskIDMapper maps this variable using tr_smp_sizes and nReps to tss and rep
    tr_smp_sizes: tuple = (100, 200, 500, 1000, 2000, 5000, 10000)
    nReps: int = 10
    nc: int = 10
    bs: int = 16
    lr: float = 0.001
    es: int = 1
    pp: int = 1
    es_va: int = 1
    es_pat: int = 40
    ml: str = '../../temper/'
    mt: str = 'AlexNet3D_Dropout'
    ssd: str = '../../SampleSplits_multimodal/'
    scorename: str = 'ass'
    cuda_avl: bool = True
    nw: int = 8
    cr: str = 'reg'
    k: int = 0
    tss: int = 100  # modification automated via slurmTaskIDMapper
    rep: int = 0  # modification automated via slurmTaskIDMapper
    seed: int = 1

class MRIDataset(Dataset):

    def __init__(self, cfg, mode):
        self.df = readFrames(cfg.ssd, mode, cfg.tss, cfg.rep)
        self.scorename = cfg.scorename
        self.cr = cfg.cr
        self.k = cfg.k

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        X, y = read_X_y_5D_idx(self.df, idx, self.scorename, self.cr, self.k)
        return [X, y]


def readFrames(ssd, mode, tss, rep):

    # Read Data Frame
    df = pd.read_csv(ssd + mode + '_' + str(tss) +
                     '_rep_' + str(rep) + '.csv')

    print('Mode ' + mode + ' :' + 'Size : ' +
          str(df.shape) + ' : DataFrames Read ...')

    return df


def read_X_y_5D_idx(df, idx, scorename, cr, k):

    Xs, Xf, y, all_measures = [], [], [], []

    # Read image
    relative_path_lowres_sMRI = '/data/users2/ibatta/data/features/lowresSMRI/ADNI/nii/' #'/data/qneuromark/Data/ADNI/Updated/T1/ADNI/'
    rel_path_MIP = '/data/users2/vitkyal/projects/complex/allMIPs/'
    sMRI_file_name = 'smwc1T1.nii.gz'
    #mask_path = '/data/users2/vitkyal/projects/SMLvsDL/reprex/saliency_analysis/average_smrimask.nii'
    #mask_path = '/data/users2/vitkyal/projects/complex/binary_mask_missingbrain.nii'
    fNs = relative_path_lowres_sMRI + df['lowres_smriPath'].iloc[idx] + sMRI_file_name
    url = df['lowres_fmriPath'].iloc[idx]
    session = ('{}'.format(*url.split('/')[3:]))
    subj_id = ('{}'.format(*url.split('/')[0:]))
    fNf = rel_path_MIP + 'minMIP_' + subj_id + '_' + session + '.nii'
    #fNfm = rel_path_mIP + 'minMIP_' + subj_id + '_' + session + '.nii'
    s = torch.tensor(nib.load(fNs).get_fdata(), dtype=None, device=None, requires_grad=False)
    f = torch.tensor(nib.load(fNf).get_fdata(), dtype=None, device=None, requires_grad=False)
    #fm = torch.tensor(nib.load(fNfm).get_fdata(), dtype=None, device=None, requires_grad=False)

    #/data/users2/ibatta/data/features/lowresSMRI/ADNI/nii/011_S_6367/Accelerated_Sagittal_MPRAGE/2018-05-16_12_17_49.0/S686265/anat/
    
    Xc1 = torch.zeros_like(s)
    Xc2 = torch.zeros_like(s)
    Xc3 = torch.zeros_like(s)
    Xc4 = torch.zeros_like(s)
    Xc5 = torch.zeros_like(s)
    Xc6 = torch.zeros_like(s)
    Xc7 = torch.zeros_like(s)
    Xc8 = torch.zeros_like(s)
    s_mask = torch.zeros_like(s)
    f_mask = torch.zeros_like(s)
    fm_mask = torch.zeros_like(s)
    maskd = (s > 0.03)
    #maskd = torch.tensor(nib.load(mask_path).get_fdata(), dtype=torch.bool)
    s_mask[maskd] = s[maskd]
    f_mask[maskd] = f[maskd]
    #fm_mask[maskd] = fm[maskd]
    s = (s_mask - s_mask.min()) / (s_mask.max() - s_mask.min())
    f = (f_mask - f_mask.min()) / (f_mask.max() - f_mask.min())
    #fm = (fm_mask - fm_mask.min()) / (fm_mask.max() - fm_mask.min())

    #Xc1[maskd] = f[maskd] / s[maskd]
    #Xc2[maskd] = np.sqrt(s[maskd]*s[maskd] + f[maskd]*f[maskd])
    #Xc3[maskd] = f[maskd] * s[maskd]
    #Xc4[maskd] = fm[maskd] / s[maskd]
    #Xc5[maskd] = np.sqrt(s[maskd]*s[maskd] + fm[maskd]*fm[maskd])
    #Xc6[maskd] = fm[maskd] * s[maskd]
    #Xc7[maskd] = s[maskd] * s[maskd] 
    #Xc8[maskd] = f[maskd] * f[maskd] 
    #Try Xc3, Xc7 and Xc8

    #Xc1 = (Xc1 - Xc1.min()) / (Xc1.max() - Xc1.min()) #division
    #Xc2 = (Xc2 - Xc2.min()) / (Xc2.max() - Xc2.min()) #amplitude
    #Xc3 = (Xc3 - Xc3.min()) / (Xc3.max() - Xc3.min()) #multiplication
    #Xc4 = (Xc4 - Xc4.min()) / (Xc4.max() - Xc4.min()) #division
    #Xc5 = (Xc5 - Xc5.min()) / (Xc5.max() - Xc5.min()) #amplitude
    #Xc6 = (Xc6 - Xc6.min()) / (Xc6.max() - Xc6.min()) #multiplication
    #Xc7 = (Xc7 - Xc7.min()) / (Xc7.max() - Xc7.min())
    #Xc8 = (Xc8 - Xc8.min()) / (Xc8.max() - Xc8.min())
    Xc1 = s
    Xc2 = f
    #Xc3 = fm
    
    Xc1 = np.reshape(Xc1, (1, Xc1.shape[0], Xc1.shape[1], Xc1.shape[2]))
    Xc2 = np.reshape(Xc2, (1, Xc2.shape[0], Xc2.shape[1], Xc2.shape[2]))
    #Xc3 = np.reshape(Xc3, (1, Xc3.shape[0], Xc3.shape[1], Xc3.shape[2]))
    #Xc4 = np.reshape(Xc4, (1, Xc4.shape[0], Xc4.shape[1], Xc4.shape[2]))
    #Xc5 = np.reshape(Xc5, (1, Xc5.shape[0], Xc5.shape[1], Xc5.shape[2]))
    #Xc6 = np.reshape(Xc6, (1, Xc6.shape[0], Xc6.shape[1], Xc6.shape[2]))
    #Xc7 = np.reshape(Xc7, (1, Xc7.shape[0], Xc7.shape[1], Xc7.shape[2]))
    #Xc8 = np.reshape(Xc8, (1, Xc8.shape[0], Xc8.shape[1], Xc8.shape[2]))

    #X = np.float32(np.concatenate((Xc1, Xc2), axis=0)) #concat in dim 1 #da ma dm
    X = np.float32(Xc2)  #THIS CHANGES IF ITS UNIMODAL OR MULTIMODAL

    # Read label
    labelencoder = LabelEncoder()
    df['ResearchGroup'] = labelencoder.fit_transform(df['ResearchGroup'])
    df['ResearchGroup'] += 1
    y = int(df[scorename].iloc[idx])

    if scorename == 'ResearchGroup':
        y -= 1

    if cr == 'reg':
        y = np.array(np.float32(y))
    elif cr == 'clx':
        y = np.array(y)

    return X, y


'''def addPadding(input_tensor):
    desired_size = (64, 64, 64)
    current_size = input_tensor.shape[-3:]  
    
    resized_image = F.interpolate(input_tensor, size=desired_size, mode='trilinear', align_corners=False)
    
    return resized_image

def addPadding(input_tensor): #non interpolate function
    desired_size = (64, 64, 64)
    current_size = input_tensor.shape[-3:]
    padding = []
    for i in range(len(current_size)):
        diff = desired_size[i] - current_size[i]
        pad_before = diff // 2
        pad_after = diff - pad_before
        padding.extend([pad_before, pad_after])

    padding.extend([0] * (len(input_tensor.shape) - len(current_size) - 2))

    padded_size = input_tensor.shape[:-3] + tuple(desired_size)
    padded_image = F.pad(input_tensor, pad=padding[::-1], mode='constant', value=0)
    return padded_image'''

def train(dataloader, net, optimizer, criterion, cuda_avl):

    net.train()

    # Iterate over dataloader batches
    for _, data in enumerate(dataloader, 0):

        # Fetch the inputs
        inputs, labels = data
        #inputs = addPadding(input)

        #inputS, inputF, labels = data
        #concat_input = np.float32(np.concatenate((inputS, inputF), axis=0))
        #inputs = torch.from_numpy(concat_input)

        # Wrap in variable and load batch to gpu
        if cuda_avl:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs[0].squeeze(), labels)
        loss.backward()
        optimizer.step()

    return loss


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
        #inputs = addPadding(input)

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


def evalMetrics(dataloader, net, cfg):

    all_measures = ['swarest1_ALFF.nii', 'swarest1_KccReHo.nii', 'swarest1_DegreeCentrality_DegreeCentrality_PositiveBinarizedSumBrain.nii','swarest1_PerAF.nii', 'swarest1_DegreeCentrality_DegreeCentrality_PositiveWeightedSumBrain.nii', 'swarest1_VMHC.nii', 'swarest1_fALFF.nii', 'smwc1T1.nii.gz']
    #print(all_measures[cfg.k])
    # Batch Dataloader
    y_true, y_pred = test(dataloader, net, cfg.cuda_avl, cfg.cr)
    y_t = []
    y_p = []

    net.eval()

    # Iterate over dataloader batches
    for _, data in enumerate(dataloader, 0):

        inputs, labels = data
        #inputs = addPadding(input)

        if cfg.cuda_avl:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # Forward pass
        outputs = net(inputs)
        predictions = F.softmax(outputs[0], dim=1)

        y_t.extend(labels.cpu().numpy())
        y_p.extend(predictions.cpu().detach().numpy()[:, 1])  # Probability of class 1

    y_t = np.array(y_t)
    y_p = np.array(y_p)

    if cfg.cr == 'clx':

        # Evaluate classification performance
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true,y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
        #precision = 0
        recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
        roc_auc = 0 #= roc_auc_score(y_t, y_p, multi_class='ovr')

        return acc, bal_acc, conf_matrix, f1, precision, recall, roc_auc, class_report

    elif cfg.cr == 'reg':

        # Evaluate regression performance
        mae = mean_absolute_error(y_true, y_pred)
        ev = explained_variance_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        r, p = pearsonr(y_true, y_pred)

        return mae, ev, mse, r2, r, p

    else:
        print('Check cr flag')


def generate_validation_model(cfg):

    # Initialize net based on model type (mt, nc)

    net = initializeNet(cfg)

    # Training parameters
    epochs_no_improve = 0
    valid_acc = 0

    if cfg.cr == 'clx':
        criterion = nn.CrossEntropyLoss()
        reduce_on = 'max'
        m_val_acc = 0
        history = pd.DataFrame(columns=['k', 'scorename', 'iter', 'epoch',
                                        'tr_acc', 'tr_bal_acc', 'tr_conf_matrix', 'tr_f1', 'tr_precision', 'tr_recall', 'tr_roc_auc', 'tr_class_report', 'va_acc', 'va_bal_acc', 'va_conf_matrix', 'va_f1', 'va_precision', 'va_recall', 'va_roc_auc', 'loss', 'va_class_report'])

    elif cfg.cr == 'reg':
        criterion = nn.MSELoss()
        reduce_on = 'min'
        m_val_acc = 100
        history = pd.DataFrame(columns=['k', 'scorename', 'iter', 'epoch', 'tr_mae', 'tr_ev', 'tr_mse',
                                        'tr_r2', 'tr_r', 'tr_p', 'val_mae', 'val_ev', 'val_mse', 'val_r2', 'val_r', 'val_p', 'loss'])
    else:
        print('Check config flag cr')

    # Load model to gpu
    if cfg.cuda_avl:
        criterion.cuda()
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # Declare optimizer
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)

    # Declare learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode=reduce_on, factor=0.5, patience=7, verbose=True)

    # Batch Dataloader
    trainloader = loadData(cfg, 'tr')
    validloader = loadData(cfg, 'va')

    for epoch in range(cfg.es):

        # Train
        print('Training: ')
        loss = train(trainloader, net, optimizer, criterion, cfg.cuda_avl)
        loss = loss.data.cpu().numpy()

        if cfg.cr == 'clx':

            print('Validating: ')

            # Evaluate classification perfromance on training and validation data
            tr_acc, tr_bal_acc, tr_conf_matrix, tr_f1, tr_precision, tr_recall, tr_roc_auc, tr_class_report = evalMetrics(trainloader, net, cfg)
            va_acc, va_bal_acc, va_conf_matrix, va_f1, va_precision, va_recall, va_roc_auc, va_class_report = evalMetrics(validloader, net, cfg)

            # Log Performance
            history.loc[epoch] = [cfg.k, cfg.scorename, cfg.iter, epoch, tr_acc, tr_bal_acc, tr_conf_matrix, tr_f1, tr_precision, tr_recall, tr_roc_auc, tr_class_report, va_acc, va_bal_acc, va_conf_matrix, va_f1, va_precision, va_recall, va_roc_auc, loss, va_class_report]

            # Check for maxima (e.g. accuracy for classification)
            isBest = va_acc > m_val_acc

        elif cfg.cr == 'reg':

            print('Validating: ')

            # Evaluate regression perfromance on training and validation data
            train_mae, train_ev, train_mse, train_r2, train_r, train_p = evalMetrics(
                trainloader, net, cfg)
            valid_acc, valid_ev, valid_mse, valid_r2, valid_r, valid_p = evalMetrics(
                validloader, net, cfg)

            # Log Performance
            history.loc[epoch] = [cfg.k, cfg.scorename, cfg.iter, epoch, train_mae, train_ev, train_mse, train_r2,
                                  train_r, train_p, valid_acc, valid_ev, valid_mse, valid_r2, valid_r, valid_p, loss]

            # Check for minima (e.g. mae for regression)
            isBest = valid_acc < m_val_acc

        else:
            print('Check cr flag')

        # Write Log
        history.to_csv(cfg.ml + 'history' + str(cfg.k) + '.csv', index=False)

        # Early Stopping
        if cfg.es_va:

            # If minima/maxima
            if isBest:

                # Save the model
                torch.save(net.state_dict(), open(
                    cfg.ml + 'model_state_dict.pt', 'wb'))

                # Reset counter for patience
                epochs_no_improve = 0
                m_val_acc = valid_acc

            else:

                # Update counter for patience
                epochs_no_improve += 1

                # Check early stopping condition
                if epochs_no_improve == cfg.es_pat:

                    print('Early stopping!')

                    # Stop training: Return to main
                    return history, m_val_acc

        else:
            print('build loss or other cases')

        # Decay Learning Rate
        scheduler.step(valid_acc)


def evaluate_test_accuracy(cfg):

    # Load validated net
    net = loadNet(cfg)
    net.eval()

    # Dataloader
    cfg.bs = 1
    testloader = loadData(cfg, 'te')

    if cfg.cr == 'clx':

        # Initialize Log File
        outs = pd.DataFrame(columns=['k', 'iter', 'te_acc', 'te_bal_acc', 'te_conf_matrix', 'te_f1', 'te_precision', 'te_recall', 'te_roc_auc', 'te_class_report'])

        print('Testing: ')

        # Evaluate classification performance
        te_acc, te_bal_acc, te_conf_matrix, te_f1, te_precision, te_recall, te_roc_auc, te_class_report = evalMetrics(testloader, net, cfg)

        # Log Performance

        outs.loc[0] = [cfg.k, cfg.iter, te_acc, te_bal_acc, te_conf_matrix, te_f1, te_precision, te_recall, te_roc_auc, te_class_report]

    elif cfg.cr == 'reg':

        # Initialize Log File
        outs = pd.DataFrame(columns=[
                            'k', 'iter', 'mae_te', 'ev_te', 'mse_te', 'r2_te', 'r_te', 'p_te'])

        print('Testing: ')

        # Evaluate regression performance
        mae, ev, mse, r2, r, p = evalMetrics(testloader, net, cfg)

        # Log Performance

        outs.loc[0] = [cfg.k, cfg.iter, mae, ev, mse, r2, r, p]

    else:
        print('Check cr mode')

    # Write Log
    outs.to_csv(cfg.ml+'test' + str(cfg.k) + '.csv', index=False)


def loadData(cfg, mode):

    # Batch Dataloader
    prefetch_factor = 8 # doesn't seem to be working; tried 1, 2, 4, 8, 16, 32 - mem used stays the same! need to verify the MRIdataset custom functionality maybe
    dset = MRIDataset(cfg, mode)

    dloader = DataLoader(dset, batch_size=cfg.bs,
                         shuffle=True, num_workers=cfg.nw, drop_last=False, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True) #change: drop_last=False

    return dloader


def loadNet(cfg):

    # Load validated model
    net = initializeNet(cfg)
    model = torch.nn.DataParallel(net)
    net = 0
    net = load_net_weights2(model, cfg.ml+'model_state_dict.pt')

    return net


def updateIterML(cfg):

    # Update Iter (in case of multitask training)
    if cfg.pp:
        cfg.iter += 1

    # Map slurmTaskID to training sample size (tss) and CV rep (rep)
    cfg = slurmTaskIDMapper(cfg)

    # Update Model Location
    cfg.ml = cfg.ml+cfg.mt+'_scorename_'+cfg.scorename+'_iter_' + \
        str(cfg.iter)+'_tss_'+str(cfg.tss)+'_rep_'+str(cfg.rep)+'_bs_'+str(cfg.bs)+'_lr_' + \
        str(cfg.lr)+'_espat_'+str(cfg.es_pat)+'/'

    # Make Model Directory
    try:
        os.stat(cfg.ml)
    except:
        os.mkdir(cfg.ml)

    return cfg


def slurmTaskIDMapper(cfg):

    # Map iter value (slurm taskID) to training sample size (tss) and crossvalidation repetition (rep)
    tv, rv = np.meshgrid(cfg.tr_smp_sizes, np.arange(cfg.nReps))
    tv = tv.reshape((1, np.prod(tv.shape)))
    rv = rv.reshape((1, np.prod(tv.shape)))
    tss = tv[0][cfg.iter]
    rep = rv[0][cfg.iter]
    print(tss, rep)
    cfg.tss = tss
    cfg.rep = rep
    print(cfg.iter, cfg.tss, cfg.rep)

    return cfg


def initializeNet(cfg):

    # Initialize net based on model type (mt, nc)
    if cfg.mt == 'AlexNet3D_Dropout':
        net = AlexNet3D_Dropout(num_classes=cfg.nc)
    else:
        print('Check model type')

    return net


def load_net_weights2(net, weights_filename):

    # Load trained model
    state_dict = torch.load(
        weights_filename,  map_location=lambda storage, loc: storage)
    state = net.state_dict()
    state.update(state_dict)
    net.load_state_dict(state)

    return net

def sensitivity_raw(model, gradmode, im, mask, target_class=None, taskmode='clx', cuda_avl=True, verbose=False):
    # model: pytorch model set to eval()
    # im: 5D image - nSubs x numChannels x X x Y x Z
    # mask: group input data mask - numChannles x X x Y x Z
    # gradmode: 'saliency', 'filtered_saliency', 'imtimes_saliency', 'filtered_imtimes_saliency', 'intermediate'
    # sal_map: gradient [4D image: X X Y X Z X nSubs]
    if cuda_avl:
        im = torch.Tensor(im).cuda()
    else:
        im = torch.Tensor(im)
    im = Variable(im, requires_grad=True)

    
    torch.save(im, 'salfunction_1.pt')
    
    # Forward Pass
    output, intermediate = model(im)
    # Predicted labels
    if taskmode == 'clx':
        output = F.softmax(output, dim=1)
        #print(output, output.shape)
    torch.save(output, 'salfunction_2.pt')
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
    if cuda_avl:
        one_hot_output = Variable(one_hot_output.cuda())
    print(one_hot_output)
    # Backward pass
    #if 'intermediate' in gradmode:
        #output.backward(gradient=one_hot_output, inputs=intermediate)
        # Gradient
        #sa_map = intermediate.grad.cpu().numpy() # Remove the subject axis
        #sal_map = np.abs(sa_map)
        #continue
    #else:
    print(output)
    print(output.shape)
    output.backward(gradient=one_hot_output)
    #torch.save(output, 'salfunction_3.pt')
    sal_map = im.grad.cpu().numpy() # Remove the subject axis
    #torch.save(sa_map, 'salfunction_4.pt')
    #sal_map = np.abs(sa_map) #uncomment if you wanna take the abs
    #torch.save(sal_map, 'salfunction_5.pt')
    # Gradient
    # sal_map = np.squeeze(im.grad.cpu().numpy(), axis=1) # Removes the channel axis
    # sal_map = im.grad.cpu().numpy().squeeze(axis=0) # Remove the subject axis
    # sal_map = im.grad.cpu().numpy() # Remove the subject axis
    '''
    if not 'intermediate' in gradmode:
        if 'imtimes' in gradmode:
            #sal_map = sal_map * im.cpu().detach().numpy()
            sal_map *= mask
        if 'filtered' in gradmode:
            sal_map *= mask
            #sal_map = fil_im_5d(sal_map, normalize_method='zscore')
        else:
            # only normalize withouxt filtering
            #sal_map = normalize_5D(sal_map, method='zscore')
        # Mask Gradient
        # mask = np.tile(mask[None], (im.shape[0], 1, 1, 1))
            sal_map *= mask
    ''' 
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

def loadMasks(cfg, mode):
    relative_path_lowres_sMRI = '/data/users2/ibatta/data/features/lowresSMRI/ADNI/nii/'
    sMRI_file_name = 'smwc1T1.nii.gz'
    df = readFrames(cfg.ssd, mode, cfg.tss, cfg.rep)
    idx = 0
    fNs = relative_path_lowres_sMRI + df['lowres_smriPath'].iloc[idx] + sMRI_file_name
    s = nib.load(fNs).get_fdata()  

    masks = (s > 0.03)
    return masks

def test_saliency(cfg):
    cfg.bs = 1 #otherwise it skips the last batch
    for md in ['te']:#,'va','te']:
        
        print('Testing saliency / intermediate feature extraction..')
        cuda_avl = cfg.cuda_avl
        
        net = loadNet(cfg)
        #net = AlexNet3D_Dropout(2)
    
        if cfg.cuda_avl:
            net.load_state_dict(torch.load(cfg.ml+'model_state_dict.pt'))
            net.cuda()
            net = torch.nn.DataParallel(
                net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True
        else:
            net.load_state_dict(torch.load(cfg.ml+'model_state_dict.pt', map_location=torch.device('cpu')))
        net.eval()
        
        isal = []
        #sal , fsal, imsal, fimsal = [],[],[],[]
        dataloader = loadData(cfg, md)#, get_shapes=False)
        # Iterate over dataloader batches
        all_labels = []
        
        
        for i, data in enumerate(dataloader, 0):
            # print('Running new batch..')
            inputs, labels = data
            #inputs = addPadding(input)
            mask_path = '/data/users2/vitkyal/projects/SMLvsDL/reprex/saliency_analysis/average_smrimask.nii'#loadMasks(cfg, md)
            #mask_path = '/data/users2/vitkyal/projects/complex/binary_mask.nii'
            #maskfile = nib.load(mask_path)
            #save inputs as a nibabel file. Something like "inputSal_"+str(i)
            #save_file_name = 'inputSal_' + str(i) + '.pt'
            #torch.save(inputs[0][0], save_file_name)
            #nib.save(nib.Nifti1Image(inputs[0][0], affine=maskfile.affine, header= maskfile.header), save_file_name)

            all_labels.append(np.squeeze(labels))
            
            # # Wrap in variable and load batch to gpu
            # if cuda_avl:
            #     inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            # else:
            #     inputs, labels = Variable(inputs), Variable(labels)
            # tempmask = np.ones(inputs.shape[2:])
            masks = nib.load(mask_path).get_fdata() 
            sal_out = sensitivity_raw(net, 'BPraw', inputs, masks, taskmode=cfg.cr, cuda_avl=cuda_avl)
            # import pdb; pdb.set_trace()
            isal.append(sal_out)
            #continue
            
        all_labels = np.hstack(all_labels).squeeze()
        # np.savetxt(cfg.ml+'labels_%s.txt'%md, all_labels, fmt='%d')
        
        print('Saving saliency results to: \n'+cfg.ml)
        

        #import pdb; pdb.set_trace() #this is a debugger
        isal = np.vstack(isal)
        #np.savetxt(cfg.ml+'isal_%s.csv'%md, isal, delimiter=',')
        with open(cfg.ml+'nonabs_sal.pkl','wb') as f:
            pickle.dump(isal,f)

def CaptumSal(cfg):
    cfg.bs = 1 #otherwise it skips the last batch
    for md in ['te']:#,'va','te']:
        
        print('Testing saliency / intermediate feature extraction..')
        cuda_avl = cfg.cuda_avl
        
        net = loadNet(cfg)
        #net = AlexNet3D_Dropout(2)
        if cfg.cuda_avl:
            #net.load_state_dict(torch.load(cfg.ml+'model_state_dict.pt'))
            net.cuda()
            net = torch.nn.DataParallel(
                net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True
        else:
            #net.load_state_dict(torch.load(cfg.ml+'model_state_dict.pt', map_location=torch.device('cpu')))
            pass
        net.eval()
        
        isal = []
        #sal , fsal, imsal, fimsal = [],[],[],[]
        dataloader = loadData(cfg, md)#, get_shapes=False)
        # Iterate over dataloader batches
        all_labels = []
        
        
        for i, data in enumerate(dataloader, 0):
            # print('Running new batch..')
            input, labels = data
            #inputs = addPadding(input)
            mask_path = '/data/users2/vitkyal/projects/SMLvsDL/reprex/saliency_analysis/average_smrimask.nii'#loadMasks(cfg, md)
            #maskfile = nib.load(mask_path)
            #save inputs as a nibabel file. Something like "inputSal_"+str(i)
            #save_file_name = 'inputSal_' + str(i) + '.pt'
            #torch.save(inputs[0][0], save_file_name)
            #nib.save(nib.Nifti1Image(inputs[0][0], affine=maskfile.affine, header= maskfile.header), save_file_name)

            all_labels.append(np.squeeze(labels))
            
            # # Wrap in variable and load batch to gpu
            # if cuda_avl:
            #     inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            # else:
            #     inputs, labels = Variable(inputs), Variable(labels)
            # tempmask = np.ones(inputs.shape[2:])
            masks = nib.load(mask_path).get_fdata()
            saliency = Saliency(net)
            #inputs = torch.rand(1,1,60,60,60)
            #print(type())
            #print(inputs.shape, all_labels.shape)
            grads = saliency.attribute(inputs, target=all_labels[0].item())
            print(grads.shape)
            sal_out = grads.squeeze().cpu().detach().numpy()
            print(sal_out.shape)
            #sal_out = sensitivity_raw(net, 'BPraw', inputs, masks, taskmode=cfg.cr, cuda_avl=cuda_avl)
            # import pdb; pdb.set_trace()
            isal.append(sal_out)
            #continue
            
        all_labels = np.hstack(all_labels).squeeze()
        # np.savetxt(cfg.ml+'labels_%s.txt'%md, all_labels, fmt='%d')
        
        print('Saving saliency results to: \n'+cfg.ml)
        

        #import pdb; pdb.set_trace() #this is a debugger
        isal = np.array(isal)
        print("final shape ",isal.shape)
        print(cfg.ml+"test_sal.pkl")
        #np.savetxt(cfg.ml+'isal_%s.csv'%md, isal, delimiter=',')
        with open(cfg.ml+'test_sal.pkl','wb') as f:
            pickle.dump(isal,f)