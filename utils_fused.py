import os
import torch
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
    relative_path_lowres_sMRI = '/data/users2/ibatta/data/features/lowresSMRI/ADNI/nii/'
    #relative_path_lowres_fMRI = '/data/users2/ibatta/data/features/fmrimeasures/ADNI/nii/'
    rel_path_MIP = '/data/users2/vitkyal/projects/complex/MIP/max/MIP_'
    #all_measures = ['swarest1_ALFF.nii', 'swarest1_KccReHo.nii', 'swarest1_DegreeCentrality_DegreeCentrality_PositiveBinarizedSumBrain.nii','swarest1_PerAF.nii', 'swarest1_DegreeCentrality_DegreeCentrality_PositiveWeightedSumBrain.nii', 'swarest1_VMHC.nii', 'swarest1_fALFF.nii', 'smwc1T1.nii.gz']
    sMRI_file_name = 'smwc1T1.nii.gz'
    fNs = relative_path_lowres_sMRI + df['lowres_smriPath'].iloc[idx] + sMRI_file_name
    #fNf = relative_path_lowres_fMRI + df['lowres_fmriPath'].iloc[idx] + all_measures[k]
    url = df['lowres_fmriPath'].iloc[idx]
    session = ('{}'.format(*url.split('/')[3:]))
    subj_id = ('{}'.format(*url.split('/')[0:]))
    fNf = rel_path_MIP + subj_id + '_' + session + '.pt'
    #Xs = np.float32(nib.load(fNs).get_fdata()) 
    Xs = torch.tensor(nib.load(fNs).get_fdata(), dtype=None, device=None, requires_grad=False)
    #Xf = np.float32(nib.load(fNf).get_fdata()) #increase dim of X
    Xf = torch.load(fNf)
    
    #Xs = (Xs - Xs.min()) / (Xs.max() - Xs.min())
    #Xf = (Xf - Xf.min()) / (Xf.max() - Xf.min())
    #X = np.float32(np.concatenate((Xs, Xf), axis=0)) #concat in dim 1
    #X = np.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2]))
    #Xf = np.reshape(Xf, (1, Xf.shape[0], Xf.shape[1], Xf.shape[2]))

    #here is the possibly complex part that i'm coding for rn
    #Xs = (Xs - torch.mean(Xs))/torch.std(Xs)
    #Xf = (Xf - torch.mean(Xf))/torch.std(Xf)
    c = torch.zeros_like(Xf)
    d = torch.zeros_like(Xf)
    maskd = (Xs > 0.05)
    c[maskd] = Xf[maskd] / Xs[maskd]
    d[maskd] = np.sqrt(Xs[maskd]*Xs[maskd] + Xf[maskd]*Xf[maskd])
    Xc2 = np.float32((c - torch.mean(c))/torch.std(c))
    Xc1 = np.float32((d - torch.mean(d))/torch.std(d))
    # initialize output tensor with desired value
    #c = torch.full_like(Xs, fill_value=float('nan'))

    # zero mask
    #mask = (Xf != 0)

    # finally perform division
    #c[mask] = Xs[mask] / Xf[mask]
    #Xc2 = Xs/Xf
    #Xc2 = np.float32(torch.nan_to_num(c, nan=1))

    Xc1 = (Xc1 - Xc1.min()) / (Xc1.max() - Xc1.min())
    Xc2 = (Xc2 - Xc2.min()) / (Xc2.max() - Xc2.min())
    
    Xc1 = np.reshape(Xc1, (1, Xc1.shape[0], Xc1.shape[1], Xc1.shape[2]))
    Xc2 = np.reshape(Xc2, (1, Xc2.shape[0], Xc2.shape[1], Xc2.shape[2]))

    X = np.float32(np.concatenate((Xc1, Xc2), axis=0)) #concat in dim 1
    #X = Xs  #THIS CHANGES IF ITS UNIMODAL OR MULTIMODAL

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


def train(dataloader, net, optimizer, criterion, cuda_avl):

    net.train()

    # Iterate over dataloader batches
    for _, data in enumerate(dataloader, 0):

        # Fetch the inputs
        inputs, labels = data
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

    if cfg.cr == 'clx':

        # Evaluate classification performance
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        return acc, bal_acc

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
                                        'tr_acc', 'bal_tr_acc', 'val_acc', 'bal_val_acc', 'loss'])
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
            train_acc, bal_train_acc = evalMetrics(trainloader, net, cfg)
            valid_acc, bal_valid_acc = evalMetrics(validloader, net, cfg)

            # Log Performance
            history.loc[epoch] = [cfg.k, cfg.scorename, cfg.iter, epoch, train_acc,
                                  bal_train_acc, valid_acc, bal_valid_acc, loss]

            # Check for maxima (e.g. accuracy for classification)
            isBest = valid_acc > m_val_acc

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
    testloader = loadData(cfg, 'te')

    if cfg.cr == 'clx':

        # Initialize Log File
        outs = pd.DataFrame(columns=['k', 'iter', 'acc_te', 'bal_acc_te'])

        print('Testing: ')

        # Evaluate classification performance
        acc, bal_acc = evalMetrics(testloader, net, cfg)

        # Log Performance

        outs.loc[0] = [cfg.k, cfg.iter, acc, bal_acc]

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
                         shuffle=True, num_workers=cfg.nw, drop_last=True, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True)

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
