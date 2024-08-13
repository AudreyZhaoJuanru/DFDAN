# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction,CLASSI,FD,CLASSI_2
from utils.dataset import H5Dataset
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc
import kornia

def loss_adv_total(x_adv,x_clean):
    feature_adv_B, feature_adv_D, _ = DIDF_Encoder(x_adv)
    feature_clean_B, feature_clean_D, _ = DIDF_Encoder(x_clean)
    data_adv_hat, _ = DIDF_Decoder(x_adv, feature_adv_B)

    classi_clean = CLASSIFIER(feature_clean_D)
    classi_adv = CLASSIFIER(feature_adv_D)

    cc_loss_B = cc(feature_adv_B, feature_clean_B)
    cc_loss_D = cc(feature_adv_D, feature_clean_D)
    loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

    loss_reconst = Loss_ssim(x_clean, data_adv_hat) + MSELoss(x_clean, data_adv_hat)

    loss_classi = Loss_cross(classi_clean, torch.tensor([1., 0.]).repeat(batch_size, 1).cuda()) \
                  + Loss_cross(classi_adv, torch.tensor([0.,1.]).repeat(batch_size,1).cuda())

    loss = coeff_decomp * loss_decomp + coeff_reconst * loss_reconst + coeff_classi * loss_classi
    return loss


def loss_clean_total(x_clean,x_pos,x_neg_1,x_neg_2,x_neg_3):
    _, _, feature_clean = DIDF_Encoder(x_clean)
    _, _, feature_pos=DIDF_Encoder(x_pos)
    _, _, feature_neg_1 = DIDF_Encoder(x_neg_1)
    _, _, feature_neg_2 = DIDF_Encoder(x_neg_2)
    _, _, feature_neg_3 = DIDF_Encoder(x_neg_3)

    cc_loss_pos=cc(feature_clean, feature_pos)
    cc_loss_neg=cc(feature_clean, feature_neg_1)+cc(feature_clean, feature_neg_2)+cc(feature_clean, feature_neg_3)
    loss = (cc_loss_neg) ** 2 / (1.01 + cc_loss_pos)
    return loss



'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()
model_str = 'CDDFuse'

# . Set the hyper-parameters for training
num_epochs = 2
lr = 1e-4
weight_decay = 0
batch_size = 4
input_size=256
nb_class=4
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']

# Coefficients of the loss function
coeff_decomp = 2.
coeff_reconst=1.
coeff_classi=1.

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5


# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
CLASSIFIER=nn.DataParallel(CLASSI_2(channel_nb=64,class_nb=2)).to(device)
FailtDia=nn.DataParallel(FD(seg_size=input_size,channel_nb=1,class_nb=nb_class)).to(device)
# BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
# DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    CLASSIFIER.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer4 = torch.optim.Adam(
#     DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
# scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIMLoss(1, reduction='mean')
Loss_cross = nn.CrossEntropyLoss()

# data loader
trainloader = DataLoader(H5Dataset(r"data/CWRU_train.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    ''' train '''
    for i, (x_clean, x_adv_1,x_adv_2,x_adv_3,x_pos,x_neg_1,x_neg_2,x_neg_3) in enumerate(loader['train']):
        x_clean,x_adv_1,x_adv_2,x_adv_3,x_pos,x_neg_1,x_neg_2,x_neg_3= \
            x_clean.cuda(),x_adv_1.cuda(),x_adv_2.cuda(),x_adv_3.cuda(),x_pos.cuda(),x_neg_1.cuda(),x_neg_2.cuda(),x_neg_3.cuda()

        DIDF_Encoder.train()
        DIDF_Decoder.train()
        CLASSIFIER.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        CLASSIFIER.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        loss=loss_adv_total(x_adv_1,x_clean)+loss_adv_total(x_adv_2,x_clean)+loss_adv_total(x_adv_3,x_clean)\
             +loss_clean_total(x_clean,x_pos,x_neg_1,x_neg_2,x_neg_3)


        loss.backward()
        nn.utils.clip_grad_norm_(
            DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            CLASSIFIER.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

        optimizer1.step()
        optimizer2.step()
        optimizer3.step()


        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )

    # adjust the learning rate

    scheduler1.step()
    scheduler2.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6

if True:
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'CLASSIFIER':CLASSIFIER.state_dict(),
    }
    torch.save(checkpoint, os.path.join("models/CDDFuse.pth"))

'''
------------------------------------------------------------------------------
Test
------------------------------------------------------------------------------
'''