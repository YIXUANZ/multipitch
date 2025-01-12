# -*- coding: utf-8 -*-

import torch
from scipy import linalg
import numpy as np
import scipy
import torch.nn as nn
class TorchSignalToFrames(object):
    def __init__(self, frame_size=1024, frame_shift=80):
        super(TorchSignalToFrames, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift
    def __call__(self, in_sig):
        sig_len = in_sig.shape[-1]
        nframes = (sig_len // self.frame_shift)
        a = torch.zeros(tuple(in_sig.shape[:-1]) + (nframes, self.frame_size), device=in_sig.device)
        start = -self.frame_size // 2
        end = start + self.frame_size
        for i in range(nframes):
            if end < sig_len:
                if start >= 0:
                    a[..., i, :]=in_sig[..., start:end]
                else:
                    head_pos = -start
                    a[..., i, head_pos:]=in_sig[...,:end]
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size]=in_sig[..., start:]

            start = start + self.frame_shift
            end = start + self.frame_size
        return a

class STFT(object):
    def __init__(self, frame_size=1024, frame_shift=80, device='cpu:0'):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.device = device
        self.frame = TorchSignalToFrames(frame_size=self.frame_size, 
                                         frame_shift=self.frame_shift)
        D = linalg.dft(frame_size)
        W = np.hamming(self.frame_size)
        DR = np.real(D)
        DI = np.imag(D)
        self.DR = torch.from_numpy(DR).float().to(self.device)
        self.DR = self.DR.contiguous().transpose(0, 1)
        self.DI = torch.from_numpy(DI).float().to(self.device)
        self.DI = self.DI.contiguous().transpose(0, 1)
        self.W = torch.from_numpy(W).float().to(self.device)
    def __call__(self, audio):
        audio = self.frame(audio)
        # est_s2 = self.frame(est_s2)
        # mix = self.frame(mix)
        stft_audio = self.get_stft(audio)
        # stft_est_s2 = self.get_stft(est_s2)
        # stft_mix = self.get_stft(mix)
        return stft_audio

    def get_stft(self, frames):
        frames = frames * self.W
        stft_R = torch.matmul(frames, self.DR)[..., 0:(self.frame_size//2+1)]
        stft_I = torch.matmul(frames, self.DI)[..., 0:(self.frame_size//2+1)]
        stft = torch.cat((stft_R[:,None,:,:], stft_I[:,None,:,:]), dim=1)
        return stft 


class stftm_loss(object):
    def __init__(self, frame_size=512, frame_shift=256, loss_type='mae', device='cpu:0'):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.loss_type = loss_type
        self.device = device
        self.frame = TorchSignalToFrames(frame_size=self.frame_size, 
                                         frame_shift=self.frame_shift)
        D = linalg.dft(frame_size)
        W = np.hamming(self.frame_size)
        DR = np.real(D)
        DI = np.imag(D)
        self.DR = torch.from_numpy(DR).float().to(self.device)
        self.DR = self.DR.contiguous().transpose(0, 1)
        self.DI = torch.from_numpy(DI).float().to(self.device)
        self.DI = self.DI.contiguous().transpose(0, 1)
        self.W = torch.from_numpy(W).float().to(self.device)
    def __call__(self, outputs, labels, loss_mask, nframes):
        outputs = self.frame(outputs)
        labels = self.frame(labels)
        loss_mask = self.frame(loss_mask)
        outputs = self.get_stftm(outputs)
        labels = self.get_stftm(labels)
        
        masked_outputs = outputs * loss_mask
        masked_labels = labels * loss_mask
        if self.loss_type == 'mse':
            loss = torch.sum((masked_outputs - masked_labels)**2) / torch.sum(loss_mask)
        elif self.loss_type == 'mae':
            loss =  torch.sum(torch.abs(masked_outputs - masked_labels)) / torch.sum(loss_mask)
        
        return loss
    def get_stftm(self, frames):
        frames = frames * self.W
        stft_R = torch.matmul(frames, self.DR)
        stft_I = torch.matmul(frames, self.DI)
        stftm = torch.abs(stft_R) + torch.abs(stft_I)
        return stftm
class mse_loss(object):
    def __call__(self, outputs, labels, loss_mask, nframes):
        masked_outputs = outputs * loss_mask
        masked_labels = labels * loss_mask
        loss = torch.sum((masked_outputs - masked_labels)**2.0) / torch.sum(loss_mask)
        return loss