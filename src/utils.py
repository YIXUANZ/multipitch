import os
import numpy as np
from typing import *

import torch
from torch import nn
from torch import Tensor

class ToTensor(object):
    r"""Convert ndarrays in sample to Tensors."""
    def __call__(self, x):
        return torch.from_numpy(x).float()

def mvnorm(wav_tensor, eps=1e-8, std=None):
    # mvn
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)

def to_local_average_cents(salience, center=None, fmin=30., fmax=1000., vecSize=486):
    '''
    find the weighted average cents near the argmax bin in output pitch class vector

    :param salience: output vector of salience for each pitch class
    :param fmin: minimum ouput frequency (corresponding to the 1st pitch class in output vector)
    :param fmax: maximum ouput frequency (corresponding to the last pitch class in output vector)
    :param vecSize: number of pitch classes in output vector
    :return: predicted pitch in cents
    '''

    if not hasattr(to_local_average_cents, 'mapping'):
        # the bin number-to-cents mapping
        fmin_cents = freq2cents(fmin)
        fmax_cents = freq2cents(fmax)
        to_local_average_cents.mapping = np.linspace(fmin_cents, fmax_cents, vecSize) # cents values corresponding to the bins of the output vector

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience)) # index of maximum value in output vector
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents.mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :]) for i in
                         range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")


def act2pitch(est_act, voicing_threshold=0.5):
    '''
    Convert estimated vector to fundamental frequency
    '''
    est_act = est_act[0,:].transpose(0,1).cpu().numpy()
    est_act = 1. / (1. + np.exp(-est_act))

    est_voicing = est_act[:,0]
    est_voicing[est_voicing > voicing_threshold] = 1
    est_voicing[est_voicing <= voicing_threshold] = 0

    est_cents = []
    for act in est_act[:,1:]:
        est_cents.append(to_local_average_cents(act))
    est_freq = cents2freq(np.array(est_cents))
    est_freq[np.isnan(est_freq)] = 0

    est_f0 = est_freq * est_voicing
    return est_f0

def freq2cents(f0, f_ref = 10.):
    '''
    Convert a given frequency into its corresponding cents value, according to given reference frequency f_ref
    :param f0: f0 value (in Hz)
    :param f_ref: reference frequency for conversion to cents (in Hz)
    :return: value in cents
    '''
    c = 1200 * np.log2(f0/(f_ref + 2e-30) + 2e-30)
    return c

def cents2freq(cents, f_ref = 10.):
    '''
    conversion from cents value to f0 in Hz

    :param cents: pitch value in cents
    :param fref: reference frequency used for conversion
    :return: f0 value
    '''
    f0 = f_ref * 2 ** (cents / 1200)
    return f0

class Norm(nn.Module):

    def __init__(self, mode: Literal['utterance', 'frequency', 'none']) -> None:
        super().__init__()
        self.mode = mode

    def forward(self, X: Tensor, norm_paras: Any = None, inverse: bool = False) -> Any:
        if not inverse:
            return self.norm(X, norm_paras=norm_paras)
        else:
            return self.inorm(X, norm_paras=norm_paras)

    def norm(self, X: Tensor, norm_paras: Any = None, ref_channel: int = None) -> Tuple[Tensor, Any]:
        """ normalization
        Args:
            X: [B, Chn, F, T], complex
            norm_paras: the paramters for inverse normalization or for the normalization of other X's

        Returns:
            the normalized tensor and the paramters for inverse normalization
        """
        if self.mode == 'none':
            return X, None

        B, C, F, T = X.shape
        if norm_paras is None:
            
            Xr = X[:, [ref_channel], :, :].clone()  # [B,1,F,T], complex

            if self.mode == 'frequency':
                XrMM = torch.abs(Xr).mean(dim=2, keepdim=True) + 1e-8  # Xr_magnitude_mean, [B,1,F,1]
            else:
                assert self.mode == 'utterance', self.mode
                XrMM = torch.abs(Xr).mean(dim=(2, 3), keepdim=True) + 1e-8  # Xr_magnitude_mean, [B,1,1,1]
        else:
            Xr, XrMM = norm_paras
        X[:, :, :, :] /= XrMM
        return X, (Xr, XrMM)

    def inorm(self, X: Tensor, norm_paras: Any) -> Tensor:
        """ inverse normalization
        Args:
            x: [B, Chn, F, T], complex
            norm_paras: the paramters for inverse normalization 

        Returns:
            the normalized tensor and the paramters for inverse normalization
        """

        Xr, XrMM = norm_paras
        return X * XrMM

    def extra_repr(self) -> str:
        return f"{self.mode}"