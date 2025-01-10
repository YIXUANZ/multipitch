import os
import numpy as np

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

def act2pitch(est_act, voicing_threshold=0.5):
    '''
    Convert estimated vector to fundamental frequency
    '''
    est_act = 1. / (1. + np.exp(-est_act))
    est_act = np.transpose(est_act)

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
