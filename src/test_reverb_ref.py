import os
import torch
import timeit
import shutil
import random
import librosa
import scipy.io
import numpy as np
import soundfile as sf
import scipy.io as sio

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import librosa.display
from scipy.io import savemat

from norm import Norm
from torchstft import STFT
import sisdr_loss as sisdr_lib
from crossnet_v3 import CrossNet
from stft_enh import STFT as STFT_ENH
from dccrn_pitch import Net as DCCRN_PITCH
from criteria import PitchLossFunction, SPLossFunction
from utils.utils import wavNormalize
from utils.pipeline_modules import NetFeeder
from f0_to_target_convertor import freq2cents, cents2freq
from doublePitchErrorMeasure import doublePitchErrorMeasure
from pyutils import compLossMask, numParams, logging, plotting, metric_logging
from dataset_praat import TrainingDataset, EvalDataset, TestDataset, ToTensor, TrainCollate, EvalCollate, TestCollate

from torch.cuda.amp import autocast, GradScaler


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


def to_viterbi_cents(salience, vecSize=486, smoothing_factor=12, modelTag=993):
    """
    Find the Viterbi path using a transition prior that induces pitch
    continuity.
    """
    from hmmlearn import hmm

    # uniform prior on the starting pitch
    starting = np.ones(vecSize) / vecSize

    # transition probabilities inducing continuous pitch
    xx, yy = np.meshgrid(range(vecSize), range(vecSize))
    transition = np.maximum(smoothing_factor - abs(xx - yy), 0)
    transition = transition / np.sum(transition, axis=1)[:, None]

    # emission probability = fixed probability for self, evenly distribute the
    # others
    self_emission = 0.1
    emission = (np.eye(vecSize) * self_emission + np.ones(shape=(vecSize, vecSize)) *
                ((1 - self_emission) / vecSize))

    # fix the model parameters because we are not optimizing the model
    model = hmm.MultinomialHMM(vecSize, starting, transition)
    model.startprob_, model.transmat_, model.emissionprob_ = \
        starting, transition, emission

    # find the Viterbi path

    observations = np.argmax(salience, axis=1)
    path = model.predict(observations.reshape(-1, 1), [len(observations)])

    if(modelTag=='CREPE'):
        return np.array([to_local_average_cents_CREPE(salience[i, :], path[i]) for i in
                         range(len(observations))])
    else:
        return np.array([to_local_average_cents(salience[i, :], path[i]) for i in
                         range(len(observations))])

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    # mvn
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)

class Checkpoint(object):
    def __init__(self, start_epoch=None, start_iter=None, train_loss=None, eval_loss=None, best_loss=np.inf, state_dict_pitch=None, optimizer=None):
        self.start_epoch = start_epoch
        self.start_iter = start_iter
        self.train_loss = train_loss
        self.eval_loss = eval_loss
        self.best_loss = best_loss
        self.state_dict_pitch = state_dict_pitch
        self.optimizer = optimizer
    
    
    def save(self, is_best, filename, best_model):
        print('Saving checkpoint at "%s"' % filename)
        torch.save(self, filename)
        if is_best:
            print('Saving the best model at "%s"' % best_model)
            shutil.copyfile(filename, best_model)
        print('\n')


    def load(self, filename):
        if os.path.isfile(filename):
            print('Loading checkpoint from "%s"\n' % filename)
            checkpoint = torch.load(filename, map_location='cpu')
            
            self.start_epoch = checkpoint.start_epoch
            self.start_iter = checkpoint.start_iter
            self.train_loss = checkpoint.train_loss
            self.eval_loss = checkpoint.eval_loss
            self.best_loss = checkpoint.best_loss
            self.state_dict_pitch = checkpoint.state_dict_pitch
            self.optimizer = checkpoint.optimizer
        else:
            raise ValueError('No checkpoint found at "%s"' % filename)

class Checkpoint_SP(object):
    def __init__(self, start_epoch=None, start_iter=None, train_loss=None, eval_loss=None, best_loss=np.inf, state_dict_eh=None, optimizer=None):
        self.start_epoch = start_epoch
        self.start_iter = start_iter
        self.train_loss = train_loss
        self.eval_loss = eval_loss
        self.best_loss = best_loss
        self.state_dict_eh = state_dict_eh
        self.optimizer = optimizer
    
    def save(self, is_best, filename, best_model):
        print('Saving checkpoint at "%s"' % filename)
        torch.save(self, filename)
        if is_best:
            print('Saving the best model at "%s"' % best_model)
            shutil.copyfile(filename, best_model)
        print('\n')

    def load(self, filename):
        if os.path.isfile(filename):
            print('Loading checkpoint from "%s"\n' % filename)
            checkpoint = torch.load(filename, map_location='cpu')
            
            self.start_epoch = checkpoint.start_epoch
            self.start_iter = checkpoint.start_iter
            self.train_loss = checkpoint.train_loss
            self.eval_loss = checkpoint.eval_loss
            self.best_loss = checkpoint.best_loss
            self.state_dict_eh = checkpoint.state_dict_eh
            self.optimizer = checkpoint.optimizer
        else:
            raise ValueError('No checkpoint found at "%s"' % filename)

class Model(object):
    
    def __init__(self, args):
        # Hyper parameters
        self.label_size = args.label_size
        self.batch_size = args.batch_size
        self.model_name = args.model_name

        self.alpha = 1
        
        self.to_tensor = ToTensor()
        self.DCCRN_SP_criterion = SPLossFunction()
        self.DCCRN_PITCH_criterion = PitchLossFunction()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.norm = Norm(mode='utterance').to(self.device)
        self.stft_enh = STFT_ENH(n_fft=256, n_hop=128, win_len=256).to(self.device)
        self.stft = STFT(frame_size=1024, frame_shift=80, device=self.device)
        print(self.device)

    def train(self,args):
        
        self.resume = args.resume
        self.max_epoch = args.max_epoch
        self.eval_steps = args.eval_steps
        self.learning_rate = args.learning_rate
        self.use_praat = args.use_praat

        with open(args.train_list,'r') as train_list_file:
            self.train_list = [line.strip() for line in train_list_file.readlines()]
        with open(args.eval_list,'r') as eval_list_file:
            self.cv_list = [line.strip() for line in eval_list_file.readlines()]
        self.num_train_sentences = len(self.train_list)

        self.log_path = args.log_path 
        self.model_path = args.model_path

        # create a training dataset and an evaluation dataset
        trainSet = TrainingDataset(self.train_list, self.use_praat)
        evalSet = EvalDataset(self.cv_list, self.use_praat)

        # create data loaders for training and evaluation
        train_loader = DataLoader(trainSet,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True, 
                                  drop_last=True,
                                  collate_fn=TrainCollate())
        eval_loader = DataLoader(evalSet,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 pin_memory=True, 
                                 drop_last=True,
                                 collate_fn=EvalCollate())
        
        torch.manual_seed(0)
        
        # create a network
        print('model', self.model_name)

        os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
        # torch.distributed.init_process_group(backend="nccl")
        dccrn_pitch = DCCRN_PITCH().cuda()
        dccrn_pitch = torch.nn.DataParallel(dccrn_pitch)
        dccrn_pitch.to(self.device)
        
        print('Number of learnable parameters: %d' % (numParams(dccrn_pitch)))
        
        print(dccrn_pitch)

        optimizer = torch.optim.Adam([{'params': dccrn_pitch.parameters()}], lr=self.learning_rate, amsgrad=True)        
        if self.resume:
            print('Load model from "%s"' % self.resume)
            checkpoint = torch.load(self.resume, map_location='cpu')
            dccrn_pitch.load_state_dict(checkpoint.state_dict_pitch)
            start_epoch = checkpoint.start_epoch
            start_iter = checkpoint.start_iter
            best_loss = checkpoint.best_loss
            optimizer.load_state_dict(checkpoint.optimizer)
        else:
            start_epoch = 0
            start_iter = 0
            best_loss = np.inf

        cnt = 0.
        num_train_batches = self.num_train_sentences // self.batch_size
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, steps_per_epoch=len(train_loader), epochs=self.max_epoch)
        print(best_loss)

        scaler = GradScaler()
        for epoch in range(start_epoch, self.max_epoch):
            accu_train_loss = 0.0
            dccrn_pitch.train()

            start  = timeit.default_timer()
            for i, (s1_clean_feat, s2_clean_feat, est_feat1, est_feat2, s1_f0_label, s2_f0_label) in enumerate(train_loader):

                i += start_iter

                # device
                s1_clean_feat = s1_clean_feat.to(self.device, dtype=torch.float32)
                s2_clean_feat = s2_clean_feat.to(self.device, dtype=torch.float32)
                est_feat1 = est_feat1.to(self.device, dtype=torch.float32)
                est_feat2 = est_feat2.to(self.device, dtype=torch.float32)
                # mvn 
                s1_clean_feat = normalize_tensor_wav(s1_clean_feat)
                s2_clean_feat = normalize_tensor_wav(s2_clean_feat)
                est_feat1 = normalize_tensor_wav(est_feat1)
                est_feat2 = normalize_tensor_wav(est_feat2)
                clean_feat = torch.cat((s1_clean_feat, s2_clean_feat), 1)
                est_feat = torch.cat((est_feat1, est_feat2), 1)
                # complex stft feature
                est_stft_feat1, est_stft_feat2 = self.stft(est_feat1[:,0,:]), self.stft(est_feat2[:,0,:])

                s1_f0_label = s1_f0_label.to(self.device, dtype=torch.float32)
                s2_f0_label = s2_f0_label.to(self.device, dtype=torch.float32)

                optimizer.zero_grad()
                # forward + backward + optimize
                with autocast():
                    with torch.enable_grad():
                        feat = torch.cat((est_stft_feat1, est_stft_feat2), dim=1)
                        feat = feat.transpose(3, 2)
                        est_f0_1, est_f0_2 = dccrn_pitch(feat)

                    _, perm = self.DCCRN_SP_criterion(est_feat, clean_feat)
                    loss = self.DCCRN_PITCH_criterion(est_f0_1, est_f0_2, s1_f0_label, s2_f0_label, perm, f0_diff=True) # matched permutation
                
                # loss.backward()
                scaler.scale(loss).backward()
                           
                nn.utils.clip_grad_value_(dccrn_pitch.parameters(), 1.0)

                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()

                running_loss = loss.data.item()
                tracking_loss = loss.data.item()
                accu_train_loss += running_loss

                cnt += 1.

                del loss, feat, s1_clean_feat, s2_clean_feat, est_feat1, est_feat2, est_stft_feat1, est_stft_feat2, s1_f0_label, s2_f0_label

                end = timeit.default_timer()

                curr_time = end - start

                print('iter = {}/{}, epoch = {}/{}, loss = {:.5f}, f0_loss = {:.5f}, time/batch = {:.5f}'.format(i+1,
                num_train_batches, epoch+1, self.max_epoch, running_loss, tracking_loss, curr_time))

                if (i+1) % self.eval_steps == 0:

                    start = timeit.default_timer()

                    avg_train_loss = accu_train_loss / cnt

                    avg_eval_loss, avg_eval_acc = self.validate(dccrn_pitch, eval_loader)

                    dccrn_pitch.train()

                    print('Epoch [%d/%d], Iter [%d/%d]  ( TrainLoss: %.4f | EvalLoss: %.4f | EvalF0Loss: %.4f )' % (epoch+1,self.max_epoch,i+1,self.num_train_sentences//self.batch_size,avg_train_loss,avg_eval_loss,avg_eval_acc))

                    is_best = True if avg_eval_acc < best_loss else False
                    best_loss = avg_eval_acc if is_best else best_loss

                    checkpoint = Checkpoint(epoch, i, avg_train_loss, avg_eval_loss, best_loss, dccrn_pitch.state_dict(), optimizer.state_dict())

                    model_name = self.model_name + '_latest.model'
                    best_model = self.model_name + '_best.model'

                    checkpoint.save(is_best, os.path.join(self.model_path, model_name), os.path.join(self.model_path, best_model))

                    logging(self.log_path, self.model_name +'_loss_log.txt', checkpoint, self.eval_steps)

                    accu_train_loss = 0.0
                    cnt = 0.

                if (i+1)%num_train_batches == 0:
                    break

            avg_eval_loss, avg_eval_acc = self.validate(dccrn_pitch, eval_loader)
            scheduler.step(avg_eval_loss)

            dccrn_pitch.train()

            print('After {} epoch the performance on validation is:'.format(epoch+1))
            print(avg_eval_loss)

            metric_logging(self.log_path, self.model_name +'_metric_log.txt', epoch, [avg_eval_loss, avg_eval_acc])
            start_iter = 0.

    def validate(self, dccrn_pitch, eval_loader):
        #print('********** Started evaluation on validation set ********')
        dccrn_pitch.eval()
        
        with torch.no_grad():
            mtime = 0
            ttime = 0.
            cnt = 0.
            accu_eval_loss = 0.0
            accu_eval_pitch_loss = 0.0
            for k, (s1_clean_feat, s2_clean_feat, est_feat1, est_feat2, s1_f0_label, s2_f0_label) in enumerate(eval_loader):

                start = timeit.default_timer()

                # device
                s1_clean_feat = s1_clean_feat.to(self.device, dtype=torch.float32)
                s2_clean_feat = s2_clean_feat.to(self.device, dtype=torch.float32)
                est_feat1 = est_feat1.to(self.device, dtype=torch.float32)
                est_feat2 = est_feat2.to(self.device, dtype=torch.float32)
                # mvn 
                s1_clean_feat = normalize_tensor_wav(s1_clean_feat)
                s2_clean_feat = normalize_tensor_wav(s2_clean_feat)
                est_feat1 = normalize_tensor_wav(est_feat1)
                est_feat2 = normalize_tensor_wav(est_feat2)
                clean_feat = torch.cat((s1_clean_feat, s2_clean_feat), 1)
                est_feat = torch.cat((est_feat1, est_feat2), 1)
                # complex stft feature
                est_stft_feat1, est_stft_feat2 = self.stft(est_feat1[:,0,:]), self.stft(est_feat2[:,0,:])

                s1_f0_label = s1_f0_label.to(self.device, dtype=torch.float32)
                s2_f0_label = s2_f0_label.to(self.device, dtype=torch.float32)

                feat = torch.cat((est_stft_feat1, est_stft_feat2), dim=1)
                feat = feat.transpose(3, 2)
                est_f0_1, est_f0_2 = dccrn_pitch(feat)

                _, perm = self.DCCRN_SP_criterion(est_feat, clean_feat)
                eval_loss = self.DCCRN_PITCH_criterion(est_f0_1, est_f0_2, s1_f0_label, s2_f0_label, perm) # matched permutation

                del feat, clean_feat, est_feat, s1_clean_feat, s2_clean_feat, est_feat1, est_feat2, est_stft_feat1, est_stft_feat2, s1_f0_label, s2_f0_label

                accu_eval_loss += eval_loss
                accu_eval_pitch_loss += eval_loss
                
                cnt += 1.
                
                end = timeit.default_timer()
                curr_time = end - start
                
            avg_eval_loss = accu_eval_loss / cnt
            avg_eval_pitch_loss = accu_eval_pitch_loss / cnt

        return avg_eval_loss, avg_eval_pitch_loss


    def test(self, args, model_srate=8000, model_input_size=929):

        from torchmetrics.audio import PermutationInvariantTraining
        from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

        self.model_name = args.model_name
        self.model_file = args.model_file
        self.output_dir = args.output_dir
        self.voicing_threshold = 0.5
        # os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
        
        # separation network
        model_file = '/fs/ess/PAA0005/yixuanz/psc_exp/crossnet_sep_libri360mix_reverb_100k_reverb_target_sisdr_mc_r1/crossnet_sep_libri360mix_reverb_100k_reverb_target_sisdr_mc_r1_best.model'
        # model_file = '/fs/ess/PAA0005/yixuanz/psc_exp/crossnet_sep_libri360mix_reverbnoisy_100k_reverb_target_sisdr_mc/crossnet_sep_libri360mix_reverbnoisy_100k_reverb_target_sisdr_mc_best.model'
        crossnet = CrossNet(
            dim_input=2,
            dim_output=4,
            num_layers=12,
            dim_squeeze=16,
            num_heads=4,
            num_freqs=129,
        )
        crossnet.to(self.device)
        crossnet.eval()

        print('Load model from "%s"' % model_file)
        checkpoint_sp = Checkpoint_SP()
        checkpoint_sp.load(model_file)
        for key in list(checkpoint_sp.state_dict_eh.keys()):
            checkpoint_sp.state_dict_eh[key.replace('module.', '')] = checkpoint_sp.state_dict_eh.pop(key)
        crossnet.load_state_dict(checkpoint_sp.state_dict_eh)

        # create a network
        print('model', self.model_name)
        dccrn_pitch = DCCRN_PITCH()
        dccrn_pitch.to(self.device)

        print('Number of learnable parameters: %d' % numParams(dccrn_pitch))
        print(dccrn_pitch)

        # loss and optimizer
        dccrn_pitch.eval()

        print('Load model from "%s"' % self.model_file)
        checkpoint = Checkpoint()
        checkpoint.load(self.model_file)
        for key in list(checkpoint.state_dict_pitch.keys()):
            checkpoint.state_dict_pitch[key.replace('module.', '')] = checkpoint.state_dict_pitch.pop(key)
        dccrn_pitch.load_state_dict(checkpoint.state_dict_pitch)

        with open(args.test_list,'r') as test_list_file:
            self.test_list = [line.strip() for line in test_list_file.readlines()]

        with torch.no_grad():

            testSet = TestDataset(self.test_list)

            # create a data loader for test
            test_loader = DataLoader(testSet,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=2,
                                     collate_fn=TestCollate())

            cnt = 0.
            ttime = 0.
            mtime = 0.
            accu_test_loss = 0.0
            accu_test_nframes = 0
            total_err_list = []
            total_sisdr_list = []
            for k, (mix_feat, target_feat, s1_f0_label, s2_f0_label) in enumerate(test_loader):
                print(k)
                start = timeit.default_timer()

                # separation
                mix_feat = mix_feat.to(self.device, dtype=torch.float32)
                mix_feat = mix_feat[None,:].to(self.device, dtype=torch.float32)
                mix_ilens = torch.tensor([mix_feat.shape[2]]).repeat(mix_feat.shape[0])

                target_feat = target_feat.to(self.device, dtype=torch.float32)
                target_feat = target_feat[None,:].to(self.device, dtype=torch.float32)

                X, stft_paras = self.stft_enh.stft(mix_feat)  # [B,C,F,T], complex
                B, C, F, T = X.shape
                X, norm_paras = self.norm.norm(X, ref_channel=0)
                X = X.permute(0, 2, 3, 1)  # B,F,T,C; complex
                X = torch.view_as_real(X).reshape(B, F, T, -1)  # B,F,T,2C
                out = crossnet(X)
                if not torch.is_complex(out):
                    out = torch.view_as_complex(out.float().reshape(B, F, T, -1, 2))  # [B,F,T,Spk]
                out = out.permute(0, 3, 1, 2)  # [B,Spk,F,T]
                Yr_hat = self.norm.inorm(out, norm_paras)
                spk_est = self.stft_enh.istft(Yr_hat, stft_paras)  

                # mvn 
                est_feat1 = normalize_tensor_wav(spk_est[:,0,:])
                est_feat2 = normalize_tensor_wav(spk_est[:,1,:])
                # complex stft feature
                est_stft_feat1, est_stft_feat2 = self.stft(est_feat1), self.stft(est_feat2)

                s1_f0_label = s1_f0_label[0,:].cpu().numpy()
                s2_f0_label = s2_f0_label[0,:].cpu().numpy()

                feat = torch.cat((est_stft_feat1, est_stft_feat2), dim=1)
                feat = feat.transpose(3, 2)
                spk1_f0_act, spk2_f0_act = dccrn_pitch(feat)

                spk1_f0_act, spk2_f0_act = spk1_f0_act[0,:].cpu().numpy(), spk2_f0_act[0,:].cpu().numpy()
                
                # spk1
                spk1_f0_act = 1. / (1. + np.exp(-spk1_f0_act))
                spk1_f0_act = np.transpose(spk1_f0_act)

                spk1_voicing_est = spk1_f0_act[:,0]
                spk1_voicing_est[spk1_voicing_est > self.voicing_threshold] = 1
                spk1_voicing_est[spk1_voicing_est <= self.voicing_threshold] = 0
                
                spk1_f0_act = spk1_f0_act[:,1:]

                spk1_cents = []
                for act in spk1_f0_act:
                    spk1_cents.append(to_local_average_cents(act))
                spk1_cents = np.array(spk1_cents)


                spk1_frequencies = cents2freq(spk1_cents)
                spk1_frequencies[np.isnan(spk1_frequencies)] = 0
                spk1_est_f0 = spk1_frequencies * spk1_voicing_est

                # spk2
                spk2_f0_act = 1. / (1. + np.exp(-spk2_f0_act))
                spk2_f0_act = np.transpose(spk2_f0_act)

                spk2_voicing_est = spk2_f0_act[:,0]
                spk2_voicing_est[spk2_voicing_est > self.voicing_threshold] = 1
                spk2_voicing_est[spk2_voicing_est <= self.voicing_threshold] = 0
                
                spk2_f0_act = spk2_f0_act[:,1:]

                spk2_cents = []
                for act in spk2_f0_act:
                    spk2_cents.append(to_local_average_cents(act))
                spk2_cents = np.array(spk2_cents)
                
                spk2_frequencies = cents2freq(spk2_cents)
                spk2_frequencies[np.isnan(spk2_frequencies)] = 0
                spk2_est_f0 = spk2_frequencies * spk2_voicing_est

                spk1_est_f0 = np.append([0], spk1_est_f0)
                spk2_est_f0 = np.append([0], spk2_est_f0)
                min_len = min(len(s1_f0_label), len(spk1_est_f0))
                print(len(s1_f0_label), len(spk1_est_f0))
                s1_f0_label = s1_f0_label[:min_len]
                s2_f0_label = s2_f0_label[:min_len]
                spk1_est_f0 = spk1_est_f0[:min_len]
                spk2_est_f0 = spk2_est_f0[:min_len]

                Etotal = doublePitchErrorMeasure(s1_f0_label[:,None], s2_f0_label[:,None], spk1_est_f0[:,None], spk2_est_f0[:,None], 'GLOBALPERM', rt=5)
                print(sum(Etotal[:6]))
                total_err_list.append(sum(Etotal[:6]))

            print(np.mean(np.asarray(total_sisdr_list)))
            print(np.mean(np.asarray(total_err_list)))
