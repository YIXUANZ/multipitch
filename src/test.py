import sys
import torch
import h5py
import librosa
import numpy as np
import soundfile as sf

from stft import STFT
from torchstft import STFT as STFT_pt
from crossnet import CrossNet
from dccrn import DCCRN
from utils import ToTensor, mvnorm, Norm, act2pitch, freq2cents, cents2freq


class Test(object):

    def __init__(self, sp_model_file='./pretrained/anechoic_ref/sp_weights.pth', pt_model_file='./pretrained/anechoic_ref/pt_weights.pth', device=None, output=None):
        '''
        :param args
        '''
        self.to_tensor = ToTensor()
        if device:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.sp_model_file = sp_model_file
        self.pt_model_file = pt_model_file
        self.output_path = output
        
        self.stft_sp = STFT(n_fft=256, n_hop=128, win_len=256)
        self.stft_pt = STFT_pt(frame_size=1024, frame_shift=80)
        self.norm = Norm(mode='utterance')

    def sep(self, mix, net_sp):
        # speaker separation module
        mix = self.to_tensor(mix[None, None, :])
        X, stft_paras = self.stft_sp.stft(mix)
        B, C, F, T = X.shape
        X, norm_paras = self.norm.norm(X, ref_channel=0)
        X = X.permute(0, 2, 3, 1)  # B,F,T,C; complex
        X = torch.view_as_real(X).reshape(B, F, T, -1)  # B,F,T,2C
        out = net_sp(X)
        if not torch.is_complex(out):
            out = torch.view_as_complex(out.float().reshape(B, F, T, -1, 2))  # [B,F,T,Spk]
        out = out.permute(0, 3, 1, 2)  # [B,Spk,F,T]
        Yr_hat = self.norm.inorm(out, norm_paras)
        sp_est = self.stft_sp.istft(Yr_hat, stft_paras)  
        return sp_est

    def pt(self, sp_est, net_pt):
        # pitch tracking module
        sp_est_1, sp_est_2 = mvnorm(sp_est[:,0,:]), mvnorm(sp_est[:,1,:])
        sp_est_stft_1, sp_est_stft_2= self.stft_pt(sp_est_1), self.stft_pt(sp_est_2)
        feat = torch.cat((sp_est_stft_1, sp_est_stft_2), dim=1).transpose(3, 2)
        spk1_f0_act, spk2_f0_act = net_pt(feat)
        return spk1_f0_act, spk2_f0_act

    def predict(self, filename, model_srate=8000):
        '''
        :param filename
        '''
        # speaker separation
        net_sp = CrossNet(
            dim_input=2,
            dim_output=4,
            num_layers=12,
            dim_squeeze=16,
            num_heads=4,
            num_freqs=129,
        )
        net_pt = DCCRN()
        
        print('Load model.')

        net_sp.load_state_dict(torch.load(self.sp_model_file))
        net_pt.load_state_dict(torch.load(self.pt_model_file))

        net_sp.eval()
        net_pt.eval()

        with torch.no_grad():
            # read and resample audios
            mix, fs_orig = sf.read(filename)
            if fs_orig != model_srate:
                mix = librosa.resample(mix, fs_orig, model_srate)

            sp_est = self.sep(mix, net_sp)
            spk1_f0_act, spk2_f0_act = self.pt(sp_est, net_pt)
            spk1_f0, spk2_f0 = act2pitch(spk1_f0_act), act2pitch(spk2_f0_act)

            h5f = h5py.File(self.output_path + '/' + "output.h5", 'w')
            h5f.create_dataset('spk1_f0', data=spk1_f0)
            h5f.create_dataset('spk2_f0', data=spk2_f0)       
            h5f.close()

        return spk1_f0, spk2_f0


if __name__ == "__main__":
    # Parse arguments manually
    args = sys.argv[1:]  # Exclude the script name
    arguments = {}
    for i in range(0, len(args), 2):  # Iterate in steps of 2 (key-value pairs)
        key = args[i]
        value = args[i + 1]
        arguments[key] = value

    # Extract individual arguments
    test_audio_file = arguments.get("--file")
    sp_model_file = arguments.get("--sp_model")
    pt_model_file = arguments.get("--pt_model")
    device = arguments.get("--device")
    output_dir = arguments.get("--output")

    test = Test(sp_model_file, pt_model_file, device, output_dir)
    test.predict(test_audio_file)

    