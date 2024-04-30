import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import boto3
import io
from io import BytesIO

class AcousticSoundsData(Dataset):

    def __init__(self,
                 annotations_file,
                 #audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device,
                 bucket_name
                 ):
        self.annotations = pd.read_excel(annotations_file)
        #self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.bucket_name = bucket_name

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index, self.bucket_name)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index, bucket_name):
        path = self.annotations.iloc[index, 10]
        s3 = boto3.client('s3')
        audio_data = io.BytesIO()
        s3.download_fileobj(bucket_name, path, audio_data)
        audio_data.seek(0)
        return audio_data

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]


if __name__ == "__main__":
    
    
    s3 = boto3.client('s3')
    BUCKET = '2307-01-acoustic-loggers-for-leak-detection-a'
    object_key = 'Metadata_Audio_Connected/train_data.xlsx'

    response = s3.get_object(Bucket=BUCKET, Key=object_key)
    excel_data = response['Body'].read()

    s3_location = BytesIO(excel_data)
    
    ANNOTATIONS_FILE = s3_location
    #AUDIO_DIR = "/home/valerio/datasets/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = AcousticSoundsData(ANNOTATIONS_FILE,
                            #AUDIO_DIR,
                        mel_spectrogram,
                        SAMPLE_RATE,
                        NUM_SAMPLES,
                        device,
                        BUCKET)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
            
    