# Importing required libraries 
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import torchaudio
from torchvision import datasets
from torchvision.datasets import VisionDataset
import boto3
import io
from io import BytesIO
from torchaudio.transforms import MelSpectrogram


# Making an audio Preporcessing class 
class SignalPrecessing(VisionDataset):
    # Function to transform an audio data.
    def __init__(self,
                 annotations_file,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device,
                 bucket_name
                 ):
        self.annotations = annotations_file
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.bucket_name = bucket_name
    
    # Function to get the number of audios we are dealing with. 
    def __len__(self):
        return len(self.annotations)
    
    # Function that extract the required features from the audio
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
        signal = torch.log(signal + 1e-9)
        return signal, label
    
    # This function cut the audio signal to the required number of sample 
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    # This function increases the number of sample of an audio if the audio is hsort 
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    # This function perform resampling on the audio signal
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    # This function perform resampling on the audio by pulling the signal down if is long
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    # This function gets the path to the audio signal
    def _get_audio_sample_path(self, index, bucket_name):
        path = self.annotations.iloc[index, 1]
        s3 = boto3.client('s3')
        audio_data = io.BytesIO()
        s3.download_fileobj(bucket_name, path, audio_data)
        audio_data.seek(0)
        return audio_data
    
    # getting the label of the signal
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 0]
