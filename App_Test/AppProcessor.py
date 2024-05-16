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
from cnn import CNNNetwork


class AcousticSoundsData(VisionDataset):

    def __init__(self,
                 annotations_file,
                 #audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device,
                 bucket_name
                 ):
        self.annotations = annotations_file
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
        path = self.annotations.iloc[index, 1]
        s3 = boto3.client('s3')
        audio_data = io.BytesIO()
        s3.download_fileobj(bucket_name, path, audio_data)
        audio_data.seek(0)
        return audio_data

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 0]

def predict(model, input, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
    return predicted
 
def annotation(bucket_name, folder_name):
    
    s3 = boto3.client('s3')
    
    # List objects in the specified folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    
    if 'Contents' in response:
        for obj in response['Contents']:
            # Extract only the file name (remove the folder prefix)
            file_name = obj['Key'].split('/')[-1]
            key = f"{folder_name}/{file_name}"
            dict = {'Leak_Status': '0', 'key': key}
            index_label = 'custom_index'
            df = pd.DataFrame(dict, index=[index_label])
    return df
    

def create_data_loader(data, batch_size=1):
    train_dataloader = DataLoader(data, batch_size = batch_size)
    return train_dataloader

class_mapping = [
    "0",
    "1"
]

if __name__ == "__main__":
   

    bucket_name = "2307-01-acoustic-loggers-for-leak-detection-a"
    folder_name = "App_Uploaded_Audios"

            # Download the audio file from S3
    annotation = annotation(bucket_name, folder_name)
   
    
    if annotation is None:
       print("Error: Failed to download audio file from S3.")
    
    acoustic_data = AcousticSoundsData(annotation,
                                       transformation=MelSpectrogram(sample_rate=2250,
                                                                    n_fft=1024,
                                                                    hop_length=512,
                                                                    n_mels=64),
                                        target_sample_rate=2250,
                                        num_samples=10000,
                                        device='cpu',
                                        bucket_name=bucket_name)
    
    print(f"There are {len(acoustic_data)} samples in the dataset.")
    
    input, target = acoustic_data[0][0], acoustic_data[0][1] # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)

    cnn = CNNNetwork()
    state_dict = torch.load("NeuralNetwork.pth")
    cnn.load_state_dict(state_dict)
    
    # make an inference
    predicted, expected = predict(cnn, input, target,
                                  class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")
    
