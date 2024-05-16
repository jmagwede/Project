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
                 Audio,
                 #audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device
                 ):
        self.Audio = Audio
        #self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        
        self.length = 1
        
    def __repr__(self):
        return f"AcousticSoundsData(Audio={self.Audio}, transformation={self.transformation}, ...)"
    
    def __len__(self):
    # Return the total number of samples in the dataset
        return self.length
    

    def __getitem__(self, Audio):
        
        signal, sr = torchaudio.load(self.Audio)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal

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

def predict(model, input, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
    return predicted
 
def name(bucket_name, folder_name):
    # Create an S3 client
    s3 = boto3.client('s3')
    
    # List objects in the specified folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    
    # Extract file names from the response
    audio_files = []
    if 'Contents' in response:
        for obj in response['Contents']:
            # Extract only the file name (remove the folder prefix)
            file_name = obj['Key'].split('/')[-1]
            # Add the file name to the list
            audio_files.append(file_name)
    
    return file_name

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
    file_name = name(bucket_name, folder_name)
    key = f"{folder_name}/{file_name}"
    if file_name is None:
       print("Error: Failed to download audio file from S3.")
       
            
    s3 = boto3.client('s3')
            
    audio_data_io = io.BytesIO()
            
    s3.download_fileobj(bucket_name, key, audio_data_io)
        
    audio_data_io.seek(0)
    

    
    acoustic_data = AcousticSoundsData(audio_data_io,
                                       transformation=MelSpectrogram(sample_rate=2250,
                                                                    n_fft=1024,
                                                                    hop_length=512,
                                                                    n_mels=64),
                                        target_sample_rate=2250,
                                        num_samples=10000,
                                        device='cpu')
    
    data_loader = create_data_loader(acoustic_data, batch_size=1)
    
    # Load the trained model
    cnn = CNNNetwork()
    state_dict = torch.load("NeuralNetwork.pth")
    cnn.load_state_dict(state_dict)
    
    # Make predictions
    predictions = []
    for batch_input in data_loader:
        batch_input = batch_input.unsqueeze(0)  # Add batch dimension
        batch_input = batch_input.to('cpu')
        print(batch_input.size())
        ''' predicted = predict(cnn, batch_input, class_mapping)
        predictions.append(predicted)
    
    # Output the predictions
    print(predictions)'''