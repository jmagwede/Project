
import torch, torchaudio
import boto3
import io
from io import BytesIO
import pandas as pd
from cnn import CNNNetwork
 
 
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

def _get_audio_sample_path(annotations, i, bucket_name):
    path = annotations.iloc[i, 10]
    s3 = boto3.client('s3')
    audio_data = io.BytesIO()
    s3.download_fileobj(bucket_name, path, audio_data)
    audio_data.seek(0)
    return audio_data


def process(audio):
    signal, sr = torchaudio.load(audio)
    signal = signal.to('cpu')
    
    signal = _resample_if_necessary(signal, sr)
    signal = _mix_down_if_necessary(signal)
    signal = _cut_if_necessary(signal)
    signal = _right_pad_if_necessary(signal)
    return signal

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

cnn = CNNNetwork()
state_dict = torch.load("NeuralNetwork.pth")
cnn.load_state_dict(state_dict)

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        n_mels=64)

# Lets get the Audio
s3 = boto3.client('s3')
BUCKET = '2307-01-acoustic-loggers-for-leak-detection-a'
object_key = 'Development Layer/validation_data.xlsx'

response = s3.get_object(Bucket=BUCKET, Key=object_key)
excel_data = response['Body'].read()

s3_location = BytesIO(excel_data)

ANNOTATIONS_FILE = s3_location

df = pd.read_excel(ANNOTATIONS_FILE)
labels = df.iloc[:, 6]

class_mapping = [
    "1",
    "0"
]

# Extracting the Mel Spectrogram
for i in range(len(df)):
    audio = _get_audio_sample_path(df, i, BUCKET)
    signal, rate = torchaudio.load(audio)
    sr = rate
    signal = signal.to('cpu')
    print(sr)
    #signal = _resample_if_necessary(signal, sr)
    print(sr)
    #signal = _mix_down_if_necessary(signal)
    #signal = _cut_if_necessary(signal)
    #signal = _right_pad_if_necessary(signal)
    
    
    
    signal = mel_spectrogram(signal)
    input = signal[0][0]
    input, target = signal[0][0], signal[0][1] 
    input.unsqueeze_(0)
    predicted = predict(cnn, input, target, class_mapping)
    
print(predicted)