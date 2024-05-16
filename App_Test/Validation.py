import torch
import torchaudio

from cnn import CNNNetwork
from AcousticSounds import AcousticSoundsData
from train import SAMPLE_RATE, NUM_SAMPLES
from io import BytesIO
import boto3
import pandas as pd


class_mapping = [
    "0",
    "1"
    ]


'''s3 = boto3.client('s3')
BUCKET = '2307-01-acoustic-loggers-for-leak-detection-a'
object_key = 'Development Layer/validation_data.xlsx'

response = s3.get_object(Bucket=BUCKET, Key=object_key)
excel_data = response['Body'].read()

s3_location = BytesIO(excel_data)

ANNOTATIONS_FILE = s3_location'''

BUCKET = '2307-01-acoustic-loggers-for-leak-detection-a'

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

def annotation(bucket_name, folder_name):
    
    s3 = boto3.client('s3')
    
    # List objects in the specified folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    
    if 'Contents' in response:
        for obj in response['Contents']:
            # Extract only the file name (remove the folder prefix)
            file_name = obj['Key'].split('/')[-1]
            key = f"{folder_name}/{file_name}"
            dict = {'Leak_Status': 0, 'key': key}
            index_label = 'custom_index'
            df = pd.DataFrame(dict, index=[index_label])
    return df

bucket_name = "2307-01-acoustic-loggers-for-leak-detection-a"
folder_name = "App_Uploaded_Audios"

            # Download the audio file from S3
ANNOTATIONS_FILE = annotation(bucket_name, folder_name)

if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("NeuralNetwork.pth")
    cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
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
                            "cpu",
                            BUCKET)


    # get a sample from the acoustic sound dataset for inference
    input, target = usd[0][0], usd[0][1] # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)
    predicted, _ = predict(cnn, input, target,
                                  class_mapping)

    print(f"Predicted: '{predicted}'")
    