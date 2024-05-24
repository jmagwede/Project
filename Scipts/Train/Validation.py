import torch
import torchaudio

from cnn import CNNNetwork
from AcousticSounds import AcousticSoundsData
from train import SAMPLE_RATE, NUM_SAMPLES
from io import BytesIO
import boto3


class_mapping = [
    'Non_leak',
    'leak'
]

s3 = boto3.client('s3')
BUCKET = '2307-01-acoustic-loggers-for-leak-detection-a'
#object_key = 'Development Layer/validation_data.xlsx'
object_key = 'Development Layer/validation_data.xlsx'

response = s3.get_object(Bucket=BUCKET, Key=object_key)
excel_data = response['Body'].read()

s3_location = BytesIO(excel_data)

ANNOTATIONS_FILE = s3_location

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("500_Audio_Nn.pth")
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
    # [batch size, num_channels, fr, time]
    
    predicted_list = []
    expected_list = []
    for i in range(len(usd)):
        input, target = usd[i][0], usd[i][1]
        input.unsqueeze_(0)
        predicted, expected = predict(cnn, input, target,
                                  class_mapping)
        predicted_list = predicted_list + [predicted]
        expected_list = expected_list + [expected]
    '''input, target = usd[0][0], usd[0][1]
    input.unsqueeze_(0)
    predicted, expected = predict(cnn, input, target,
                                  class_mapping)'''
    
    
print(f"Predicted: '{predicted_list}', expected: '{expected_list}'")
