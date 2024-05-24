import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from io import BytesIO
import boto3

from AcousticSounds import AcousticSoundsData
from cnn import CNNNetwork


s3 = boto3.client('s3')
BUCKET = '2307-01-acoustic-loggers-for-leak-detection-a'
object_key = 'Development Layer/train_500_data.xlsx'

response = s3.get_object(Bucket=BUCKET, Key=object_key)
excel_data = response['Body'].read()

s3_location = BytesIO(excel_data)
    

BATCH_SIZE = 5
EPOCHS = 4
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = s3_location
#AUDIO_DIR = "/home/valerio/datasets/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

    
def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
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
                            BUCKET
                            )
    
    train_dataloader = create_data_loader(usd, BATCH_SIZE)

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "500_Audio_Nn.pth")
    print("Trained the Neural Network saved")