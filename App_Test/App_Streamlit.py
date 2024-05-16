
import streamlit as st
 
from pydub import AudioSegment
import os

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from AppProcessor import AcousticSoundsData
from cnn import CNNNetwork
import io
import boto3
import pandas as pd


class_mapping = [
    "Not Leaking",
    "Leaking"
]

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

def upload_to_s3(file_content, bucket_name, folder_name, file_name):
    s3 = boto3.client('s3')
    file_path = os.path.join(folder_name, file_name)
    try:
        s3.upload_fileobj(io.BytesIO(file_content), bucket_name, file_path)
        return True, f"File uploaded successfully to {bucket_name}/{file_path}"
    except Exception as e:
        return False, f"Error uploading file to S3: {e}"
    


def main():
    st.title("Audio Analysis App")

    # Add tabs
    tabs = ["Upload Audio", "Feedback", "Download Info"]
    selected_tab = st.radio("Select Tab", tabs)

    if selected_tab == "Upload Audio":
        st.subheader("Upload your audio file")
        uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/ogg', start_time=0)
            st.success("Audio file uploaded successfully!")
            
            bucket_name = "2307-01-acoustic-loggers-for-leak-detection-a"
            folder_name = "s3://2307-01-acoustic-loggers-for-leak-detection-a/App_Uploaded_Audios/"

                # Get the filename and content of the uploaded file
            file_name = uploaded_file.name
            file_content = uploaded_file.getvalue()

            # Upload the file to S3
            success, message = upload_to_s3(file_content, bucket_name, folder_name, file_name)
            if success:
                st.success(message)
            else:
                st.error(message)

    elif selected_tab == "Feedback":
        
            st.subheader("Audio Feedback")
            
            bucket_name = "2307-01-acoustic-loggers-for-leak-detection-a"
            folder_name = "App_Uploaded_Audios"

            # Download the audio file from S3
            ANNOTATIONS_FILE = annotation(bucket_name, folder_name)
            
            cnn = CNNNetwork()
            state_dict = torch.load("NeuralNetwork.pth")
            cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=64
            )

            usd = AcousticSoundsData(ANNOTATIONS_FILE,
                            #AUDIO_DIR,
                                    mel_spectrogram,
                                    22050,
                                    22050,
                                    "cpu",
                                    bucket_name=bucket_name)


    # get a sample from the acoustic sound dataset for inference
            input, target = usd[0][0], usd[0][1] # [batch size, num_channels, fr, time]
            input.unsqueeze_(0)
            predicted, _ = predict(cnn, input, target,
                                          class_mapping)
 
            st.write(f"Leak Status: {predicted}")


    elif selected_tab == "Download Info":
        
            st.subheader("Download Audio Information")
            bucket_name = "2307-01-acoustic-loggers-for-leak-detection-a"
            folder_name = "App_Uploaded_Audios"

            # Download the audio file from S3
            ANNOTATIONS_FILE = annotation(bucket_name, folder_name)
            
            cnn = CNNNetwork()
            state_dict = torch.load("NeuralNetwork.pth")
            cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=64
            )

            usd = AcousticSoundsData(ANNOTATIONS_FILE,
                            #AUDIO_DIR,
                                    mel_spectrogram,
                                    22050,
                                    22050,
                                    "cpu",
                                    bucket_name=bucket_name)


    # get a sample from the acoustic sound dataset for inference
            input, target = usd[0][0], usd[0][1] # [batch size, num_channels, fr, time]
            input.unsqueeze_(0)
            predicted, _ = predict(cnn, input, target,
                                          class_mapping)
            leak_status = predicted

            info = f"Duration: {leak_status} seconds\n" \
                 

            st.write(info)

            # Download button
            st.download_button(
                label="Download Audio Info",
                data=info.encode('utf-8'),
                file_name="audio_info.pdf",
                mime="pdf/plain"
            )

if __name__ == "__main__":
    main()