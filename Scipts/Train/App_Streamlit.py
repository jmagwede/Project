
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


class_mapping = [
    "Not Leaking",
    "Leaking"
]

def upload_to_s3(file_content, bucket_name, folder_name, file_name):
    s3 = boto3.client('s3')
    file_path = os.path.join(folder_name, file_name)
    try:
        s3.upload_fileobj(io.BytesIO(file_content), bucket_name, file_path)
        return True, f"File uploaded successfully to {bucket_name}/{file_path}"
    except Exception as e:
        return False, f"Error uploading file to S3: {e}"
    

def download_from_s3(bucket_name, folder_name):
    
    aws_access_key_id = 'AKIATNJHRXAPQBHVQARV'
    aws_secret_access_key = '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'
    region_name = 'eu-west-1'


    s3 = boto3.client('s3', 
                    aws_access_key_id=aws_access_key_id, 
                    aws_secret_access_key=aws_secret_access_key, 
                    region_name=region_name)
    
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{folder_name}/")
        if 'Contents' in response:
            file_name = response['Contents'][0]['Key'].split('/')[-1]
            response = s3.get_object(Bucket=bucket_name, Key=f"{folder_name}/{file_name}")
            key = f"{folder_name}/{file_name}"
            return key
        else:
            print("No objects found in the specified folder.")
            return None
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        return None
    

def predict(model, input, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
    return predicted

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
            key = download_from_s3(bucket_name, folder_name)
            if key is None:
                print("Error: Failed to download audio file from S3.")
                return
            
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
            
            # Process the audio using the AcousticSoundsData class

            input = acoustic_data[0][0] # [batch size, num_channels, fr, time]
            input.unsqueeze_(0)
            
            cnn = CNNNetwork()
            state_dict = torch.load("NeuralNetwork.pth")
            cnn.load_state_dict(state_dict)
            
            predicted = predict(cnn, input, class_mapping)
            
            st.write(f"Leak Status: {predicted}")


    elif selected_tab == "Download Info":
        
            st.subheader("Download Audio Information")
            leak_status = predicted

            info = f"Duration: {predicted} seconds\n" \
                 

            st.write(info)

            # Download button
            st.download_button(
                label="Download Audio Info",
                data=info.encode('utf-8'),
                file_name="audio_info.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()