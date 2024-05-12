
import streamlit as st
 
from pydub import AudioSegment
import os

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from AcousticSounds import AcousticSoundsData
from cnn import CNNNetwork
import io


class_mapping = [
    "Not Leaking",
    "Leaking"
]

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
            
            

    elif selected_tab == "Feedback":
        if uploaded_file is not None:
            st.subheader("Audio Feedback")
            
            audio_tensor, _ = torchaudio.load(uploaded_file)
            acoustic_data = AcousticSoundsData(audio_tensor,
                                               transformation=MelSpectrogram(sample_rate=2250,
                                                                            n_fft=1024,
                                                                            hop_length=512,
                                                                            n_mels=64),
                                                target_sample_rate=2250,
                                                num_samples=10000,
                                                device='cpu')
            
            # Process the audio using the AcousticSoundsData class

            input,= acoustic_data[0][0] # [batch size, num_channels, fr, time]
            input.unsqueeze_(0)
            
            cnn = CNNNetwork()
            state_dict = torch.load("NeuralNetwork.pth")
            cnn.load_state_dict(state_dict)
            
            predicted = predict(cnn, input, class_mapping)
            
            st.write(f"Leak Status: {predicted}")


    elif selected_tab == "Download Info":
        if uploaded_file is not None:
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