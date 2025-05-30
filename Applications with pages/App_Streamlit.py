
import streamlit as st
 
from pydub import AudioSegment
import os
import pandas as pd
import numpy as np

import torch
#import torchaudio
#from torchaudio.transforms import MelSpectrogram
#from AcousticSounds import AcousticSoundsData
#from cnn import CNNNetwork
#import io


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
    acoustic_alert_logo_url = "The acoustic alert logo"
    st.title("Audio Analysis App")
    st.sidebar.markdown('Welcome to AcousticAlert, Your ultimate leak detector, safeguarding your pipes against unseen threats')
    st.sidebar.image(acoustic_alert_logo_url, width=250, use_column_width=True)
    st.sidebar.markdown('    ')

    page_options = ["Home page", "Project summary", "Explore the Data", "Leak detector", "Meet the Team", "Contact Us", "Frequently Asked Questions"]
    page_selection = st.sidebar.selectbox("Choose Option", page_options)

    # Add tabs only under the "Leak detector" page
    if page_selection == "Leak detector":
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
                input, = acoustic_data[0][0] # [batch size, num_channels, fr, time]
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

    # Building our the "Home" page
    if page_selection == "Home page":
        homepage_image_url = "possible homepage"
        st.title("Welcome To AcousticAlert")
        st.markdown("***Where sound meets prevention: Leak detection made easy!***")
        st.image(homepage_image_url, width=250, use_column_width=True)

    # Building our "Project Summary" page
    if page_selection == "Project summary":
       st.title("Project summary")
       st.markdown("The project aims to address the pressing issue of undetected leaks in water distribution networks through the integration of acoustic loggers and machine learning algorithms.")
       st.markdown("Leaks in these networks result in significant water loss, financial burdens for utilities, infrastructure damages, environmental degradation, and public health risks.")
       st.markdown("The project aimed to address the pervasive issue of undetected leaks in water distribution networks by developing an innovative solution.")
       st.markdown("This solution involved the creation of a mobile application that empowers individuals to actively participate in leak detection efforts. Instead of relying solely on traditional methods or specialized equipment, the app allows users to upload audio recordings captured near water pipelines.")
       st.markdown("Leveraging advanced algorithms and machine learning techniques, the app analyzes these audio inputs to determine the presence of leaks within the distribution network.")
       st.markdown("Through this accessible and user-friendly approach, the solution aimed to empower individuals to contribute to water conservation efforts and promote the sustainable management of precious water resources.")

    # Bulding our the "Explore the Data" page
    if page_selection == "Explore the Data":
        st.title("Audio Analysis App")
        st.subheader ("**Exploratory Data Analysis**")
        st.image('Leaking wavefront',width= 250,use_column_width=True)
        st.image('No leak wavefront',width= 250,use_column_width=True)
        st.markdown("The two images above present waveforms of signals measured over time, with the first labeled Leaking Waveform and the second labeled No Leak Waveform.")
        st.markdown("There are significant differences between the two waveforms. The amplitude range of the Leaking Waveform is much broader, approximately spanning from -0.2 to 0.3, indicating more substantial signal activity and variability. In contrast, the No Leak Waveform has a narrower amplitude range, confined between about -0.05 and 0.05, suggesting a quieter signal with less noise or activity.")
        st.markdown("The noise level is notably different between the two waveforms. The Leaking Waveform exhibits higher overall amplitude variations, indicative of increased noise or signal disturbances. This is further evidenced by a prominent spike around the 2000-second mark, pointing to a significant anomaly or event. On the other hand, the No Leak Waveform shows lower amplitude variations, indicative of a more stable and consistent signal. Although there is a noticeable spike around the 3500-second mark in the No Leak Waveform, it is relatively smaller in magnitude compared to the overall amplitude range of the Leaking Waveform.")
        st.image('leaking',width= 250,use_column_width=True)
        st.image('no leaking',width= 250,use_column_width=True)
        st.markdown("The provided images display two waveforms, each representing signal data over a short duration, approximately from 0.4550 to 0.4725 seconds. The first waveform is labeled Leaking, and the second is labeled No Leaking.")
        st.markdown("The Leaking waveform demonstrates significant amplitude variations, with values ranging from approximately -0.04 to 0.12. This wide range of amplitude suggests the presence of disturbances or irregularities in the signal, typical of a leaking condition. The waveform features multiple peaks and troughs, indicating periodic and dynamic changes in amplitude.")
        st.markdown("In contrast, the No Leaking waveform displays much smaller amplitude variations, confined between approximately -0.015 and 0.015. This narrow amplitude range indicates a more stable and consistent signal environment, with less noise and fewer disturbances.")


   # Building out the "Meet the Team" page
    if page_selection == "Meet the Team":
        meet_the_team_image_path = "Meet the team"
        st.markdown("""
            ### Our Team
            Meet the amazing team behind our project. Our diverse and talented team members are dedicated to making a difference.   
        """)
        st.image(meet_the_team_image_path, width=250, use_column_width=True)


    if page_selection == "Contact Us":
        # URL to the raw image file on GitHub
        contacts_image_url = "Contact us - Copy"
        #st.title('Contact Us')
        st.image(contacts_image_url,width= 250,use_column_width=False)
        st.markdown("""
        ### Get in Touch
        We would love to hear from you. Here are the ways you can reach us:

        - **Tel:** 012 6730 391
        - **LinkedIn:** [AcuasticAlert](https://www.linkedin.com)
        - **Twitter:** [@AcuasticAlert](https://twitter.com/AcuasticAlert)
        - **Instagram:** [@AcuasticAlert](https://instagram.com/AcuasticAlert)
        - **Address:** 11 Adriana Cres, Rooihuiskraal, Centurion, 0154
    """)
        
    if page_selection == "Frequently Asked Questions":
        st.title("Frequently Asked Questions (FAQ)")

        # Define FAQ questions and answers
        faq_data = {
            "Q1. What is AcuasticAlert?":
                "AcuasticAlert is a project aimed at addressing the issue of undetected leaks in water distribution networks through the integration of acoustic loggers and machine learning algorithms.",
            
            "Q2. How does AcuasticAlert work?":
                "AcuasticAlert works by leveraging a mobile application that allows users to upload audio recordings captured near water pipelines. Advanced algorithms and machine learning techniques analyze these audio inputs to determine the presence of leaks within the distribution network.",
            
            "Q3. Why is leak detection important?":
                "Leak detection is crucial because undetected leaks in water distribution networks result in significant water loss, financial burdens for utilities, infrastructure damages, environmental degradation, and public health risks.",
            
            "Q4. How can I get involved with AcuasticAlert?":
                "You can get involved by using the AcuasticAlert mobile application to actively participate in leak detection efforts. Additionally, you can provide feedback, spread awareness about the project, or contribute to its development as a collaborator or supporter.",
            
            "Q5. Is AcuasticAlert free to use?":
                "Yes, AcuasticAlert is free to use. The mobile application is accessible to anyone who wants to contribute to water conservation efforts and promote the sustainable management of water resources."
        }

        # Display FAQ questions and answers
        for question, answer in faq_data.items():
            st.markdown(f"**{question}**")
            st.write(answer)
            st.write("")


        
if __name__ == "__main__":
    main()