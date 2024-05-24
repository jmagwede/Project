from pydub import AudioSegment
import os
import streamlit as st
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from AppProcessor import SignalPrecessing
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
    

# A function to upload the audio to the s3 bucket
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
    page_options = ["Home page", "Project Summary", "Explore the Data", "Aploading Audio", "Meet the Team", "Contact Us", "Frequently Asked Questions"]
    page_selection = st.sidebar.selectbox("Choose Option", page_options)

    if page_selection == "Aploading Audio":
        tabs = ["Upload Audio", "Feedback", "Download Info"]
        selected_tab = st.radio("Select Tab", tabs)
        
        if selected_tab == "Upload Audio":
            st.subheader("Upload your audio file")
            #st.image('The acoustic alert logo.png',width= 250,use_column_width=True)
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
            state_dict = torch.load("500_Audio_Nn.pth")
            cnn.load_state_dict(state_dict)

           #load urban sound dataset dataset
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=64
            )

            usd = SignalPrecessing(ANNOTATIONS_FILE,
                            #AUDIO_DIR,
                                    mel_spectrogram,
                                    22050,
                                    22050,
                                    "cpu",
                                    bucket_name=bucket_name)


    # get a sample from the acoustic sound dataset for inference
            input, target = usd[0][0], usd[0][1] # [batch size, num_channels, fr, time]
            input.unsqueeze_(0)
            predicted, target = predict(cnn, input, target,
                                          class_mapping)
 
            st.write(f"Leak Status: {predicted}")

        elif selected_tab == "Download Info":
            st.subheader("Download Audio Information")

            # Assuming 'predicted' is defined elsewhere
            leak_status = predicted

            info = f"Duration: {predicted} seconds\n" \

            st.write(info)

            st.download_button(
                label="Download Audio Info",
                data=info.encode('utf-8'),
                file_name="audio_info.txt",
                mime="text/plain"
            )

    # Building our the "Home" page
    if page_selection == "Home page":
        st.title("Welcome To AcuasticAlert")
        st.markdown("***Where sound meets prevention: Leak detection made easy!***")
        st.image('The acoustic alert logo.png', width=250, use_column_width=True)

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
        st.image('Leaking wavefront.png',width= 250,use_column_width=True)
        st.image('No leak wavefront.png',width= 250,use_column_width=True)
        st.image('leaking.png',width= 250,use_column_width=True)
        st.image('no leaking.png',width= 250,use_column_width=True)
    
   # Building out the "Meet the Team" page
    if page_selection == "Meet the Team":
       
        st.markdown("""
            ### Our Team
            Meet the amazing team behind our project. Our diverse and talented team members are dedicated to making a difference.   
        """)
        st.image('Meet the team.png', width=250, use_column_width=True)


    if page_selection == "Contact Us":
        # URL to the raw image file on GitHub
        #st.title('Contact Us')
        st.image('contact us.jpeg',width= 250,use_column_width=False)
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