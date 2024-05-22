import streamlit as st
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

def predict(model, input, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
    return predicted

def upload_to_s3(file_content, bucket_name, folder_name, file_name):
    # Implementation of upload_to_s3 function

    def download_from_s3(bucket_name, folder_name):
    # Implementation of download_from_s3 function

        def main():
            st.title("Audio Analysis App")

    page_options = ["Home page", "Project summary", "Explore the Data", "Leak detector", "Meet the Team", "Contact Us", "Frequently Asked Questions"]
    page_selection = st.sidebar.selectbox("Choose Option", page_options)

    if page_selection == "Leak detector":
        tabs = ["Upload Audio", "Feedback", "Download Info"]
        selected_tab = st.radio("Select Tab", tabs)
        
        if selected_tab == "Upload Audio":
            st.subheader("Upload your audio file")
            uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

            if uploaded_file is not None:
                st.audio(uploaded_file, format='audio/ogg', start_time=0)
                st.success("Audio file uploaded successfully!")
                
                bucket_name = "your_bucket_name"
                folder_name = "App_Uploaded_Audios"

                file_name = uploaded_file.name
                file_content = uploaded_file.getvalue()

                success, message = upload_to_s3(file_content, bucket_name, folder_name, file_name)
                if success:
                    st.success(message)
                else:
                    st.error(message)

        elif selected_tab == "Feedback":
            st.subheader("Audio Feedback")

            bucket_name = "your_bucket_name"
            folder_name = "App_Uploaded_Audios"

            key = download_from_s3(bucket_name, folder_name)
            if key is None:
                st.error("Error: Failed to download audio file from S3.")
                return

            s3 = boto3.client('s3')
            audio_data_io = io.BytesIO()
            s3.download_fileobj(bucket_name, key, audio_data_io)
            audio_data_io.seek(0)

            audio_tensor, _ = torchaudio.load(audio_data_io)
            acoustic_data = AcousticSoundsData(audio_tensor,
                                               transformation=MelSpectrogram(sample_rate=2250,
                                                                            n_fft=1024,
                                                                            hop_length=512,
                                                                            n_mels=64),
                                                target_sample_rate=2250,
                                                num_samples=10000,
                                                device='cpu')

            input, = acoustic_data[0][0] 
            input.unsqueeze_(0)

            cnn = CNNNetwork()
            state_dict = torch.load("NeuralNetwork.pth")
            cnn.load_state_dict(state_dict)

            predicted = predict(cnn, input, class_mapping)

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
        homepage_image_url = "https://media.discordapp.net/attachments/1241491632197472327/1241492751841890385/possible_homepage.png?ex=664a65b9&is=66491439&hm=67d44b87ec6b6a5677711b21b89855f5d09a3f0a09330e380611386af6a1c68c&=&format=webp&quality=lossless&width=595&height=395"
        st.title("Welcome To AcuasticAlert")
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
        st.image('https://media.discordapp.net/attachments/1241491632197472327/1241498038082474034/Leaking_wavefront.PNG?ex=664a6aa6&is=66491926&hm=80cbea2c107f1a5f40c86e1706f4d47449d6fe5d6f6fc518d07ec90f5acbcd66&=&format=webp&quality=lossless',width= 250,use_column_width=True)
        st.image('https://media.discordapp.net/attachments/1241491632197472327/1241498102842785853/No_leak_wavefront.PNG?ex=664a6ab5&is=66491935&hm=9d455f1fe3d4abf86ab9f16f276844c6cc4349c19772c3708cc16cd254100b28&=&format=webp&quality=lossless',width= 250,use_column_width=True)
        st.image('https://media.discordapp.net/attachments/1241491632197472327/1241498068940230759/leaking.PNG?ex=664a6aad&is=6649192d&hm=ad1dbe9977de6a80e073be11e532b32fdb964dd71cf4551c3a05465a0cc65762&=&format=webp&quality=lossless&width=1025&height=319',width= 250,use_column_width=True)
        st.image('https://media.discordapp.net/attachments/1241491632197472327/1241498135235399852/no_leaking.PNG?ex=664a6abd&is=6649193d&hm=4ccd6ddd2c5c198c2e0de618193314232a7e5883ac610f12cf47c43fc5c04f74&=&format=webp&quality=lossless&width=1025&height=316',width= 250,use_column_width=True)
        
    
   # Building out the "Meet the Team" page
    if page_selection == "Meet the Team":
        meet_the_team_image_path = "https://media.discordapp.net/attachments/1241491632197472327/1241492779255988436/Meet_the_team.PNG?ex=664a65c0&is=66491440&hm=0bc20b31c1ac19da41b13f6f194fa1ed091cee8a72e48eca109ceec940559199&=&format=webp&quality=lossless"
        st.markdown("""
            ### Our Team
            Meet the amazing team behind our project. Our diverse and talented team members are dedicated to making a difference.   
        """)
        st.image(meet_the_team_image_path, width=250, use_column_width=True)


    if page_selection == "Contact Us":
        # URL to the raw image file on GitHub
        contacts_image_url = "https://media.discordapp.net/attachments/1241491632197472327/1241496533770436722/Contact_us_-_Copy.webp?ex=664a693f&is=664917bf&hm=3625380a4ef7d52c2417e39cb1c2ac59c20ebeef40c3ae1bb6dd09e1d910efa7&=&format=webp"
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