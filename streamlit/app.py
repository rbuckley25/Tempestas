import streamlit as st
import pandas as pd


def read_video(file):
    video_file = open(file, 'rb')
    return video_file.read()

st.title('Learning to Drive in Adverse Weather Conditions Using Deep Learning: Autoencoder & GAN Approaches')

st.header('Waypoint Driving Vs Latent Space Driving')

waypoint_col, latent_col = st.columns(2)

with waypoint_col:
    st.write("Waypoint Model and Standard CARLA Camera")
    
    st.video(read_video('./video/W_Full_Turn.mp4'))


with latent_col:
    st.write("Waypoint Model and Standard CARLA Camera")
    
    st.video(read_video('./video/Clear_Clear.mp4'))


st.header('CycleGAN')


st.header('Driving across Multiple Weather Conditions')


w_type = {
    'Sunny': 'Clear',
    'Hard Rain':'Wet',
    'Wet Sunset': 'Cloudy'
}
weather1, weather2 = st.columns(2)

with weather1:
    option1 = st.selectbox('Select Weather Type',('Sunny','Hard Rain','Wet Sunset'),key=1)
    option2 = st.selectbox('Select AutoEncoder Type',('Sunny','Hard Rain','Wet Sunset'),key=2)


    st.video(read_video('./video/'+w_type[option1]+'_'+w_type[option2]+'.mp4'))

with weather2:
    option3 = st.selectbox('Select Weather Type',('Sunny','Hard Rain','Wet Sunset'),key=3)
    option4 = st.selectbox('Select AutoEncoder Type',('Sunny','Hard Rain','Wet Sunset'),key=4)


    st.video(read_video('./video/'+w_type[option3]+'_'+w_type[option4]+'.mp4'))
