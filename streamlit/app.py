from statistics import mean
import streamlit as st
import pandas as pd
import tensorboard
from streamlit_tensorboard import st_tensorboard
import os


def read_video(file):
    video_file = open(file, 'rb')
    return video_file.read()

w_type = {
    'Sunny': 'Clear',
    'Hard Rain':'Wet',
    'Wet Sunset': 'Cloudy'
}

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

st.text('The CycleGAN converts an image from domain A (Clear Weather) to domain B (Hard Rain or Wet Sunset)')
weather = st.radio(
     "Select Weather to Convert to",
     ('Hard Rain', 'Wet Sunset'))

ims_e = [os.path.join('./Images/'+w_type[weather]+'/early',image) for image in os.listdir('./Images/'+w_type[weather]+'/early')]
ims_l = [os.path.join('./Images/'+w_type[weather]+'/late',image) for image in os.listdir('./Images/'+w_type[weather]+'/late')]

st.subheader('CycleGAN after 2 rounds of Training')
e1,e2,e3,e4 = st.columns(4)

with e1:
    st.text('')
    st.image(ims_e[3])
with e2:
    st.image(ims_e[0])
with e3:
    st.image(ims_e[1])
with e4:
    st.image(ims_e[2])

st.subheader('CycleGAN after 30 rounds of Training')
l1,l2,l3,l4 = st.columns(4)

with l1:
    st.image(ims_l[3])
with l2:
    st.image(ims_l[0])
with l3:
    st.image(ims_l[1])
with l4:
    st.image(ims_l[2])



st.header('Driving across Multiple Weather Conditions')



weather1, weather2 = st.columns(2)

with weather1:
    option1 = st.selectbox('Agent 1 Weather Type',('Sunny','Hard Rain','Wet Sunset'),key=1)
    option2 = st.selectbox('Agent 1 AutoEncoder Type',('Sunny','Hard Rain','Wet Sunset'),key=2)


    st.video(read_video('./video/'+w_type[option1]+'_'+w_type[option2]+'.mp4'))
    st.dataframe(pd.read_csv('./model_results/FullModel/'+w_type[option1]+'_'+w_type[option2]+'_right_turn.csv').drop(['Unnamed: 0'],axis=1).agg({'Reward':mean, 'Length':mean, 'Completed':sum}))

with weather2:
    option3 = st.selectbox('Agent 2 Weather Type',('Sunny','Hard Rain','Wet Sunset'),key=3)
    option4 = st.selectbox('Agent 2 AutoEncoder Type',('Sunny','Hard Rain','Wet Sunset'),key=4)

    st.video(read_video('./video/'+w_type[option3]+'_'+w_type[option4]+'.mp4'))
    st.dataframe(pd.read_csv('./model_results/FullModel/'+w_type[option3]+'_'+w_type[option4]+'_right_turn.csv').drop(['Unnamed: 0'],axis=1).agg({'Reward':mean, 'Length':mean, 'Completed':sum}))

st.header('TensorBoard Results')

st_tensorboard(logdir='./tensorboard_res')
