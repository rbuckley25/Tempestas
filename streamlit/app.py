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

st.header('Driving with Waypoints Vs Driving with Image Data')

st.text('The below vides show examples of two agents, one trained to just follow the nearest waypoint and the other to follow the nearest one point and make use of image data')
st.text('While both models are able to make the turn it is clear that the image based model makes the turn much more smoothly')

waypoint_col, latent_col = st.columns(2)

with waypoint_col:
    st.write("Waypoint Model and CARLA Semtantic Camera")
    
    st.video(read_video('./video/W_Full_Turn.mp4'))


with latent_col:
    st.write("Image based Model and Trained Semantic Camera")
    
    st.video(read_video('./video/Clear_Clear.mp4'))


st.header('CycleGAN')

st.text('The CycleGAN converts an image from domain A (Clear Weather) to domain B (Hard Rain or Wet Sunset) as well as the reverse')
st.text('Below are a number of examples of the translation from both domains and the difference between early training and final training results')
weather = st.radio(
     "Select Weather to Convert to",
     ('Hard Rain', 'Wet Sunset'))

ims_e = [os.path.join('./Images/'+w_type[weather]+'/early',image) for image in os.listdir('./Images/'+w_type[weather]+'/early')]
ims_l = [os.path.join('./Images/'+w_type[weather]+'/late',image) for image in os.listdir('./Images/'+w_type[weather]+'/late')]
ims_e.sort()
ims_l.sort()


st.subheader('CycleGAN Results after 2 rounds of Training')
h1,h2 = st.columns(2)
e1,e2,e3,e4 = st.columns(4)
with h1:
    st.text('Image from A to B')
with e1:
    st.image(ims_e[2])
with e2:
    st.image(ims_e[1])
with h2:
    st.text('Image from B to A')
with e3:
    st.image(ims_e[3])
with e4:
    st.image(ims_e[0])

st.subheader('CycleGAN Results after 30 rounds of Training')
h3,h4 = st.columns(2)
l1,l2,l3,l4 = st.columns(4)

with h3:
    st.text('Image from A to B')
with l1:
    st.image(ims_l[2])
with l2:
    st.image(ims_l[1])
with h4:
    st.text('Image from B to A')
with l3:
    st.image(ims_l[3])
with l4:
    st.image(ims_l[0])



st.header('Driving across Multiple Weather Conditions')
st.text('The below section allows you to see the agent perform across a number of weather conditions as well as the effect the different Perception Modules have on agents ability to perform')
st.text('Each Perception Module has been trained using Transfer Learning based of the orginal Module (Sunny)')

weather1, weather2 = st.columns(2)

with weather1:
    option1 = st.selectbox('Agent 1 Weather Type',('Sunny','Hard Rain','Wet Sunset'),key=1)
    option2 = st.selectbox('Agent 1 AutoEncoder Type',('Sunny','Hard Rain','Wet Sunset'),key=2)


    st.video(read_video('./video/'+w_type[option1]+'_'+w_type[option2]+'.mp4'))
    df1 = pd.read_csv('./model_results/FullModel/'+w_type[option1]+'_'+w_type[option2]+'_right_turn.csv').drop(['Unnamed: 0'],axis=1).agg({'Reward':mean, 'Length':mean, 'Completed':sum}).round(2)


base = pd.read_csv('./model_results/FullModel/Clear_Clear_right_turn.csv').drop(['Unnamed: 0'],axis=1).agg({'Reward':mean, 'Length':mean, 'Completed':sum}).round(2)

with weather2:
    option3 = st.selectbox('Agent 2 Weather Type',('Sunny','Hard Rain','Wet Sunset'),key=3)
    option4 = st.selectbox('Agent 2 AutoEncoder Type',('Sunny','Hard Rain','Wet Sunset'),key=4)

    st.video(read_video('./video/'+w_type[option3]+'_'+w_type[option4]+'.mp4'))
    df2 = pd.read_csv('./model_results/FullModel/'+w_type[option3]+'_'+w_type[option4]+'_right_turn.csv').drop(['Unnamed: 0'],axis=1).agg({'Reward':mean, 'Length':mean, 'Completed':sum}).round(2)

st.subheader('Metrics (With Difference from Baseline)')
metric1,metric2,metric3,metric4,metric5,metric6 = st.columns(6)
with metric1:
    st.metric(label="Avg Reward", value=str(df1[0]), delta=str(df1[0]-base[0]))

with metric2:
    st.metric(label="Avg Length", value=str(df1[1]), delta=str(df1[1]-base[1]))

with metric3:
    st.metric(label="Completed ?/10", value=str(df1[2]), delta=str(df1[2]-base[2]))

with metric4:
    st.metric(label="Avg Reward", value=str(df2[0]), delta=str(df2[0]-base[0]))

with metric5:
    st.metric(label="Avg Length", value=str(df2[1]), delta=str(df2[1]-base[1]))

with metric6:
    st.metric(label="Completed ?/10", value=str(df2[2]), delta=str(df2[2]-base[2]))

st.header('TensorBoard Results')

st_tensorboard(logdir='./tensorboard_res')
