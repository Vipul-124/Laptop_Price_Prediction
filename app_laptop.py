import pandas as pd
import numpy as np
import streamlit as st
from sklearn import *
import pickle

df = pickle.load(open('data1.pkl', 'rb'))
pipe_rf = pickle.load(open('rf1.pkl', 'rb'))

st.title('Laptop Price Predictor')

st.header('Fill the details to predict the Laptop price')


# Company - drop down
company = st.selectbox('Brand', df['Company'].unique())
# Typename - drop down
type = st.selectbox('Type', df['TypeName'].unique())
# Ram - drop down
ram = st.selectbox('Ram(in GB)', df['Ram'].unique())
# Weight - number input
weight = st.number_input('Weight of the Laptop')
# Touch Screen
touch = st.selectbox('Touch Screen', ['No', 'Yes'])
# IPS
ips = st.selectbox('IPS Display', ['No', 'Yes'])
# CPU Brand
cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())
# HDD
hdd = st.selectbox('HDD', df['HDD'].unique())
# SSD
ssd = st.selectbox('SSD', df['SSD'].unique())
# GPU Brand
gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
# OS
os = st.selectbox('OS', df['os'].unique())


if st.button('Predict Laptop Price') :
    if touch == "Yes" :
        touch=1
    else :
        touch=0
    if ips == "Yes" :
        ips = 1
    else :
        ips = 0
    test_data = np.array([company, type, ram, weight, touch, ips, cpu, hdd, ssd, gpu, os])
    test_data = test_data.reshape([1,11])

    st.success(pipe_rf.predict(test_data)[0])