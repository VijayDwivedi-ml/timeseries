import streamlit as st
import pandas as pd
import numpy as np
#import sqlite3
#from PIL import Image
import requests
from MyFunction import * 

# Web page basic configuration
st.set_page_config(
    page_title="Asthma Management App",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# # Author details for contacting him
# col1,col2 = st.columns(2)
# with col1:
    # st.sidebar.image("IMG/VIJAY_DWIVEDI.jpg", width = 80)
# with col1:
    # st.sidebar.markdown('<a href="mailto:vijay.dwivedi@ge.com"> 📧 Vijay Dwivedi</a>', unsafe_allow_html=True)


# Main page basic details

st.title("Asthma Management App 📱️")
st.sidebar.title('Please Provide Details')

select1 = st.sidebar.selectbox(' 🧒 👨 👴', ['19-30','31-40', '41-50', 'Above 50'])

select2 = st.sidebar.selectbox('🧑‍🤝‍🧑', ['Male','Female'])

select3 = st.sidebar.selectbox('🚬, 🍷', ['Yes','No', 'Either'])

select4 = st.sidebar.selectbox('🍲, 💊', ['Yes','No'])

select5 = st.sidebar.selectbox('Asthma medicine missed', ['Yes','No'])

select5 = st.sidebar.selectbox('When you have taken last medicine', ['Yesterday','One week back', 'Two Week Back'])
 
# Main function for the app
 
def main():
    st.sidebar.header("The Weather ☁️")
    city = st.sidebar.text_input("Enter the City", 'Bengaluru' ).lower()
    if st.sidebar.button("Find"):
        general,temperature, humidity, icon, AQI = find_current_weather(city)
        col_1,col_2, col_3 = st.columns(3)
        with col_1:
            st.metric(label = "Temperature 🌡️",value=f"{temperature}°C", delta_color="normal")
        with col_2:
            st.metric(label = "Humidity 🚿",value=f"{humidity}%")
        with col_3:
            st.write('Weather ☁️: ', general)
            st.image(icon)
        # with col_4:
            # st.metric(label = "AQI 😷 ",value=f"{AQI}")
        # df2 = DataBase()[DataBase()['Age'] == select1]
        # df3 = df2[df2['Gender'] == select2]
        # df4 = df3[df3['SmokingHabit'] == select3]
        # st.write(df4.head())
        st.markdown("<h2 style='text-align: center; color: lightgreen;'>Thank You for Using the App</h2>", unsafe_allow_html=True)
#        st.balloons()

    
if __name__ == '__main__':
    main()
