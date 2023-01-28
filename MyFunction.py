import streamlit as st
#from PIL import Image
import requests
import pandas as pd
import numpy as np
#import sqlite3

# Weather data for any city

def find_current_weather(city):
    API_KEY = "642b399f532d6a45a46c3a8c83d0a18e"
    base_url  = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    weather_data = requests.get(base_url).json()
#   st.json(weather_data)
    lat = weather_data['coord']['lat']
    lon = weather_data['coord']['lon']
    base_url2 = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    weather_data2 = requests.get(base_url2).json()
#    st.json(weather_data2)
    
    try:
        general = weather_data['weather'][0]['main']
        icon_id = weather_data['weather'][0]['icon']
        temperature = round(weather_data['main']['temp'])
        humidity = round(weather_data['main']['humidity'])
        icon = f"http://openweathermap.org/img/wn/{icon_id}@2x.png"
     #   lat = weather_data['coord']['lat']
     #   lon = weather_data['coord']['lon']
        aqi = weather_data2['list'][0]['main']['aqi']
        AQI = 0
        if aqi == 1:
            AQI = 'Good'
        elif aqi == 2:
            AQI = 'Fair'
        elif aqi == 3:
            AQI = 'Moderate'
        elif aqi == 4:
            AQI = 'Poor'
        elif aqi == 5:
            AQI = 'Very Poor'
            
        
    except KeyError:
        st.error("City Not Found")
        st.stop()
    return general,temperature, humidity, icon, AQI
