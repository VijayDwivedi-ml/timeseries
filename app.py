#import warnings
#warnings.filterwarnings('ignore')  # Hide warnings
import datetime as dt
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator

##################
# Set up sidebar #
##################

# Add in location to select image.

option = st.sidebar.selectbox('symbol', ('ONMOBILE', 'GATI'))

st.write((option))


##############
# Stock data #
##############


df = pd.read_csv('data/ONMOBILE.csv')

df.reset_index(inplace=True)
df.set_index("Date", inplace=True)

indicator_bb = BollingerBands(df['Close'])

bb = df
bb['bb_h'] = indicator_bb.bollinger_hband()
bb['bb_l'] = indicator_bb.bollinger_lband()
bb = bb[['Close','bb_h','bb_l']]

macd = MACD(df['Close']).macd()

rsi = RSIIndicator(df['Close']).rsi()


###################
# Set up main app #
###################
n=100
st.write('Stock Bollinger Bands')

st.line_chart(bb.tail(n))

progress_bar = st.progress(0)

st.write('Stock Moving Average Convergence Divergence (MACD)')
st.area_chart(macd.tail(n))

st.write('Stock RSI ')
st.line_chart(rsi.tail(n))


st.write('Recent data ')
st.dataframe(df.tail(n))
