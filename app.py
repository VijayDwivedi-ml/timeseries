import streamlit as st
import pandas as pd
import numpy as np
import sqlite3

#Create a SQL connection to our SQLite database
con = sqlite3.connect("chemistry.db")
df = pd.read_sql_query('SELECT * from chemistry', con)
con.close()

st.title('Product Identification based on Reactant based of Chemical Reaction')
st.sidebar.title('Chemistry Project by Vedang Dwivedi, XII')
st.sidebar.image(https://github.com/VijayDwivedi-ml/timeseries/blob/main/vedang_pic_final.JPG, width = 100)

select1 = st.sidebar.selectbox('Reactant Selection', ['Alkyne','Alkene','Alkanes','Alcohol','Alcohol(Secondary)','Aldehyde','Ketone','Carboxylic Acid','Ether','Amine', ' '])
st.write('You Selected Reactant:', select1)
st.write('\n')
st.write('\n')
st.write('\n')
df2 = df[df['Reactant1'] == select1]
st.write(df2)

st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')


select2 = st.sidebar.selectbox('Process-Reagent Selection', ['Hydrogenation','Halogenation','Ozonolysis','Diboration','Dedydration','Oxidation','Grignard Reagent','Tollens Reagent','Fehlings', 'Reagent', ' '])
st.write('You Selected Reactant and Process-Reagent:', select1, ',',select2)
st.write('\n')
st.write('\n')
st.write('\n')
df3 = df2[df2['Process_Reagent'] == select2]
st.write(df3)

st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')

select3 = st.sidebar.selectbox('Product Selection', ['Alkene', 'Halo Alkane', 'Aldehyde', 'Alcohol', 'Alkane',' '])
st.write('You Selected Reactant, Process-Reagent and Product:', select1, ',', select2, ', ', select3)
st.write('\n')
st.write('\n')
st.write('\n')
df4 = df3[df3['Product1'] == select3]
st.write(df4)

st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')


st.write("##################################################################")





