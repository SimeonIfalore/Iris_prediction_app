import streamlit as st
import sklearn
import pickle
import pandas as pd
import numpy as np


iris_data = pickle.load(open("irismodel.sav", 'rb'))

st.title('Iris Data prediction app')
#adding images
from PIL import Image
setosa = Image.open("iris_setosa.jpg")
virginica = Image.open("Iris_virginica.jpg")
versicolor = Image.open("iris_versicolor.jpg")

def user_report():
  sepal_length = st.sidebar.slider('sepal.length', 4.3, 10.0, 0.1)
  sepal_width = st.sidebar.slider('sepal.width',2.0 ,10.0, 0.1 )
  petal_length = st.sidebar.slider('petal.length', 1.0,10.0, 0.1 )
  petal_width = st.sidebar.slider('petal.width', 0.1,10.0, 0.1 )


  user_report_data= {
      'sepal.length' : sepal_length,
      'sepal.width':sepal_width,
      'petal.length': petal_length,
      'petal.width': petal_width
  }
  user_report_data = pd.DataFrame(user_report_data, index=[0])
  return user_report_data

user_data = user_report()
st.header('Iris data')
st.write(user_data)

iris = iris_data.predict(user_data)
st.subheader('iris prediction')

if (iris ==  0):
    st.image(setosa, width = 350,  caption='Setosa')
elif (iris == 1 ):
   st.image(versicolor, width = 350,  caption='Versicolor')
else:
    st.image(virginica, width = 350,  caption='Virginica')
