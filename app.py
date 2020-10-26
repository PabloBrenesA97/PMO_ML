# Global Libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model
import plotly.express as px

# Charge model
et_model = load_model('et_model') 
# Global dataset
df = pd.read_csv('final_df.csv')

def predict(input_df):
  """ 
    Method to predict if estimation will be add in a project 
    returns prediction value
  """
  # Start timer
  start_time = time.perf_counter()
  # Show Spinner
  with st.spinner('⏳ Pensando...'):
    # Predict
    predictions_df = predict_model(estimator=et_model, data=input_df)
    
    # Print timer
    st.info('Tiempo estimado de predicción: %.3fs' % (time.perf_counter() - start_time))

    return predictions_df['Label'][0], predictions_df['Score'][0]
  

def transform_data(name, hours):
  """ 
    Method to build a new input_df with NLP table included into dataframe.
  """

  ## Add new line 
  new_row = {'name': name, 'hours': hours}
  df_proc = df.append(new_row, ignore_index=True)
  
  # Embedding name
  result_df = vectorizer(df_proc)
  result_df = result_df.tail(1)

  # Adding hours
  result_df['hours'] = df_proc.tail(1).hours

  # Reset index
  result_df = result_df.reset_index()
  return result_df

def vectorizer(df):
  """ 
    Method that return dataframe vectorized
  """
  vectorizer_name = TfidfVectorizer()
  data_name = vectorizer_name.fit_transform(df.name)
  tfidf_tokens_name = vectorizer_name.get_feature_names()
  result_df = pd.DataFrame(data = data_name.toarray(),columns = tfidf_tokens_name)
  return result_df

def top_50_words():
  """ Visualization of top 50 words"""
  result_df = vectorizer(df)
  df_freq = result_df.T.sum(axis=1)
  y_pos = np.arange(50)
  fig, ax = plt.subplots()
  plt.bar(y_pos, df_freq.sort_values(ascending=False)[:50], align='center', alpha=0.5)
  plt.xticks(y_pos, df_freq.sort_values(ascending=False)[:50].index,rotation='vertical',fontsize=7)
  plt.ylabel('Frecuencia')
  st.pyplot(fig)

def distribution_hours_visualization():
  """ Visualization of dist hours"""
  fig = px.violin(df, y="hours", box=True,points='all', title ="Distribución de las horas de los items")
  st.plotly_chart(fig)

def word_cloud_visualization():
  """ Visualization of word cloud"""
  # Embedded
  result_df = vectorizer(df)
  # Create wordCloud
  Cloud = WordCloud( background_color="white",
    max_words=2000,
    width = 1024,
    height = 720,).generate_from_frequencies(result_df.T.sum(axis=1))
  # Plot
  fig, ax = plt.subplots()
  plt.imshow(Cloud)
  plt.axis('off')
  st.pyplot(fig)
  
def run():
  """ 
    Streamlit app
  """
  # Hidden some defautl streamlit components
  hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}    
  """
  st.markdown(hide_footer_style, unsafe_allow_html=True)

  # ===========================
  # Sidebar
  # ===========================
  
  # Title Menu
  st.sidebar.markdown("<h1 style='text-align: center;'>Menú</h1>", unsafe_allow_html=True)

  # Menu option
  menu_selectbox = st.sidebar.selectbox(
    "Elija una opción: ",
    ("Predicción", "Gráficos"))

  if menu_selectbox == "Predicción":
    # ===========================
    # Predict Section
    # ===========================
    st.write(""" 
      # Predicción del ítem del estimador - PMO

        Muestra la predicción sí **el ítem del estimador** va a ser usado en un proyecto usando las variables ***descripción y horas***.
      
      ## Complete los siguientes entradas para realizar la predicción:
    """)

    name = st.text_input("Nombre del ítem:")
    hours = st.number_input('Horas:', min_value=1, max_value=400, value=8)
    
    # Output text
    output=""
    
    # Make prediction
    if st.button("Predecir"):
      
      if name == "" or str(hours) == "":
        # Show alert white spaces
        #TODO: 
        st.error('No pueden haber espacios en blanco!')
      else:
        # Get table with NLP
        input_df = transform_data(name, hours)
      
        output_prediction, value = predict(input_df=input_df)
        output_prediction_text = ''
        if output_prediction == '0':
          output_prediction_text = '(no se utilizará)'
        else:
          output_prediction_text = '(se utilizará)'

        # Show output
        st.success('La predicción es {} {} con un valor de {}'.format(output_prediction,output_prediction_text,value))

  else:
    # ===========================
    # Graph Section
    # ===========================
    # Title graph
    st.write(""" 
      # Módulo Analítico - Gráficos
        Muestra gráficos analíticos los cuales revelan ciertos patrones ocultos en los datos.
    """)
    # Menu graphs options
    menu_graphs_selectbox = st.selectbox(
      "Elija la visualización: ",
      ("Word Cloud", "Top 50 palabras","Distribución de horas"))
    # Option word cloud
    if menu_graphs_selectbox == "Word Cloud":
      # Show Spinner
      with st.spinner('⏳ Creando...'):
        word_cloud_visualization()
    elif menu_graphs_selectbox == "Top 50 palabras":
      with st.spinner('⏳ Creando...'):
        top_50_words()
    elif menu_graphs_selectbox == "Distribución de horas":
      with st.spinner('⏳ Creando...'):
        distribution_hours_visualization()
if __name__ == '__main__':
  run()