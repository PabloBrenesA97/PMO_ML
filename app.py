# Global Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
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
    # Import library
    from pycaret.classification import load_model, predict_model
  
    # Charge model
    model = load_model('et_model_pmo_v1') 
    
    # Predict
    predictions_df = predict_model(estimator=model, data=input_df)
    
    # Print timer
    st.info('Tiempo estimado de predicción: %.3fs' % (time.perf_counter() - start_time))

    return predictions_df['Label'][0], predictions_df['Score'][0]
  

def transform_data(name, hours):
  """ 
    Method to build a new input_df with NLP table included into dataframe.
  """
  # Import 
  from sklearn.feature_extraction.text import TfidfVectorizer

  ## Add new line 
  new_row = {'name': name, 'hours': hours}
  df_proc = df.append(new_row, ignore_index=True)
  
  # Embedding name
  result_df = vectorizer(df)
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

        if output_prediction == '0':
          output_prediction = '(no se utilizará)'
        else:
          output_prediction = '(se utilizará)'

        # Show output
        st.success('La predicción es {} con un valor de {}'.format(output_prediction,value))

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
      ("Word Cloud", "X"))
    # Option word cloud
    if menu_graphs_selectbox == "Word Cloud":
      # Show Spinner
      with st.spinner('⏳ Creando...'):
        word_cloud_visualization()

if __name__ == '__main__':
  run()