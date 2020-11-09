# Global Libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model
import plotly.express as px
from PIL import Image
import base64
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Charge model
et_model = load_model('et_model') 
# Global dataset
df = pd.read_csv('final_df.csv')
# Template dataset
df_template = pd.read_csv('sources/default_template.csv')
# Agregar los stopwords
stop_words_eng = set(stopwords.words('english')) 
stop_words_sp = set(stopwords.words('spanish')) 
final_stop_words = set(list(stop_words_eng) + list(stop_words_sp))

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
    if predictions_df.shape[0] > 1:
      return predictions_df['Label'], predictions_df['Score']
    else:
      return predictions_df['Label'][0], predictions_df['Score'][0]
  

def transform_data_online(name, hours):
  """ 
    Method to build a new input_df with NLP table included into dataframe.
  """

  ## Add new line 
  new_row = {'name': name, 'hours': hours}
  df_proc = df.append(new_row, ignore_index=True)
  df_proc['length'] = df_proc.name.str.len()

  # Embedding name
  result_df = vectorizer(df_proc)
  result_df = result_df.tail(1)

  # Adding hours
  result_df['hours'] = df_proc.tail(1).hours
  result_df['length'] = df_proc.tail(1).length
  
  # Reset index
  result_df = result_df.reset_index()
  return result_df

def transform_data_batch(input_df):
  """ 
    Method to build a new input_df with NLP table included into dataframe.
  """
  input_rows = input_df.shape[0]
  ## Add df into origin
  input_df['length'] = input_df.name.str.len()
  df_proc = df.append(input_df, ignore_index=True)
  
  # Embedding name
  result_df = vectorizer(df_proc)
  result_df = result_df.tail(input_rows)

  # Adding hours
  result_df['hours'] = df_proc.tail(input_rows).hours
  result_df['length'] = df_proc.tail(input_rows).length

  # Reset index
  result_df = result_df.reset_index()
  return result_df

def vectorizer(df):
  """ 
    Method that return dataframe vectorized
  """
  vectorizer_name = TfidfVectorizer(stop_words=final_stop_words)
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

def show_image(src, title = None):
  """ Show image using a src """
  image = Image.open(src)
  st.image(image, caption= title, use_column_width=True)

def get_table_download_link(df, title):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="'+title+'.csv">Descargar archivo csv</a>'
    return href
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
    
      # Menu option
    menu_selectbox_prediction = st.selectbox(
    "Tipo de predicción: ",
    ("Online", "Batch"))
    if menu_selectbox_prediction == "Online":
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
          input_df = transform_data_online(name, hours)
        
          output_prediction, value = predict(input_df=input_df)
          output_prediction_text = ''
          if output_prediction == 0:
            output_prediction_text = '(no se utilizará)'
          else:
            output_prediction_text = '(se utilizará)'

          # Show output
          st.success('La predicción es {} {} con un valor de {}'.format(output_prediction,output_prediction_text,value))
    else:
      
      # Download csv template
      show_template = st.checkbox("Mostrar template del CSV")
      if show_template:
        st.write("#### Ejemplo del CSV: ")
        st.write(df_template)
        st.markdown(get_table_download_link(df_template,'Template'), unsafe_allow_html=True)

      # Upload file
      st.set_option('deprecation.showfileUploaderEncoding', False) 
      file_upload = st.file_uploader("Cargue el archivo csv para predecir", type=['csv'])

      # Make Prediction
      if st.button("Predecir") and file_upload is not None:
        data = pd.read_csv(file_upload)
        input_df = transform_data_batch(data)
        output_prediction, value = predict(input_df=input_df)
        data['prediction'] = output_prediction
        data['score'] = value
        st.write(data)
        # Download result option
        st.markdown(get_table_download_link(data,'Resultado'), unsafe_allow_html=True)
        # Reset cache
        file_upload = None

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
      ("Top 50 palabras", "Word Cloud","Distribución de horas","Importancia de las características"))
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
    elif menu_graphs_selectbox == "Importancia de las características":
      with st.spinner('⏳ Creando...'):
        show_image('sources/feature_importance.png')
if __name__ == '__main__':
  run()