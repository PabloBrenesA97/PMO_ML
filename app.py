# Global Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Global dataset
df = pd.read_csv('final_df.csv')

@st.cache
def predict(input_df):
  """ 
    Method to predict if estimation will be add in a project 
    returns prediction value
  """
  # Import library
  from pycaret.classification import load_model, predict_model
 
  # Charge model
  model = load_model('et_model_pmo_v1') 
  
  # Predict
  predictions_df = predict_model(estimator=model, data=input_df)
  
  return predictions_df['Label'][0], predictions_df['Score'][0]

@st.cache
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
  vectorizer_name = TfidfVectorizer()
  data_name = vectorizer_name.fit_transform(df_proc.name)
  tfidf_tokens_name = vectorizer_name.get_feature_names()
  result_df = pd.DataFrame(data = data_name.toarray(),columns = tfidf_tokens_name)
  result_df = result_df.tail(1)

  # Adding hours
  result_df['hours'] = df_proc.tail(1).hours

  # Reset index
  result_df = result_df.reset_index()
  return result_df

def run():
  """ 
    Streamlit app
  """
  # Hidden some defautl streamlit components
  hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}
    #MainMenuButton {visibility: hidden;}    
  """
  st.markdown(hide_footer_style, unsafe_allow_html=True)

  # ===========================
  # Sidebar
  # ===========================
  
  # Title
  st.sidebar.markdown("<h1 style='text-align: center;'>Menu</h1>", unsafe_allow_html=True)

  # Menu option
  menu_selectbox = st.sidebar.selectbox(
    "Choose an option: ",
    ("Predict", "Graphs"))

  if menu_selectbox == "Predict":
    # ===========================
    # Predict Section
    # ===========================
    st.write(""" 
      # Predict estimation item - PMO

      Show prediction if **estimation** would be used in a project using ***description, hours, and bucket***.
      
      ## Complete the inputs to make a prediction:
    """)

    name = st.text_input("Name item:")
    hours = st.number_input('Hours:', min_value=1, max_value=400, value=8)
    
    # Output text
    output=""
    
    # Make prediction
    if st.button("Predict"):

      # Get table with NLP
      input_df = transform_data(name, hours)
    
      output, value = predict(input_df=input_df)

      # Show output
      if  output != "":  
        st.success('La predicción es {} con un valor de {}'.format(output,value))

  else:
    # ===========================
    # Graph Section
    # ===========================
    st.write("""
    ## In progress ...
    ![alt text](https://i.gifer.com/3jnq.gif "Working ...")
    """)

if __name__ == '__main__':
  run()