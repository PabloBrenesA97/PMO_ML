# Global Libraries
import streamlit as st
import pandas as pd
import numpy as np

df_master = pd.read_csv('final_df.csv')

def predict(input_df):
  """ 
    Method to predict if estimation will be add in a project 
    returns prediction value
  """
  # Import library
  from pycaret.classification import load_model, predict_model
 
  # Charge model
  model = load_model('lightgbm_model_pmo_v1') 

  # Predict
  predictions_df = predict_model(estimator=model, data=input_df)
  
  return predictions_df['Label'][0], predictions_df['Score'][0]

def get_NLP_table(input_df):
  """ 
    Method to build a new input_df with NLP table included into dataframe.
  """
  # Import library
  from pycaret.nlp import setup, create_model, assign_model
  #intialize the setup
  exp_nlp = setup(data = input_df, target = 'name', session_id=7,  custom_stopwords = ['project', 'proyecto'])
  # create a lda model
  lda = create_model('lda')
  # label the data using trained model
  lda_df = assign_model(lda)
  lda_df = lda_df.drop(['Dominant_Topic','Perc_Dominant_Topic','name','is_used'], axis=1)
  st.write(""" ### Processed Table:""")
  st.dataframe(lda_df.tail(1))
  return lda_df.tail(1)

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
    bucket = st.selectbox('Bucket:', ['AI', 'API', 'Admin Site', 'Android', 'Architech', 'Backend',
       'Devops', 'Feedback', 'Fullstack web app', 'Hybrid App',
       'Informative Website', 'PM', 'QA', 'SPA', 'Software Development',
       'iOS'])
    
    # Output text
    output=""
    
     # Build dataframe
    input_dict = {'name': name, 'hours': hours, 'bucket' : bucket}
    input_df = pd.DataFrame([input_dict])
    final_df = df_master.append(input_df, ignore_index = True)
    # Show table
    st.write(""" ### Output Table:""")
    st.dataframe(input_df)

    # Make prediction
    if st.button("Predict"):

      # Get table with NLP
      df = get_NLP_table(final_df)

      df.reset_index(drop=True, inplace=True)

      output, value = predict(input_df=df)
      output = str(output)
      
      #st.dataframe(df)
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