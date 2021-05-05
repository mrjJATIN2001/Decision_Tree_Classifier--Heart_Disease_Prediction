import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_option('deprecation.showfileUploaderEncoding', False)

# Load the pickled model
model = pickle.load(open('decision_tree_model.pkl', 'rb')) 
dataset= pd.read_csv('CLASSIFICATION DATASET.csv')

x = dataset.iloc[:, :-1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
imputer = imputer.fit(x[:, np.r_[1:5, 7:13]])  
x[:, np.r_[1:5, 7:13]]= imputer.transform(x[:, np.r_[1:5, 7:13]])   
#GENDER
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'constant', fill_value='Male', verbose=1, copy=True)
imputer = imputer.fit(x[:, 5:6]) 
x[:, 5:6]= imputer.transform(x[:, 5:6]) 
#GEOGRAPHY
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'constant', fill_value='Spain', verbose=1, copy=True)  
imputer = imputer.fit(x[:, 6:7]) 
x[:, 6:7]= imputer.transform(x[:, 6:7]) 

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 5] = labelencoder_x.fit_transform(x[:, 5])

labelencoder_x = LabelEncoder()
x[:, 6] = labelencoder_x.fit_transform(x[:, 6])

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

def predict_note_authentication(age,cp,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal):
  output= model.predict(sc_x.transform([[age,cp,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal]]))
  print("Heart Disease =", output)
  if output==[0]:
    print( 'Heart Disease Category: 0')
  elif output==[1]:
    print( 'Heart Disease Category: 1')
  elif output==[2]:
    print( 'Heart Disease Category: 2')
  elif output==[3]:
    print( 'Heart Disease Category: 3')
  else:
    print('Heart Disease Category: 4')
  print(output)
  return output

def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning Lab Experiment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Heart Disease Prediction using Decision Tree Classifier")

    age = st.number_input('Insert Age')
    cp = st.number_input('Insert CP')
    trestbps = st.number_input('Insert Trest BPS')
    chol = st.number_input('Insert Chol')
    fbs = st.number_input('Insert FBS')
    Gender = st.number_input('Insert Gender Male:1 Female:0')
    Geography = st.number_input('Insert Geography France:0 Spain:1')
    restecg = st.number_input('Insert Rest ECG')
    thalach = st.number_input('Insert Thal Ach')
    exang = st.number_input('Insert Exang')
    oldpeak = st.number_input('Insert Old Peak')
    slope = st.number_input('Insert Slope')
    ca = st.number_input('Insert CA')
    thal = st.number_input('Insert Thal')
    resul=""
    if st.button("Prediction"):
      result=predict_note_authentication(age,cp,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal)
      st.success('Model has predicted Heart Disease Category {}'.format(result))  
    if st.button("About"):
      st.header("Developed by Jatin Tak")
      st.subheader("Student, Department of Computer Engineering")
    html_temp = """
    <div class="" style="background-color:orange;" >
    <div class="clearfix">           
    <div class="col-md-12">
    <center><p style="font-size:20px;color:white;margin-top:10px;">Machine Learning Experiment : Decision Tree Classification</p></center> 
    </div>
    </div>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
if __name__=='__main__':
  main()
