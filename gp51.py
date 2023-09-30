#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!/usr/bin/env python
# coding: utf-8

# In[45]:

import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import joblib
import pandas as pd



# Load your model here
loaded_model_1 = tf.saved_model.load("l1_model_SaveModel_format")
loaded_encoder = joblib.load('onehot_encoder.pkl')

# Later, you can load the encoder from the file
loaded_encoder = joblib.load('onehot_encoder.pkl')

# Title
st.title("Predict Terror Attack")

# Define dropdown options for each feature
attack_type_options = ["bombing","kidnapping", "shooting", "hijacking", "arson", "assasination",'stabbing']


perpetrator_options = ['group a', 'group b', 'group c', 'group d']


weapon_used_options = ["explosives","incendiary" ,"firearms", "chemicals","meele"]


                      

# Create dropdowns for each feature
input1 = st.selectbox("Attack_Type:", attack_type_options)
input2 = st.selectbox("Perpetrator:", perpetrator_options)
input3 = st.selectbox("Weapon_Used:", weapon_used_options)

# Numeric input for Victims_Injured and Victims_Deceased
input4 = st.number_input("Victims_Injured:", min_value=0)
input5 = st.number_input("Victims_Deceased:", min_value=0)

df_categorical =pd.DataFrame({'Attack_Type':[input1],'Perpetrator' :[input2],'Weapon_Used':[input3]})
df_numerical=pd.DataFrame({'Victims_Injured':[input4],'Victims_deceased' :[input5]})

encoded_data_1=pd.DataFrame(loaded_encoder.transform(df_categorical).toarray())
df_final= pd.concat([df_numerical,encoded_data_1],axis=1)



# Check if Victims_Injured and Victims_Deceased are provided
if input4 > 0 or input5 > 0:
    
    input_data = df_final.values
    input_data = tf.cast(input_data, tf.float32)
    numpy_array = input_data.numpy().reshape(-1)  # Convert the EagerTensor to a NumPy array
    series = pd.Series(numpy_array)

# Add a Button for Prediction:
    if st.button("Predict"):
        # Perform predictions using your model or algorithm
        prediction = loaded_model_1([series])


        # Determine the prediction result
        if tf.round(prediction):
            prediction = 'Major Attack'
        else:
            prediction = 'Minor Attack'

        # Display the prediction result
        st.write("Prediction:", prediction)


# In[ ]:




