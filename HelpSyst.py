# Description: This program visualize health data and social determinants of health using machine learning and python

# Import the libraries
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import pandas as pd
import streamlit as st

pylint: disable=E1120

# Create a title and a sub-title
st.write("""
# Health Program Recommender System
Detect health problem in a population using machine learning and python
""")

# Open and display an image
image = Image.open('D:/Kuliah/Lomba/Hackathon Microsoft/hack.jpeg')
st.image(image, caption='ML', use_column_width=True)

# Get the data
df = pd.read_csv('D:/Kuliah/Lomba/Hackathon Microsoft/2017.csv')
# Set a subheader
st.subheader('Data Information :')
# Show the data as a table
st.dataframe(df)
# Show statistic on the data
st.write(df.describe())
# Show the data as a chart
chart = st.bar_chart(df)

# Split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:6].values
Y = df.iloc[:, -1].values
# Split the data set into 75% Training and 25% Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Get the feature input from user
def get_user_input() :
    puskesmas = st.sidebar.slider('puskesmas', 0, 4, 0)
    disease = st.sidebar.slider('disease', 0, 50, 0)
    number_of_cases	= st.sidebar.slider('number_of_cases', 0, 10000, 0)
    clean_water_facilities = st.sidebar.slider('clean_water_facilities', 0, 1, 0)
    state_of_settlement = st.sidebar.slider('state_of_settlement', 0, 1, 0)
    clean_living_behavior = st.sidebar.slider('clean_living_behaviour', 0, 1, 0)

    # Store a dictionary into a variable
    user_data = {   'puskesmas': puskesmas,
                    'disease': disease,
                    'number_of_cases' : number_of_cases,
                    'clean_water_facilities' : clean_water_facilities,
                    'state_of_settlement' : state_of_settlement,
                    'clean_living_behavior' : clean_living_behavior
                }
    # Transform the data into a dataframe
    features = pd.DataFrame(user_data, index = [0])
    return features

# Store the user input into a variable
user_input = get_user_input()

# Set a subheader and display the users input
st.subheader('User Input: ')
st.write(user_input)

# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Show the models metrics
st.subheader('Model Test Accuracy Score: ')
st.write( str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

# Store the models predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

# Set a subheader and display the classification
st.subheader('Classification: ')
st.write(prediction)
