import streamlit as st
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import time
import pandas as pd
import matplotlib.pyplot as plt


lst = [['KNN', 90], ['SVC', 91], ['Naive Bayes', 91], ['Decision Tree', 91],
       ['Random Forest', 92], ['AdaBoost', 87], ['Gradient Boosting', 91], ['XGBoost', 92]]
df2 = pd.DataFrame(lst, columns=["Model Used", "Accuracy Score in %"])
try:
    df = pd.read_csv(r"C:\Users\yadhu\Downloads\weather\weather_classification_data.csv")
except FileNotFoundError:
    st.error("Dataset file not found. Please check the file path.")
    df = pd.DataFrame()

def main():
    sel=option_menu(menu_title=None,options=["About","Prediction","Links"],icons=["info","cloud","link"],menu_icon="cast",default_index=0,
                         orientation="horizontal")
    if sel=="About":
        st.title("About This App")
        st.write("""This streamlit application predicts Weather Type based on various factors like temperature, humidity, wind speed, precipitation, and more. This model could be useful for a number of weather-dependent activities, such as outdoor events, travel and transportation, agriculture and weather alerts.""")
        if st.button("Show Dataset"):
            st.write("### Dataset Sample")
            st.dataframe(df, height=300)



            st.write(f"**Dataset Size:** {df.shape[0]} rows and {df.shape[1]} columns")

        if st.button("Show Model Accuracy Scores"):
            st.write("### Model Accuracy Scores")
            st.dataframe(df2)


        if st.button("Show Histograms"):
            st.write("### Histograms of Dataset Features")

            for column in df.columns:
                plt.figure()

                if df[column].dtype in ['float64', 'int64']:

                    plt.hist(df[column].dropna(), bins=20, color="skyblue", edgecolor="black")
                    plt.title(f"Distribution of {column}")
                    plt.xlabel(column)
                    plt.ylabel("Frequency")

                else:

                    value_counts = df[column].value_counts()
                    plt.bar(value_counts.index, value_counts.values, color="salmon", edgecolor="black")
                    plt.title(f"Category Distribution of {column}")
                    plt.xlabel(column)
                    plt.ylabel("Count")
                    plt.xticks(rotation=45)

                st.pyplot(plt)
                plt.close()

    elif sel=="Prediction":
        st.title("Weather Type Prediction")
        img = Image.open("weatherr_image.jpg")
        st.image(img, width=500)
        temperature=st.text_input("Enter Temperature","")
        humidity=st.text_input("Enter Humidity", "")
        wind_speed=st.text_input("Enter Wind Speed","")
        precipitation=st.text_input("Enter Precipitation%","")
        cloud_cover=st.selectbox("Type of cloud cover",['Partly Cloudy','Clear','Overcast','Cloudy'])
        if cloud_cover=='Partly Cloudy':
            cc=3
        elif cloud_cover=='Clear':
            cc=0
        elif cloud_cover=="Overcast":
            cc=2
        elif cloud_cover=="Cloudy":
            cc=1
        air_pressure=st.text_input("Enter the Air Pressure","")
        uv_index=st.text_input("Enter Ultra Violet Index","")
        season=st.selectbox("Select a season",["Winter","Spring","Summer","Autumn"])
        if season=="Winter":
            s=3
        elif season=="Spring":
            s=1
        elif season=="Summer":
            s=2
        elif season=="Autumn":
            s=0
        visibility=st.text_input("Enter Visibility (max horizontal distance in km)")
        location=st.selectbox("Enter location type", ["Inland","Mountain","Coastal"])
        if location=="Inland":
            l=1
        if location=="Mountain":
            l=0
        if location=="Coastal":
            l=2
        features=[temperature,humidity,wind_speed,precipitation,cc,air_pressure,uv_index,s,visibility,l]
        scaler=pickle.load(open("scalerr.sav","rb"))
        model=pickle.load(open("weatherr.sav","rb"))
        pred=st.button("Predict Weather Type")
        if pred:
            with st.spinner('Predicting the weather type...'):
                time.sleep(2)
            result=model.predict(scaler.transform([features]))
            if result==1:
                st.write("Rainy Weather")
            elif result==0:
                st.write("Cloudy Weather")
            elif result==3:
                st.write("Sunny Weather")
            elif result==2:
                st.write("Snowy Weather")
    elif sel=="Links":
        st.title("Dataset Link")
        st.write("""
        - [Kaggle Link of the Dataset](https://www.kaggle.com/datasets/nikhil7280/weather-type-classification)
        """)

        st.title("Google Collab Link")
        st.write("""
        - [Collab Link of the Dataset](https://colab.research.google.com/drive/1tALS36JBtLO_bjEsLTMbIFwmet6mbW8j?usp=sharing)
        """)
main()