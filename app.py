import streamlit as st
import pickle

st.title("Price Prediction")

pickle_in = open('Car_Price_Prediction_model', 'rb')
lr = pickle.load(pickle_in)


number2 = st.number_input('Car_Year', key='2')
number4 = st.number_input('Present_Price_of_Car', key='4')
number5 = st.number_input('Kms_Driven_by_Car', key='5')
number6 = st.number_input('Fuel_Type_of_Car(Petrol:0, Diesel:1, CNG:2)', key='6')
number7 = st.number_input('Seller_Type_of_Car(Dealer:0, Individual:1)', key='7')
number8 = st.number_input('Transmission_of_Car(Manual:0, Automatic:1)', key='8')
number9 = st.number_input("Car's_'Owner", key='9')

if st.button("Predict"):
    pred = str(lr.predict([[number2, number4, number5, number6, number7, number8, number9]]))
    st.success("Price_prediction : " + pred)