# 1. import libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

scale = StandardScaler()
encode = LabelEncoder()
model_lg = LogisticRegression()

# 2. inputs

Seat_comfort = st.number_input(label="Seat comfort", value=1)
Inflight_wifi_service = st.number_input(label="Inflight wifi service", value=1)
Inflight_entertainment = st.number_input(label="Inflight entertainment", value=1)
Online_support = st.number_input(label="Online support", value=1)
Ease_of_Online_booking = st.number_input(label="Ease of Online booking", value=1)
On_board_service = st.number_input(label="On-board service", value=1)
Leg_room_service = st.number_input(label="Leg room service", value=3)
Baggage_handling = st.number_input(label="Baggage handling", value=1)
Checkin_service = st.number_input(label="Checkin service", value=1)
Cleanliness = st.number_input(label="Cleanliness", value=1)
Online_boarding = st.number_input(label="Online boarding", value=1)

# 3. Combine input into an array of X

X_num = np.array(
    object=[
        [
        Seat_comfort,
        Inflight_wifi_service,
        Inflight_entertainment, 
        Online_support, 
        Ease_of_Online_booking,
        On_board_service, 
        Leg_room_service, 
        Baggage_handling,
        Checkin_service, 
        Cleanliness, 
        Online_boarding
        ]
    ],
    dtype=np.int32,
)
# st.write("Raw Input Data:", X_num)

# 4. import model

# # 4.1 import scaler
# with open(file="scale.pkl", mode="rb") as scale_file:
#     scale = pickle.load(file=scale_file)

# 4.2 label encoder
with open(file="encode.pkl", mode="rb") as encode_file:
    encode = pickle.load(file=encode_file)

# 4.3 Logistic Regression model 
with open(file="lg.pkl", mode="rb") as lg_file:
    lg_model = pickle.load(file=lg_file)    

# 5. Preprocessing data
# X = scale.fit_transform(X_num)
# st.write(X_num)

X = scale.fit_transform(X=X_num)
# st.write("Scaled Data:", X)


prediction = lg_model.predict(X)
prediction_decoded = encode.inverse_transform(prediction)
st.write("Prediction:", prediction)
st.write("Decoded Prediction:", prediction_decoded)



data = np.concat([X, prediction_decoded.reshape(-1, 1)], axis=1)
df = pd.DataFrame(
    data=data,
    columns=[
        'Seat comfort', 
        'Inflight wifi service',
        'Inflight entertainment', 
        'Online support', 
        'Ease of Online booking',
        'On-board service', 
        'Leg room service', 
        'Baggage handling',
        'Checkin service', 
        'Cleanliness', 
        'Online boarding',
        'satisfaction'
    ],
)
# st.write(df)
