import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load data
url = 'https://raw.githubusercontent.com/michalis0/MGT-502-Data-Science-and-Machine-Learning/main/data/yield_df.csv'
df_yield = pd.read_csv(url)

st.title('Crop Yield Prediction')

# Sidebar for user input
st.sidebar.title('Input Parameters')
item = st.sidebar.selectbox('Item', df_yield['Item'].unique())
avg_rainfall = st.sidebar.number_input('Average Rainfall (mm/year)', value=0.0)
pesticides = st.sidebar.number_input('Pesticides (tonnes)', value=0.0)
avg_temp = st.sidebar.number_input('Average Temperature', value=0.0)

# Preprocess data
X = df_yield[['Item', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']]
y = df_yield['hg/ha_yield']

# Encode 'Item' column
le = LabelEncoder()
X['Item'] = le.fit_transform(X['Item'])

# Define the scaler
scaler = MinMaxScaler()
# Fit and transform
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=True)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make prediction
user_input = [[le.transform([item])[0], avg_rainfall, pesticides, avg_temp]]
user_input_scaled = scaler.transform(user_input)
prediction = model.predict(user_input_scaled)

st.subheader('Prediction:')
st.write('Predicted Yield:', prediction[0])

# Evaluate model
predictions_test = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions_test)
mse = mean_squared_error(y_test, predictions_test)
r2 = r2_score(y_test, predictions_test)

st.subheader('Model Evaluation:')
st.write(f"MAE test set: {mae:.2f}")
st.write(f"MSE test set: {mse:.2f}")
st.write(f"R\u00b2 test set: {r2:.2f}")
