import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import math

st.set_page_config(layout="wide")
st.title("Stock Price Prediction using Linear Regression")

# Sidebar for dataset selection
dataset_option = st.sidebar.selectbox("Select Dataset", ["Divya-TSLA", "Divya-RELIANCE", "Divya-GUJGASLTD", "Divya-ADANIGREEN", "Divya-AAPL"])

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Handle missing values (fill only numeric columns with mean)
    df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)

    # Sidebar dataset preview
    st.sidebar.header("Dataset Overview")
    with st.sidebar.expander("View Dataset"):
        st.write(df.head(10))

    # Check if required columns exist
    required_columns = ['High', 'Low', 'Open', 'Volume', 'Close']
    if not all(col in df.columns for col in required_columns):
        st.error("Dataset must contain the following columns: High, Low, Open, Volume, Close")
    else:
        # UI Buttons for different sections
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Data Preprocessing", use_container_width=True):
                with st.expander("Data Preprocessing Details"):
                    st.write("### Checking for Null Values (After Handling):")
                    st.write(df.isnull().sum())  # Should be all zeros now
                    st.write("### Dataset Description:")
                    st.write(df.describe())
                    st.write("### Dataset Dimension:")
                    st.write(df.shape)
                    st.write("### First 5 rows of the dataset:")
                    st.write(df.head(5))
                    st.write("### Last 5 rows of the dataset:")
                    st.write(df.tail(5))

        with col2:
            if st.button("Data Modeling", use_container_width=True):
                with st.expander("Model Selection"):
                    st.write("### Model Selected: Linear Regression")

                    # Selecting relevant columns
                    x = df[['High', 'Low', 'Open', 'Volume']]
                    y = df['Close']

                    # Splitting the dataset
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

                    # Train the model
                    reg = LinearRegression()
                    reg.fit(x_train, y_train)

                    # Store in session state
                    st.session_state.reg = reg
                    st.session_state.x_test = x_test
                    st.session_state.y_test = y_test

                    st.success("Model trained successfully!")

        with col3:
            if st.button("Model Evaluation", use_container_width=True):
                if "reg" in st.session_state:
                    with st.expander("Model Performance"):
                        st.subheader("Model Performance")
                        reg = st.session_state.reg  
                        x_test = st.session_state.x_test
                        y_test = st.session_state.y_test
                        
                        predicted = reg.predict(x_test)
                        df1 = pd.DataFrame({'Actual': y_test.values, 'Predicted': predicted.flatten()})
                        st.write(df1.head(10))

                        # Visualization
                        fig, ax = plt.subplots()
                        df1.head(20).plot(kind='bar', ax=ax)
                        st.pyplot(fig)

                        # Error Metrics
                        mae = metrics.mean_absolute_error(y_test, predicted)
                        mse = metrics.mean_squared_error(y_test, predicted)
                        rmse = math.sqrt(mse)
                        st.write("### Model Metrics:")
                        st.write(f"Mean Absolute Error: {mae}")
                        st.write(f"Mean Squared Error: {mse}")
                        st.write(f"Root Mean Squared Error: {rmse}")
                else:
                    st.error("Please run 'Data Modeling' first to train the model.")

    # Prediction for new input
    st.subheader("Predict New Data")
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        high = st.number_input("High Price:", min_value=0.0, format="%.2f")
    with col5:
        low = st.number_input("Low Price:", min_value=0.0, format="%.2f")
    with col6:
        open_price = st.number_input("Open Price:", min_value=0.0, format="%.2f")
    with col7:
        volume = st.number_input("Volume:", min_value=0, format="%d")

    if st.button("Predict Stock Price"):
        if "reg" in st.session_state:
            new_data = np.array([[high, low, open_price, volume]])

            # Predict only if values are valid
            if np.any(np.isnan(new_data)):
                st.error("Please enter valid numerical values for prediction.")
            else:
                predicted_price = st.session_state.reg.predict(new_data)
                st.subheader(f'Predicted Close Price: {predicted_price[0]:.2f}')
        else:
            st.error("Please train the model first by clicking 'Data Modeling'.")
