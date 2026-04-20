# Room Occupancy Detection System
# Smart Room Occupancy Detection System

This project is a machine learning based web application that predicts whether a room is occupied or vacant using environmental sensor data. The prediction is made using values such as temperature, humidity, light intensity, CO2 level, and humidity ratio.

The main objective of this project is to demonstrate how machine learning can be applied in smart buildings and automation systems to improve energy efficiency and space management.

## Project Overview

Room occupancy detection is useful in offices, classrooms, meeting rooms, and other shared spaces where automatic monitoring can help reduce unnecessary power usage and improve resource utilization.

The system takes sensor values as input from the user and predicts the room status in real time through a Streamlit web interface.

## Features

- Predicts room occupancy instantly
- Interactive and user-friendly web interface
- Uses trained machine learning model
- Simple and fast execution
- Practical real-world application

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## Machine Learning Process

1. Collected occupancy detection dataset
2. Cleaned and prepared the data
3. Selected important input features
4. Trained classification model
5. Saved trained model file
6. Built web application for predictions

## Input Parameters

- Temperature
- Humidity
- Light
- CO2
- Humidity Ratio

## Output

- Occupied
- Not Occupied

## Project Structure

```bash
RoomOccupancyProject/
│── streamlit_app.py
│── train_model.py
│── model.pkl
│── datatraining.txt
│── datatest.txt
│── datatest2.txt
│── requirements.txt
│── README.md