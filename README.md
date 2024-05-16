
# Market Depth Trend Prediction Project

## Overview
This project is centered around developing a machine learning model to predict short-term market movements based on market depth data. Market depth data consists of bids and asks for a financial instrument, representing the demand and supply at different price levels. The core of the project involves two main Python scripts:

1. `Market_depth_trend_prediction_model.py`: This script contains the implementation of various machine learning models to analyze market depth data and predict the trend direction (up or down) in the short term.

2. `save_marketdepth_as_sql.py`: This script is responsible for processing and storing market depth data in a structured SQL database, facilitating efficient data retrieval for model training and analysis.

## Models Implemented
The project features an array of machine learning models, each with unique characteristics and performance metrics. The models include:

- **Dense Neural Network**: A fully connected neural network model with multiple dense layers, suitable for capturing nonlinear relationships in the data.

- **Convolutional Neural Network (CNN)**: Utilizes convolutional layers to capture spatial dependencies in the market depth data, making it effective for time-series like data structures.

- **Long Short-Term Memory (LSTM)**: A type of recurrent neural network (RNN) that is capable of learning long-term dependencies, ideal for sequential data like market depth.

- **Recurrent Neural Network (RNN)**: Another variant of RNN used to model temporal dynamics and sequences in the market depth data.

- **Simple RNN**: A simpler version of RNN with fewer parameters, providing a more efficient but less complex model.

- **CNN-RNN Hybrid**: Combines the spatial learning capabilities of CNNs with the temporal learning capabilities of RNNs, offering a robust model for time-series data.

- **LSTM-CNN Hybrid**: Integrates LSTM's ability to remember long-term dependencies with CNN's spatial feature extraction, providing a comprehensive approach to time-series analysis.

- **XGBoost**: A gradient boosting model known for its performance and speed, used for classification tasks in this project.

Each model is designed with early stopping mechanisms based on accuracy, ensuring efficient training without overfitting.

## Data Preprocessing
The project employs sophisticated data preprocessing techniques to transform raw market depth data into a format suitable for machine learning models. This includes:

- **Feature Engineering**: Extracting meaningful features from the market depth data, such as mid-price, book imbalance, and volume-weighted average price (VWAP) distances.

- **Normalization**: Scaling the features to a common scale to improve model convergence and performance.

- **Target Variable Creation**: Defining the prediction target, such as the direction of the next tick's mid-price movement.

## Project Structure
- `Market_depth_trend_prediction_model.py`: Contains the implementation of machine learning models, data preprocessing steps, and model training routines.
  
- `save_marketdepth_as_sql.py`: Scripts for processing market depth data and storing it in an SQL database for easy access and analysis.

## Usage
To use this project, ensure you have the required Python environment set up with necessary libraries like TensorFlow, Keras, XGBoost, and SQL connectors. Run the `save_marketdepth_as_sql.py` script to preprocess and store your market depth data, followed by the `Market_depth_trend_prediction_model.py` script to train and evaluate the machine learning models on your data.

## Conclusion
This project offers a comprehensive framework for predicting market trends based on depth data using various machine learning models. It is designed to be modular, allowing for easy experimentation with different models and preprocessing techniques to achieve the best predictive performance.
