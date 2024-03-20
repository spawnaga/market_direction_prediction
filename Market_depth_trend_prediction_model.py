import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import tree
from xgboost import XGBClassifier
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout, MaxPool1D, LSTM, RNN, SimpleRNN, Reshape,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.regularizers import l2
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPUs 0 and 1


def calculate_future_price_optimized(df, minutes_ahead=5):
    df['time'] = pd.to_datetime(df['time'])
    temp_df = df.copy()
    temp_df.set_index('time', inplace=True)

    # Approximate progress with tqdm by iterating through the length of the DataFrame
    for _ in tqdm(range(len(df)), desc="Processing..."):
        pass  # This is just to show the progress bar and does not do any operation

    # Continue with the resampling operation
    resampled_df = temp_df.resample(f'{minutes_ahead}min').agg({'mid_price': 'mean'})
    resampled_df['future_mid_price'] = resampled_df['mid_price'].shift(-minutes_ahead)
    resampled_df.reset_index(inplace=True)
    final_df = pd.merge_asof(df.sort_values('time'), resampled_df[['time', 'future_mid_price']].sort_values('time'),
                             on='time', direction='forward')

    return final_df


def add_direction_based_on_future_price(df):
    tqdm.pandas(desc="Calculating directions")  # Prepare tqdm to work with pandas

    conditions = [
        (df['future_mid_price'] > df['mid_price']),  # Condition for 1
        (df['future_mid_price'] < df['mid_price']),  # Condition for -1
    ]
    choices = [1, -1]  # Corresponding choices for conditions

    # Use np.select to apply conditions and choices, with tqdm to show progress
    df['direction'] = np.select(conditions, choices, default=0)  # default is 0 if neither condition is met

    df.dropna(inplace=True)
    return df


def plot_training_history(trainHistory, model_name):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(trainHistory.history['accuracy'])
    plt.plot(trainHistory.history['val_accuracy'])
    plt.title(f'Model Accuracy - {model_name}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(trainHistory.history['loss'])
    plt.plot(trainHistory.history['val_loss'])
    plt.title(f'Model Loss - {model_name}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()


class ModelTraining:
    def __init__(self, database_path):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.y_train = None
        self.model = None
        if os.path.splitext(database_path)[1] == ".csv":
            self.df = pd.read_csv(database_path)
        elif os.path.splitext(database_path)[1] == ".db":
            engine = create_engine(f"sqlite:///ES_ticks.db")
            self.df = pd.read_sql("SELECT * from ES_market_depth", engine)
        # if "time" in self.df.columns:
        #     self.df = self.df.drop("time", axis=1)
        if "lastSize" in self.df.columns:
            self.df = self.df.drop("lastSize", axis=1)
        self.early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        self.scaler = StandardScaler()
        self.logdir = "logs/fit/" + "agentLearning" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir)
        self.optimizer = Adam(learning_rate=0.01)
        self.lr_schedule = LearningRateScheduler(lambda epoch: 0.001 * (0.1 ** (epoch // 10)))
        self.models = {}

    def calculate_correlation_matrix(self, df):
        correlation_matrix = df.corr()
        return correlation_matrix

    def plot_correlation_heatmap(self, correlation_matrix):
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)
        plt.title('Feature Correlation Heatmap')
        plt.show()

    def identify_highly_correlated_features(self, correlation_matrix, threshold=0.9):
        highly_correlated_pairs = {}
        for col in correlation_matrix.columns:
            for row in correlation_matrix.index:
                if abs(correlation_matrix[col][row]) > threshold and col != row:
                    highly_correlated_pairs[(col, row)] = correlation_matrix[col][row]
        return highly_correlated_pairs

    def filter_redundant_features(self, df, highly_correlated_pairs):
        # Assuming you always remove the second feature in the tuple
        features_to_remove = {pair[1] for pair in highly_correlated_pairs}
        filtered_df = df.drop(columns=features_to_remove)
        return filtered_df

    def compute_order_book_features(self, df, depth=5):
        # Calculate mid-price and add it to the DataFrame before other calculations
        df['mid_price'] = (df['bidPrice_1'] + df['askPrice_1']) / 2

        # Initialize empty DataFrame for features
        features = pd.DataFrame(index=df.index)
        features['time'] = df['time']

        # Use the newly added 'mid_price' column for further feature calculations
        features['mid_price'] = df['mid_price']
        features['spread'] = df['askPrice_1'] - df['bidPrice_1']  # Calculate book imbalance for top 'depth' levels
        total_bid_volume = df[[f'bidSize_{i}' for i in range(1, depth + 1)]].sum(axis=1)
        total_ask_volume = df[[f'askSize_{i}' for i in range(1, depth + 1)]].sum(axis=1)
        features['book_imbalance'] = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)

        # Calculate VWAP distance from mid-price for bids and asks
        bid_vwap = sum(df[f'bidPrice_{i}'] * df[f'bidSize_{i}'] for i in range(1, depth + 1)) / total_bid_volume
        ask_vwap = sum(df[f'askPrice_{i}'] * df[f'askSize_{i}'] for i in range(1, depth + 1)) / total_ask_volume
        features['bid_vwap_distance'] = features['mid_price'] - bid_vwap
        features['ask_vwap_distance'] = ask_vwap - features['mid_price']

        # Price and volume trends over recent ticks
        for window in [1, 5, 10]:  # Example windows: last 1, 5, and 10 ticks
            df_rolled = df[['bidPrice_1', 'askPrice_1', 'bidSize_1', 'askSize_1']].rolling(window=window)
            features[f'bidPrice_trend_{window}'] = df_rolled['bidPrice_1'].mean()
            features[f'askPrice_trend_{window}'] = df_rolled['askPrice_1'].mean()
            features[f'bidSize_trend_{window}'] = df_rolled['bidSize_1'].mean()
            features[f'askSize_trend_{window}'] = df_rolled['askSize_1'].mean()

        # Historical volatility (using mid-price)
        features['historical_volatility'] = features['mid_price'].diff().rolling(window=10).std()

        # Time of day and market session
        features['time_of_day'] = df['time'].dt.hour + df['time'].dt.minute / 60 + df['time'].dt.second / 3600
        features['day_of_week'] = df['time'].dt.dayofweek

        # Calculate mid-price movement (lagged to avoid look-ahead bias)
        features['mid_price_movement'] = features['mid_price'].diff().shift(-1)

        # Calculate order flow momentum (difference in bid and ask updates, lagged)
        features['order_flow_momentum'] = (df['bidSize_1'] - df['askSize_1']).diff().shift(-1)

        # Moving Averages
        for window in [5, 10, 20]:  # Example windows
            features[f'sma_{window}'] = features['mid_price'].rolling(window=window).mean()
            features[f'ema_{window}'] = features['mid_price'].ewm(span=window, adjust=False).mean()

        # Drop NaN values created by rolling functions
        features = features.dropna()

        return features

    def preprocess_data(self, minutes_ahead=5):
        # Ensure 'time' is in datetime format
        self.df['time'] = pd.to_datetime(self.df['time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        self.df.loc[self.df['time'].isna(), 'time'] = pd.to_datetime(self.df.loc[self.df['time'].isna(), 'time'],
                                                                     format='%Y-%m-%d %H:%M:%S', errors='coerce')
        self.df.dropna(subset=['time'], inplace=True)

        # Compute custom features based on order book data
        features_df = self.compute_order_book_features(self.df)

        self.df.sort_values(by='time', inplace=True)

        self.df = calculate_future_price_optimized(self.df, minutes_ahead)

        # Make sure both DataFrames are sorted by time
        self.df.sort_values(by='time', inplace=True)
        features_df.sort_values(by='time', inplace=True)

        # Rename 'mid_price' in features_df to 'price_y' before merging
        features_df.rename(columns={'mid_price': 'price_y'}, inplace=True)

        # Calculate correlation matrix
        correlation_matrix = features_df.corr()

        # Plot heatmap
        self.plot_correlation_heatmap(correlation_matrix)

        # Identify highly correlated features
        highly_correlated_pairs = self.identify_highly_correlated_features(correlation_matrix)

        # Filter redundant features
        features_df_filtered = self.filter_redundant_features(features_df, highly_correlated_pairs)

        # Use merge_asof to merge the features back into the main DataFrame based on the 'time' column
        merged_df = pd.merge_asof(self.df.sort_values('time'), features_df_filtered.sort_values('time'), on='time',
                                  direction='nearest')

        # Use np.select to apply conditions and choices
        merged_df = add_direction_based_on_future_price(merged_df)
        merged_df.dropna(inplace=True)

        csv_filename = "preprocessed_data.csv"  # Define the name of your CSV file
        merged_df.to_csv(csv_filename, index=False)  # Save the DataFrame to a CSV file without the index
        print(f"Preprocessed data saved to {csv_filename}")
        merged_df = merged_df.drop("future_mid_price", axis=1)
        self.df = self.df.drop("future_mid_price", axis=1)

        # Select only the numerical columns (excluding 'time')
        numerical_columns = [col for col in self.df.columns if col != 'time']

        self.X = merged_df[numerical_columns].values
        self.y = merged_df['direction'].values

        # Splitting the dataset into training and testing sets, normalizing the feature set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=42, shuffle=False)
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        # Adjust y values for binary classification: map -1 to 0, keep 1 as is
        self.y_train = np.where(self.y_train == -1, 0, self.y_train)
        self.y_test = np.where(self.y_test == -1, 0, self.y_test)

    def create_models(self):
        self.models = {
            "Dense": self.Dense_model,
            "CNN": self.Dense_model,
            "LSTM": self.Dense_model,
            "RNN": self.Dense_model,
            "SimpleRNN": self.Dense_model,
            "CNNRNN": self.Dense_model,
            "xgboost": self.Dense_model,
            "treeDecision": self.Dense_model,
            "LinearRegression": self.Dense_model,
            "LogisticRegression": self.Dense_model,
            "KNeighborsClassifier": self.Dense_model,
            "RandomForestClassifier": self.Dense_model,
            "GradientBoostingClassifier": self.Dense_model,
            "SVC": self.Dense_model,
            "GaussianNB": self.Dense_model,
            "Perceptron": self.Dense_model,
            "MLPClassifier": self.Dense_model
        }

    def Dense_model(self):
        # self.y_train = to_categorical(self.y_train, num_classes=1)
        # self.y_test = to_categorical(self.y_test, num_classes=1)

        input_layer = Input(shape=(self.X_train.shape[1],))
        x = Dense(100, activation='relu', kernel_regularizer='l2')(input_layer)
        x = Dropout(0.7)(x)
        x = Dense(50, activation='relu', kernel_regularizer='l2')(x)
        x = Dropout(0.7)(x)
        x = Dense(25, activation='relu', kernel_regularizer='l2')(x)
        x = Dropout(0.7)(x)
        output_layer = Dense(1, activation='sigmoid', kernel_regularizer='l2')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        # Configure the EarlyStopping callback
        early_stop = EarlyStopping(
            monitor='val_accuracy',  # Monitor validation accuracy
            min_delta=0.001,  # Minimum change to qualify as an improvement
            patience=10,  # Number of epochs with no improvement after which training will be stopped
            verbose=1,
            mode='max',  # Since we are monitoring accuracy, which should increase
            restore_best_weights=True
            # Restore model weights from the epoch with the best value of the monitored quantity
        )

        trainHistory = model.fit(
            self.X_train, self.y_train, epochs=100, batch_size=64,
            validation_data=(self.X_test, self.y_test),
            callbacks=[self.tensorboard_callback, self.lr_schedule]  # Add early_stop to callbacks
        )

        return trainHistory, trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

    def CNN_model(self):
        # Reshape input data to be 3D for Conv1D layers: [samples, time steps, features]
        X_train_reshaped = np.expand_dims(self.X_train, axis=-1)  # Adding a dimension for 'features'
        X_test_reshaped = np.expand_dims(self.X_test, axis=-1)  # Same for X_test

        # Ensure y_train and y_test are one-hot encoded
        # num_classes = 2  # Adjust based on your actual number of classes
        # y_train_encoded = to_categorical(self.y_train, num_classes=num_classes)
        # y_test_encoded = to_categorical(self.y_test, num_classes=num_classes)

        # Define the CNN model
        input_layer = Input(shape=(X_train_reshaped.shape[1], 1))  # Adjusted shape
        x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_layer)
        x = MaxPool1D(pool_size=2)(x)
        x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool1D(pool_size=2)(x)
        x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='sigmoid')(x)  # Adjusted output layer to match num_classes
        model = Model(inputs=input_layer, outputs=output)

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        # Configure the EarlyStopping callback to monitor validation accuracy
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.001,
            patience=30,
            verbose=1,
            mode='max',
            restore_best_weights=True
        )

        # Train the model with early stopping
        trainHistory = model.fit(
            X_train_reshaped, self.y_train, epochs=100,  # Adjusted epochs
            batch_size=500, validation_data=(X_test_reshaped, self.y_test),
            callbacks=[early_stop, self.tensorboard_callback, self.lr_schedule]  # Use early_stop in callbacks
        )

        return trainHistory, trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

    def LSTM_model(self):

        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        with strategy.scope():
            # self.y_train = to_categorical(self.y_train, num_classes=3)
            # self.y_test = to_categorical(self.y_test, num_classes=3)

            input_layer = Input(shape=(self.X_train.shape[1], 1))  # Adjusted for reshaped X_train
            x = LSTM(units=64, return_sequences=True)(input_layer)
            x = Dropout(0.8)(x)
            x = LSTM(units=64, return_sequences=True)(x)
            x = Dropout(0.8)(x)
            x = LSTM(units=64, return_sequences=True)(x)
            x = Dropout(0.8)(x)
            x = LSTM(units=64, return_sequences=True)(x)
            x = Dropout(0.8)(x)
            x = LSTM(units=32, return_sequences=False)(x)  # Last LSTM layer usually does not return sequences
            x = Dropout(0.8)(x)
            output = Dense(1, activation='sigmoid')(x)  # Use softmax for multi-class classification

            model = Model(inputs=input_layer, outputs=output)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Configure the EarlyStopping callback to monitor validation accuracy
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.001,
            patience=10,
            verbose=1,
            mode='max',
            restore_best_weights=True
        )

        trainHistory = model.fit(
            self.X_train, self.y_train, epochs=100,  # Reduced epochs for demonstration
            batch_size=64,  # Adjusted batch size
            validation_data=(self.X_test, self.y_test),
            callbacks=[early_stop, self.tensorboard_callback, self.lr_schedule]
        )

        return trainHistory, trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

    def SimpleRNN_model(self):
        # Reshape input data to be 3D: [samples, time steps, features]
        X_train_reshaped = np.expand_dims(self.X_train, axis=-1)  # Adding a dimension for 'features'
        X_test_reshaped = np.expand_dims(self.X_test, axis=-1)

        # One-hot encode the target variables
        y_train_encoded = to_categorical(self.y_train, num_classes=3)
        y_test_encoded = to_categorical(self.y_test, num_classes=3)

        # Define the RNN model
        input_layer = Input(shape=(X_train_reshaped.shape[1], 1))
        x = SimpleRNN(units=32, return_sequences=True)(input_layer)
        x = Dropout(0.5)(x)
        x = SimpleRNN(units=64, return_sequences=False)(x)  # Last RNN layer usually does not return sequences
        x = Dropout(0.5)(x)
        output = Dense(2, activation='sigmoid')(x)  # Assuming 3 classes for the output layer
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

        # Configure the EarlyStopping callback to monitor validation accuracy
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.001,
            patience=10,
            verbose=1,
            mode='max',
            restore_best_weights=True
        )

        # Train the model with early stopping
        trainHistory = model.fit(
            X_train_reshaped, y_train_encoded, epochs=100,  # A more reasonable number of epochs
            batch_size=500, validation_data=(X_test_reshaped, y_test_encoded),
            callbacks=[early_stop, self.tensorboard_callback, self.lr_schedule]  # Use early_stop in callbacks
        )

        return trainHistory, trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

    def CNNRNN_model(self):
        # Ensure target variables are one-hot encoded
        y_train_encoded = to_categorical(self.y_train, num_classes=3)
        y_test_encoded = to_categorical(self.y_test, num_classes=3)

        # Reshape input data to be 3D for Conv1D layers
        X_train_reshaped = np.expand_dims(self.X_train, axis=-1)
        X_test_reshaped = np.expand_dims(self.X_test, axis=-1)

        # Define the CNN-RNN model
        input_layer = Input(shape=(X_train_reshaped.shape[1], 1))
        x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_layer)
        x = MaxPool1D(pool_size=2)(x)
        x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool1D(pool_size=2)(x)
        x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Reshape((-1, 64))(x)  # Adjusted reshape to match RNN input requirements
        x = SimpleRNN(units=64, activation='relu')(x)
        output = Dense(2, activation='sigmoid')(x)  # Using softmax for a classification task
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

        # Configure the EarlyStopping callback
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.001,
            patience=20,
            verbose=1,
            mode='max',
            restore_best_weights=True
        )

        # Train the model with early stopping
        trainHistory = model.fit(
            X_train_reshaped, y_train_encoded, epochs=100,  # A reasonable number of epochs
            batch_size=500, validation_data=(X_test_reshaped, y_test_encoded),
            callbacks=[early_stop, self.tensorboard_callback, self.lr_schedule]  # Include early_stop in callbacks
        )

        return trainHistory, trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

    def LSTMCNN_model(self):
        # Reshape input data for LSTM layers
        X_train_reshaped = np.expand_dims(self.X_train, axis=-1)
        X_test_reshaped = np.expand_dims(self.X_test, axis=-1)

        input_layer = Input(shape=(X_train_reshaped.shape[1], 1))
        x = LSTM(64, return_sequences=True)(input_layer)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = MaxPool1D(pool_size=2)(x)
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='sigmoid')(x)  # Assuming binary classification for buy/sell decision

        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

        trainHistory = model.fit(X_train_reshaped, self.y_train, epochs=100, batch_size=64,
                                 validation_data=(X_test_reshaped, self.y_test), callbacks=[self.tensorboard_callback, self.lr_schedule])

        return trainHistory, trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]


    def check_and_convert_target_variable(self):
        # Check if the target variable contains more than two unique values
        if len(np.unique(self.y_train)) > 2:
            # Convert target variable to binary format using LabelEncoder for training data
            label_encoder = LabelEncoder()
            self.y_train = label_encoder.fit_transform(self.y_train)

        if len(np.unique(self.y_test)) > 2:
            # Convert target variable to binary format using LabelEncoder for test data
            label_encoder = LabelEncoder()
            self.y_test = label_encoder.transform(self.y_test)

    def xgboost_model(self):
        # Adjust y values for binary classification: map -1 to 0, keep 1 as is
        y_train_adjusted = np.where(self.y_train == -1, 0, self.y_train)
        y_test_adjusted = np.where(self.y_test == -1, 0, self.y_test)

        # Initialize the XGBClassifier for binary classification
        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',  # Use 'logloss' for binary log loss
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic'  # For binary classification
        )

        # Fit the model on the training data
        self.model.fit(self.X_train, y_train_adjusted)

        # Predict probabilities for the test set
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]  # Get probability for class '1'

        # Calculate accuracy
        y_pred = (y_pred_proba > 0.5).astype(int)  # Convert probabilities to class labels
        accuracy = accuracy_score(y_test_adjusted, y_pred)

        # Calculate log loss
        loss = log_loss(y_test_adjusted, y_pred_proba)

        return accuracy, loss

    def treeDecision_model(self):
        # Check and convert target variable if necessary
        self.check_and_convert_target_variable()

        # Initialize the DecisionTreeClassifier
        self.model = DecisionTreeClassifier()

        # Use cross-validation to evaluate model performance
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_val_score(self.model, self.X_train, self.y_train, cv=kfold, scoring='accuracy')
        print(f'Decision Tree Cross-Validation Accuracy: {cv_results.mean()} (+/- {cv_results.std()})')

        # Fit the model on the entire dataset
        self.model.fit(self.X_train, self.y_train)

        # Predict probabilities for the test set
        y_pred_proba = self.model.predict_proba(self.X_test)

        # Predict classes for the test set
        y_pred = self.model.predict(self.X_test)

        # Calculate accuracy and log loss
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, y_pred_proba)

        return accuracy, loss

    def LinearRegression_model(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print("MSE found to be at best at", mse)
        return mse, 0

    def LogisticRegression_model(self):
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        return accuracy, loss

    def KNeighborsClassifier_model(self):
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        return accuracy, loss

    def RandomForestClassifier_model(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, y_pred, labels=y_pred)
        return accuracy, loss

    def GradientBoostingClassifier_model(self):
        self.model = GradientBoostingClassifier(n_estimators=100)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        return accuracy, loss

    def SVC_model(self):
        # Initialize the SVC model with probability estimates enabled
        self.model = SVC(probability=True)  # Enable probability estimates

        # Fit the model to the training data
        self.model.fit(self.X_train, self.y_train)

        # Predict class labels for the test set
        y_pred = self.model.predict(self.X_test)

        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)

        # Predict probabilities for the test set
        y_pred_proba = self.model.predict_proba(self.X_test)

        # Calculate log loss
        loss = log_loss(self.y_test, y_pred_proba)

        return accuracy, loss

    def GaussianNB_model(self):
        self.model = GaussianNB()
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        return accuracy, loss

    def Perceptron_model(self):
        self.model = Perceptron()
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, self.model._predict_proba_lr(self.X_test))
        return accuracy, loss

    def MLPClassifier_model(self):
        self.model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=10)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        return accuracy, loss


if __name__ == '__main__':
    mt = ModelTraining(database_path='ES_ticks.db')
    mt.preprocess_data()

    # Train the Dense model and get its history
    # accuracy, loss = mt.SVC_model()

    # Train the Dense model and get its history
    trainHistory, accuracy, loss = mt.LSTMCNN_model()
    #
    # # Plot the training history
    plot_training_history(trainHistory, "LSTMCNN")

    print(accuracy, loss)
