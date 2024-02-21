import datetime

from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
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
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout, MaxPool1D, LSTM, RNN, SimpleRNN, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # Use GPUs 0 and 1


def compute_order_book_features(df, depth=5):
    # Initialize empty DataFrame for features
    features = pd.DataFrame(index=df.index)

    # Calculate mid-price
    features['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2

    # Calculate order book imbalance for top 'depth' levels
    total_bid_volume = df[[f'bid_volume_{i}' for i in range(1, depth + 1)]].sum(axis=1)
    total_ask_volume = df[[f'ask_volume_{i}' for i in range(1, depth + 1)]].sum(axis=1)
    features['book_imbalance'] = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)

    # Calculate VWAP distance from mid-price for bids and asks
    bid_vwap = sum(df[f'bid_price_{i}'] * df[f'bid_volume_{i}'] for i in range(1, depth + 1)) / total_bid_volume
    ask_vwap = sum(df[f'ask_price_{i}'] * df[f'ask_volume_{i}'] for i in range(1, depth + 1)) / total_ask_volume
    features['bid_vwap_distance'] = features['mid_price'] - bid_vwap
    features['ask_vwap_distance'] = ask_vwap - features['mid_price']

    # Calculate mid-price movement (lagged to avoid look-ahead bias)
    features['mid_price_movement'] = features['mid_price'].diff().shift(-1)

    # Calculate order flow momentum (difference in bid and ask updates, lagged)
    features['order_flow_momentum'] = (df['bid_updates'] - df['ask_updates']).diff().shift(-1)

    return features.dropna()


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
            engine = create_engine(f"sqlite:///NQ_ticks.db")
            self.df = pd.read_sql("SELECT * from NQ_market_depth", engine)
        if "time" in self.df.columns:
            self.df = self.df.drop("time", axis=1)
        self.df = self.df.drop("lastSize", axis=1)
        self.early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        self.scaler = MinMaxScaler()
        self.logdir = "logs/fit/" + "agentLearning" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir)
        self.optimizer = Adam(learning_rate=0.01)
        self.lr_schedule = LearningRateScheduler(lambda epoch: 0.001 * (0.1 ** (epoch // 10)))
        self.models = {}

    def compute_order_book_features(self, depth=5):
        features = pd.DataFrame(index=self.df.index)

        # Correcting the column names to include underscores
        features['mid_price'] = (self.df['bidPrice_1'] + self.df['askPrice_1']) / 2

        # Calculating total bid and ask volumes at top 'depth' levels
        total_bid_volume = self.df[[f'bidSize_{i}' for i in range(1, depth + 1)]].sum(axis=1)
        total_ask_volume = self.df[[f'askSize_{i}' for i in range(1, depth + 1)]].sum(axis=1)
        features['book_imbalance'] = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)

        # VWAP distance calculations
        bid_vwap = sum(
            self.df[f'bidPrice_{i}'] * self.df[f'bidSize_{i}'] for i in range(1, depth + 1)) / total_bid_volume
        ask_vwap = sum(
            self.df[f'askPrice_{i}'] * self.df[f'askSize_{i}'] for i in range(1, depth + 1)) / total_ask_volume
        features['bid_vwap_distance'] = features['mid_price'] - bid_vwap
        features['ask_vwap_distance'] = ask_vwap - features['mid_price']

        # Calculate mid-price movement (lagged to avoid look-ahead bias)
        features['mid_price_movement'] = features['mid_price'].diff().shift(-1)

        # Calculate order flow momentum (difference in bid and ask updates, lagged)
        # features['order_flow_momentum'] = (self.df['bid_updates'] - self.df['ask_updates']).diff().shift(-1)
        return features

    def preprocess_data(self):
        # Compute custom features based on order book data
        features = self.compute_order_book_features()

        # Ensure 'features' DataFrame includes 'mid_price'
        assert 'mid_price' in features.columns, "mid_price column not found in features DataFrame"

        # Integrate 'features' back into 'self.df'
        self.df = self.df.join(features)  # Using join to add features to self.df

        # Ensure 'mid_price' is now part of 'self.df'
        assert 'mid_price' in self.df.columns, "mid_price column not integrated into self.df"

        # Proceed with using 'mid_price'
        self.df['future_mid_price'] = self.df['mid_price'].shift(-1)  # Predicting next tick's mid-price
        self.df['direction'] = (self.df['future_mid_price'] > self.df['mid_price']).astype(int)
        self.df.dropna(inplace=True)

        self.X = self.df[features.columns].values  # Use only the columns from features for X
        self.y = self.df['direction'].values

        # Splitting the dataset into training and testing sets, normalizing the feature set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=42, shuffle=False)
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

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
        self.y_train = to_categorical(self.y_train, num_classes=2)
        self.y_test = to_categorical(self.y_test, num_classes=2)

        input_layer = Input(shape=(self.X_train.shape[1],))
        x = Dense(128, activation='relu')(input_layer)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(2, activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

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
            callbacks=[early_stop, self.tensorboard_callback, self.lr_schedule]  # Add early_stop to callbacks
        )

        return trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

    def CNN_model(self):
        # Reshape input data to be 3D for Conv1D layers: [samples, time steps, features]
        X_train_reshaped = np.expand_dims(self.X_train, axis=-1)  # Adding a dimension for 'features'
        X_test_reshaped = np.expand_dims(self.X_test, axis=-1)  # Same for X_test

        # Ensure y_train and y_test are one-hot encoded
        num_classes = 3  # Adjust based on your actual number of classes
        y_train_encoded = to_categorical(self.y_train, num_classes=num_classes)
        y_test_encoded = to_categorical(self.y_test, num_classes=num_classes)

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
        output = Dense(num_classes, activation='softmax')(x)  # Adjusted output layer to match num_classes
        model = Model(inputs=input_layer, outputs=output)

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

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
            X_train_reshaped, y_train_encoded, epochs=100,  # Adjusted epochs
            batch_size=500, validation_data=(X_test_reshaped, y_test_encoded),
            callbacks=[early_stop, self.tensorboard_callback, self.lr_schedule]  # Use early_stop in callbacks
        )

        return trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

    def LSTM_model(self):

        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        with strategy.scope():
            self.y_train = to_categorical(self.y_train, num_classes=3)
            self.y_test = to_categorical(self.y_test, num_classes=3)

            input_layer = Input(shape=(self.X_train.shape[1], 1))  # Adjusted for reshaped X_train
            x = LSTM(units=32, return_sequences=True)(input_layer)
            x = Dropout(0.5)(x)
            x = LSTM(units=64, return_sequences=True)(x)
            x = Dropout(0.5)(x)
            x = LSTM(units=32, return_sequences=False)(x)  # Last LSTM layer usually does not return sequences
            x = Dropout(0.5)(x)
            output_layer = Dense(3, activation='softmax')(x)  # Use softmax for multi-class classification

            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

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

        return trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

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
        output = Dense(3, activation='softmax')(x)  # Assuming 3 classes for the output layer
        model = Model(inputs=input_layer, outputs=output)

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

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

        return trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

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
        output = Dense(3, activation='softmax')(x)  # Using softmax for a classification task
        model = Model(inputs=input_layer, outputs=output)

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        # Configure the EarlyStopping callback
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
            X_train_reshaped, y_train_encoded, epochs=100,  # A reasonable number of epochs
            batch_size=500, validation_data=(X_test_reshaped, y_test_encoded),
            callbacks=[early_stop, self.tensorboard_callback, self.lr_schedule]  # Include early_stop in callbacks
        )

        return trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

    def LSTMCNN_model(self):
        # Ensure target variables are one-hot encoded
        y_train_encoded = to_categorical(self.y_train, num_classes=3)
        y_test_encoded = to_categorical(self.y_test, num_classes=3)

        # Reshape input data to be 3D for LSTM layers
        X_train_reshaped = np.expand_dims(self.X_train, axis=-1)
        X_test_reshaped = np.expand_dims(self.X_test, axis=-1)

        # Define the LSTM-CNN model
        input_layer = Input(shape=(X_train_reshaped.shape[1], 1))
        x = LSTM(64, return_sequences=True)(input_layer)
        x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
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
        output = Dense(3, activation='softmax')(x)  # Using softmax for a classification task
        model = Model(inputs=input_layer, outputs=output)

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        # Configure the EarlyStopping callback
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
            X_train_reshaped, y_train_encoded, epochs=100,  # A reasonable number of epochs
            batch_size=500, validation_data=(X_test_reshaped, y_test_encoded),
            callbacks=[early_stop, self.tensorboard_callback, self.lr_schedule]  # Include early_stop in callbacks
        )

        return trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import cross_val_score, StratifiedKFold

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
        # Check and convert target variable if necessary
        self.check_and_convert_target_variable()

        # Initialize the XGBClassifier with specific parameters
        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',  # Use 'mlogloss' for multi-class log loss
            n_estimators=100,  # Number of trees, change as needed
            max_depth=6,  # Depth of each tree, change as needed
            learning_rate=0.1,  # Step size shrinkage used to prevent overfitting. Range is [0,1]
            subsample=0.8,  # Subsample ratio of the training instances, change as needed
            colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree, change as needed
            objective='binary:logistic'  # Specify binary classification
        )

        # Use cross-validation to evaluate model performance
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_val_score(self.model, self.X_train, self.y_train, cv=kfold, scoring='accuracy')
        print(f'XGBoost Cross-Validation Accuracy: {cv_results.mean()} (+/- {cv_results.std()})')

        # Fit the model on the entire dataset
        self.model.fit(self.X_train, self.y_train)

        # Predict probabilities and classes for the test set
        y_pred_proba = self.model.predict_proba(self.X_test)
        y_pred = self.model.predict(self.X_test)

        # Calculate accuracy and log loss
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, y_pred_proba)

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
        self.model = SVC()
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
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


def huber_loss(y_true, y_pred):
    error = tf.cast(tf.argmax(y_true) - tf.argmax(y_pred), tf.float32)
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 1
    return tf.where(is_small_error, squared_loss, linear_loss)


if __name__ == '__main__':
    mt = ModelTraining(database_path="MarketDepth_data_sample.csv")
    mt.preprocess_data()
    accuracy, loss = mt.MLPClassifier_model()
    print(accuracy, loss)
