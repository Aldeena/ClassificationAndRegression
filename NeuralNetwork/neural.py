import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow import keras


class Neural_Network:

    def __init__(self, csv_flie_path) -> None:

        self.model_grav = []
        self.model_class = []

        # To remove the scientific notation from numpy arrays
        np.set_printoptions(suppress=True)

        self.vitals_numeric = pd.read_csv(csv_flie_path)
        self.vitals_numeric.head()

        # print(self.vitals_numeric)

    def creating_model(self):
        # Separate Target Variable and Predictor Variables
        self.target = ['Gravidade']
        self.predictors = ['qPa', 'Pulso', 'Resp']

        X = self.vitals_numeric[self.predictors].values
        y = self.vitals_numeric[self.target].values

        # print("X:", X)
        # print("Y:", y)

        ### Standardization of data ###
        predictor_scaler = StandardScaler()
        target_var_scaler = StandardScaler()

        # Storing the fit object for later reference
        predictor_scaler_fit = predictor_scaler.fit(X)
        target_var_scaler_fit = target_var_scaler.fit(y)

        # print("X:", X)
        # print("Y:", y)

        # Generating the standardized values of X and y
        X = predictor_scaler_fit.transform(X)
        y = target_var_scaler_fit.transform(y)

        print("X:", X)

        # Split the data into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        # print(X_train.shape)
        # print(y_train.shape)
        # print(X_test.shape)
        # print(y_test.shape)

        self.model_grav = Sequential([Dense(units=512, input_dim=3, activation='relu'),
                                      Dense(units=256, activation='relu'),
                                      Dense(units=128, activation='relu'),
                                      Dense(units=1, activation='linear')])

        # self.model_class = Sequential([Dense(units=512, input_dim=3, activation='relu'),
        #                                 Dense(units=256, activation='relu'),
        #                                 Dense(units=128, activation='relu'),
        #                                 Dense(units=4, activation='softmax')])

        self.model_grav.summary()
        # self.model_class.summary()

        self.model_grav.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Or rmsprop
                                loss='mean_squared_error',  # categorical_crossentropy for multiclass
                                metrics=[tf.keras.metrics.RootMeanSquaredError()])

        self.model_grav.fit(X_train, y_train, epochs=300,
                            verbose=1, validation_split=0.1)

        # Making predictions on test file
        predictions = self.model_grav.predict(X_test)

        print(predictions)

        filedata2 = pd.read_csv(
            '../teste_cego.txt')

        X_final = filedata2[self.predictors].values
        # y = output_test

        predictor_scaler_test = StandardScaler()

        # Storing the fit object for later reference
        predictor_scaler_test_fit = predictor_scaler_test.fit(X_final)

        X_final = predictor_scaler_test_fit.transform(X_final)

        # print("X:",X_final)

        pred_labels_new = self.model_grav.predict(X_final, verbose=2)

        pred_labels_new = target_var_scaler_fit.inverse_transform(
            pred_labels_new)

        with open("resultados_treino_grav2.txt", "w") as output_file:
            output_file.write(f'Grav' + '\n')
            for row in pred_labels_new:
                output_file.write(str(row) + '\n')

        # print(pred_labels_new)

        # # Scaling the predicted price data back to original price scale
        # predictions = target_var_scaler_fit.inverse_transform(predictions)

        # # Scaling the y_test Grav data back to original scale
        # y_test_origin=target_var_scaler_fit.inverse_transform(y_test)

        # # Scaling the test data back to original scale
        # test_data = predictor_scaler_fit.inverse_transform(X_test)

        # test_data = pd.DataFrame(data=test_data, columns=self.predictors)
        # test_data['Grav'] = y_test_origin
        # test_data['Predicted Grav'] = predictions
        # print(test_data)

        return

    def manual_parsing(self, filePath):
        fileContent = open(filePath, 'r')

        fileData = []
        # Puts it in a matrix
        for line in fileContent:
            fileData.append(line.split(','))

        # Removes '\n' from the last collumn
        rowLength = len(fileData[0]) - 1
        for row in range(len(fileData)):
            number = fileData[row][rowLength]
            fileData[row][rowLength] = number[:-1]

        # Casts inputs to float
        for colIterator in range(len(fileData[0])-1):
            for rowIterator in range(len(fileData)):
                try:
                    fileData[rowIterator][colIterator] = float(
                        fileData[rowIterator][colIterator])
                except:
                    pass
        # Casts outputs to int
        for rowIterator in range(len(fileData)):
            try:
                fileData[rowIterator][rowLength] = int(
                    fileData[rowIterator][rowLength])
            except:
                pass

        return fileData
