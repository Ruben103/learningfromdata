from keras.models import Sequential
from keras.layers.core import Dense
from numpy import save, argmax, sqrt
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.optimizers import sgd, Adam, rmsprop, adamax
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Data import Data

class Classifier:

    def __init__(self, type, nb_features, nb_classes, epochs, batch_size, classes, run_number, rate=0.4):
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.run_number = run_number
        self.classes = classes

        if type == 'MLP':
            self.MLP()
        elif type == 'DropoutAdamax':
            self.DropoutAdamax()
        elif type == 'DropoutAdam':
            self.DropoutAdam(rate)
        elif type == 'DropoutSGD':
            self.DropoutSGD()
        elif type == 'BatchsizeAdam':
            self.BatchsizeAdam()

    def MLP(self):
        # Build the model
        print("Building model...")
        model = Sequential()
        # Single 500-neuron hidden layer with sigmoid activation
        model.add(Dense(input_dim=self.nb_features, units=500, activation='sigmoid'))
        # Output layer with softmax activation
        model.add(Dense(units=self.nb_classes, activation='softmax'))
        # Specify optimizer, loss and validation metric
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        self.model = model
        model.summary()

    def DropoutAdamax(self):
        # Build the model
        print("Building model...")
        model = Sequential()
        # Single 500-neuron hidden layer with sigmoid activation
        model.add(Dropout(rate=0.4, input_shape=(self.nb_features,)))
        model.add(Dense(input_dim=self.nb_features, units=int(self.nb_features / 0.4), activation='relu'))
        model.add(Dropout(rate=0.4, input_shape=(int(self.nb_features / 0.4),)))
        # Output layer with softmax activation
        model.add(Dense(units=self.nb_classes, activation='softmax'))
        # Specify optimizer, loss and validation metric
        opt = adamax(lr=0.0015)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        self.model = model

    def DropoutSGD(self):
        # Build the model
        print("Building model...")
        model = Sequential()
        # Single 500-neuron hidden layer with sigmoid activation
        model.add(Dropout(rate=0.4, input_shape=(self.nb_features,)))
        model.add(Dense(input_dim=self.nb_features, units=int(self.nb_features / 0.4), activation='relu'))
        model.add(Dropout(rate=0.4, input_shape=(int(self.nb_features / 0.4),)))
        # Output layer with softmax activation
        model.add(Dense(units=self.nb_classes, activation='softmax'))
        # Specify optimizer, loss and validation metric
        opt = sgd(lr=0.03, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        self.model = model

    def DropoutAdam(self, rate):
        # Build the model
        print("Building model...")
        model = Sequential()
        # Single 500-neuron hidden layer with sigmoid activation
        model.add(Dropout(rate=rate, input_shape=(self.nb_features,)))
        model.add(Dense(input_dim=self.nb_features, units=int(self.nb_features / rate), activation='relu'))
        model.add(Dropout(rate=rate, input_shape=(int((self.nb_features / rate) / rate),)))
        # Output layer with softmax activation
        model.add(Dense(units=self.nb_classes, activation='softmax'))
        # Specify optimizer, loss and validation metric
        opt = Adam(lr=0.0005)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model = model
        model.summary()


    def BatchsizeAdam(self):
        # Build the model
        print("Building model...")
        model = Sequential()
        # Single 500-neuron hidden layer with sigmoid activation
        model.add(Dropout(rate=0.4, input_shape=(self.nb_features,)))
        model.add(Dense(input_dim=self.nb_features, units=int(self.nb_features / 0.4), activation='relu'))
        model.add(Dropout(rate=0.4, input_shape=(int(self.nb_features / 0.4),)))
        # Output layer with softmax activation
        model.add(Dense(units=self.nb_classes, activation='softmax'))
        # Specify optimizer, loss and validation metric
        opt = Adam(lr=0.0005)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        self.model = model

    def fit(self, X_train, Y_train, X_dev, Y_dev, return_history=True):
        # Train the model
        callback = EarlyStopping(monitor='val_accuracy', patience=3)
        history = self.model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(X_dev, Y_dev), shuffle=True, verbose=1, callbacks=[callback])

        if return_history:
            return history

    def predict(self, X_test, X_dev, Y_dev, save_predictions=True, confusion_matrix=False):
        outputs = self.model.predict(X_test, batch_size=self.batch_size)
        pred_classes = argmax(outputs, axis=1)

        if save_predictions:
            # Save predictions to file
            save('test_set_predictions_run{0}'.format(self.run_number), pred_classes)

        Y_dev = argmax(Y_dev, axis=1)
        acc = accuracy_score(pred_classes, Y_dev)
        prec = precision_score(pred_classes, Y_dev, average='macro')
        reca = recall_score(pred_classes, Y_dev, average='macro')
        f1 = f1_score(pred_classes, Y_dev, average='macro')

        return [acc, prec, reca, f1]
