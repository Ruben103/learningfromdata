from keras.models import Sequential
from keras.layers.core import Dense
from numpy import save, argmax, sqrt
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.optimizers import *
from sklearn.metrics import precision_score, recall_score, f1_score

from Data import Data

class Classifier:

    def __init__(self, type, nb_features, nb_classes, epochs, batch_size, classes, run_number, rate=0.1):
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.run_number = run_number
        self.classes = classes

        if type == 'MLP':
            self.MLP()

        if type == 'Dropout':
            self.Dropout(rate)

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

    def Dropout(self, rate):
        # Build the model
        print("Building model...")
        model = Sequential()
        # Single 500-neuron hidden layer with sigmoid activation
        model.add(Dropout(rate=rate, input_shape=(self.nb_features,)))
        model.add(Dense(input_dim=self.nb_features, units=int(2*self.nb_features), activation='relu'))
        model.add(Dropout(rate=rate, input_shape=(int(self.nb_features),)))
        model.add(Dense(input_dim=self.nb_features, units=int(4*self.nb_classes), activation='relu'))
        # Output layer with softmax activation
        model.add(Dense(units=self.nb_classes, activation='softmax'))
        # Specify optimizer, loss and validation metric
        opt = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        self.model = model

    def fit(self, X_train, Y_train, X_dev, Y_dev, return_history=True):
        # Train the model
        callback = EarlyStopping(monitor='val_accuracy', patience=5)
        history = self.model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(X_dev, Y_dev), shuffle=True, verbose=1, callbacks=[callback])

        if return_history:
            return history

    def predict(self, X_test, X_dev, Y_dev, save_predictions=True, confusion_matrix=False):
        outputs = self.model.predict(X_test, batch_size=self.batch_size)
        pred_classes = argmax(outputs, axis=1)

        # Make confusion matrix on development data
        if confusion_matrix:
            Y_dev_names = [self.classes[x] for x in argmax(Y_dev, axis=1)]
            pred_dev = self.model.predict(X_test, batch_size=self.batch_size)
            pred_class_names = [self.classes[x] for x in argmax(pred_dev, axis=1)]
            Data().create_confusion_matrix(Y_dev_names, pred_class_names, self.classes)

        if save_predictions:
            # Save predictions to file
            save('test_set_predictions_run{0}'.format(self.run_number), pred_classes)
        Y_dev = argmax(Y_dev, axis=1)
        prec = precision_score(pred_classes, Y_dev, average='macro')
        reca = recall_score(pred_classes, Y_dev, average='macro')
        f1 = f1_score(pred_classes, Y_dev, average='macro')

        return [prec, reca, f1]
