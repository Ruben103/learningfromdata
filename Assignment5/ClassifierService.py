from keras.models import Sequential
from keras.layers.core import Dense
from numpy import save, argmax

from Data import Data

class Classifier:

    def __init__(self, type, nb_features, nb_classes, epochs, batch_size, classes, run_number):
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.run_number = run_number
        self.classes = classes

        if type == 'MLP':
            self.MLP()

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

    def fit(self, X_train, Y_train, X_dev, Y_dev, return_history=True):
        # Train the model
        history = self.model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(X_dev, Y_dev), shuffle=True, verbose=1)

        if return_history:
            return history

    def predict(self, Xtest, X_dev, Y_dev, save_predictions=True, confusion_matrix=True):
        pred_classes = self.model.predict(Xtest, batch_size=self.batch_size)

        # Make confusion matrix on development data
        if confusion_matrix:
            Y_dev_names = [self.classes[x] for x in argmax(Y_dev, axis=1)]
            pred_dev = self.model.predict(X_dev, batch_size=self.batch_size)
            pred_class_names = [self.classes[x] for x in argmax(pred_dev, axis=1)]
            Data().create_confusion_matrix(Y_dev_names, pred_class_names, self.classes)

        if save_predictions:
            # Save predictions to file
            save('test_set_predictions_run{0}'.format(self.run_number), pred_classes)