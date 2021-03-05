import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
 
import pandas as pd

class NeuralNet():

    def __init__(self, dfMovieData):
        
        self.listTrainInput = []
        self.listTrainOutput = []
        self.dfMovieData = dfMovieData
        
        self.model = Sequential()
        
        self.listMovieGenres = dfMovieData.columns[4:4 + 19].tolist() #The genres the NeuralNet will check for.  

        self.addLayers()

    def addMovie(self, dfMovie, bLiked):
        self.listTrainInput.append(dfMovie[self.listMovieGenres].to_numpy())
        self.listTrainOutput.append([int(bLiked)]) # We do int() just so we can get a "1" instead of "True" etc.
    
    def addLayers(self):
        self.model.add(Dense(3, activation='relu', input_dim=19))     # first hidden layer
        self.model.add(Dense(3, activation='relu'))                   # second hidden layer
        self.model.add(Dense(1, activation='sigmoid'))                # output layer

    def Compile(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy') # Train the neural networks model
            
    def Train(self):
        arrayTrainInput = np.array(self.listTrainInput, "uint8")
        arrayTrainOutput = np.array(self.listTrainOutput, "uint8")

        self.model.fit(arrayTrainInput, arrayTrainOutput, epochs=5000, verbose = True)

    def getPrediction(self, movie):
        movie = self.dfMovieData.sample()
        arrayMovie = np.array(movie[self.listMovieGenres].to_numpy(), "uint8")
        return self.model.predict([arrayMovie])
