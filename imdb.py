'''
Task: IMDB movie reviews sentiment prediction
Model: Bi-directional LSTM

Each sentence in the training set is represented as a sequence of word indices
Which are collapsed down to a 128-dim embedding space via an embedding layer
Then are then fed one by one into a Bidirectional LSTM that encodes the entire sentence.
The outputs of the Bidirectional LSTM are then fed into a single unit layer that is passed
through a sigmoidal activation function that represents the probability of a given sentence being positive sentiment
'''

#imports
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.preprocessing import sequence
from keras.datasets import imdb


#set hyperparameters
max_features = 20000 #set vocabulary size, words outside of this are given OOV token
batch_size = 32 #how big SGD batches are
maxlen = 100 #maximum length of sentences in words

#get dataset
#x_train is a list of lists
#Outer list is a list of training examples
#Inner list is a list of indices, where each index represents a word
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

#The input to a Keras model must be a tensor (for matrix multiplication efficiencies) 
#so must pad or truncate sentences to max length (in words)
#also speeds up batch processing, it will arrange batches where sequences are same length
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen) #returns numpy array
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train) #keras only accepts numpy arrays, not python lists
y_test = np.array(y_test)
x_train = np.array(x_train)
x_test = np.array(x_test)

#instantiate sequential model
model = Sequential()

#add layers to model (in order! because you are using a sequential model)
model.add(Embedding(max_features, 128)) #embeddings are 128 dim vectors
model.add(Bidirectional(LSTM(64))) #LSTM layer has 64 units
model.add(Dropout(0.5)) #what proportion of inputs to set to 0
model.add(Dense(1, activation='sigmoid')) #single sigmoidal output, predicting either 0 or 1, negative or positive sentiment

#compile the model
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

#train the model
print('Train...')
hist = model.fit(x_train, y_train,
		  batch_size=batch_size,
		  epochs=1,
		  validation_data=[x_test, y_test]) 
		  #using validation_data means don't have to call model.evaluate on test set to get acc on test data
