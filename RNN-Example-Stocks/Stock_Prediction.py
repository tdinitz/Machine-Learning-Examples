import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import  train_test_split
import time #helper libraries
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis
from pandas import DataFrame
from pandas import Series
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--stock", required=True,
	help="Ticker symbol for stock")
args = vars(ap.parse_args())

###### Define functions to be called below #####

# Take stock ticker, and ouput the data for that stock
def retrieve_data(stock):
	text=open("Stocks/"+stock+".us.txt").read()
	dataset=text.splitlines()[1:] #remove header
	dataset=np.array([[float(num) for num in line.split(',')[1:-1]] for line in dataset])
	return dataset

#This is hard-coded in for now
close_index=3

#Create a differenced series
def difference(dataset,interval=1):
	return dataset[interval:]-dataset[0:-interval]

# convert data to be input and output for neural net
# i.e. X=t and Y=t+1
def create_dataset(dataset, look_back=1, close_index=close_index):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		input_data = dataset[i:(i+look_back)]
		input_data = [data[close_index] for data in input_data]
		dataX.append(input_data)
		dataY.append(dataset[i + look_back, close_index])
	return np.array(dataX), np.array(dataY)


# Retrieve stock data and turn it into list of differences
stock=args["stock"]
data=retrieve_data(stock)
data=difference(data)
print(len(data))

# Split data into train and test sets
train_set=data
test_set=data

# Extract relevant data from train and test sets
look_back = 1
trainX, trainY = create_dataset(train_set, look_back)
testX, testY = create_dataset(test_set, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#Step 2 Build Model
model = Sequential()

model.add(LSTM(
    input_dim=1,
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')


#Train Model
model.fit(trainX,trainY,batch_size=128,nb_epoch=10,validation_split=0.05)

##### THINGS TO ADD
# Scale values (optional)
# Split train and test sets
# more nuanced training
# testing



