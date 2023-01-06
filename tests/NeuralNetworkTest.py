import numpy as np
from keras.models import Sequential
from keras.layers import Dense


# load the dataset
dataset = np.loadtxt('TrapTrainingSet.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:50]
y = dataset[:,50]

model = Sequential()
model.add(Dense(24, input_dim=50, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=150, batch_size=20)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

predictions = (model.predict(X) > 0.5).astype(int)

for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))