import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

data_path = 'history_bulk.csv'
df = pd.read_csv(data_path)

data = df['temp'].values

mean_data = np.mean(data)
std_data = np.std(data)

data = (data - mean_data) / std_data

sequence_length = 10 
X = []
y = []

for i in range(len(data) - sequence_length):
    X.append(data[i:i + sequence_length])
    y.append(data[i + sequence_length])

X = np.array(X)
y = np.array(y)

split_ratio = 0.8 
split_index = int(split_ratio * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(sequence_length,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

max_epochs = 500 
batch_size = 32
validation_data = (X_test, y_test)

epoch = 0
min_loss = float('inf') 
target_loss = 0.15

while min_loss > target_loss and epoch < max_epochs:

    history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, validation_data=validation_data, verbose=1)
    

    val_loss = history.history['val_loss'][0]
    

    if val_loss < min_loss:
        min_loss = val_loss
    

    epoch += 1
    print(f'Epoch {epoch}, Validation Loss: {val_loss}')


    if min_loss <= target_loss:
        print(f'Stopped training after {epoch} epochs. Validation loss: {min_loss}')
        break

y_pred = model.predict(X_test)

y_test_original = y_test * std_data + mean_data 

y_pred_original = y_pred * std_data + mean_data 

 

# Wykres oryginalnych wartości 

plt.figure(figsize=(12, 6)) 

plt.plot(y_test_original, label='Temperature from Data') 

plt.plot(y_pred_original, label='Predicted Temperature') 

plt.legend() 

plt.show() 