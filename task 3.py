#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the Neural Network Model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 image into a 1D vector
    Dense(128, activation='relu'),   # Fully connected layer with 128 units and ReLU activation
    Dense(10, activation='softmax')  # Output layer with 10 units for 10 possible digits
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
model.fit(x_train, y_train, epochs=10)

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)

# Make Predictions
sample_index = 0
sample_image = x_test[sample_index]
predicted_probs = model.predict(sample_image.reshape(1, 28, 28))
predicted_digit = tf.argmax(predicted_probs, axis=1).numpy()

print("Predicted digit:", predicted_digit)


# In[ ]:





# In[ ]:




