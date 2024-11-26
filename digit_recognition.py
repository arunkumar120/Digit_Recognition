
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow import keras

# Load MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalize the dataset
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Example prediction
sample_image = X_test[0]
sample_label = Y_test[0]

prediction = model.predict(sample_image.reshape(1, 28, 28))
predicted_label = np.argmax(prediction)

plt.imshow(sample_image, cmap='gray')
plt.title(f"True: {sample_label}, Predicted: {predicted_label}")
plt.show()
