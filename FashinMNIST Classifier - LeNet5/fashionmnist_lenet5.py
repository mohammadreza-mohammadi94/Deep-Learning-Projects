#-----------------------------------------------------------------------#
#                     Code By Mohammadreza Mohammadi                    #
#      Github:   https://github.com/mohammadreza-mohammadi94            #
#      LinkedIn: https://www.linkedin.com/in/mohammadreza-mhmdi/        #
#-----------------------------------------------------------------------#


# Import Libraries <a id=1></a>
import tensorflow as tf
import tensorflow.keras.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt

"""# 2. Import FashinMNIST Dataset <a id=2><a/>"""

(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()

# Check shape

print(f"X_train Shape: {X_train.shape}")
print(f"y_train Shape: {y_train.shape}")
print(f"X_test Shape: {X_test.shape}")
print(f"y_test Shape: {y_test.shape}")

"""# 3. Check Samples <a id=3></a>"""

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Create a dictionary to store one sample of each label
samples = {i: None for i in range(10)}

# Find one sample of each label
for img, label in zip(X_train, y_train):
    if samples[label] is None:  # If we haven't found a sample for this label yet
        samples[label] = img

# Plotting
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i in range(10):
    axes[i].imshow(samples[i], cmap='gray')
    axes[i].set_title(fashion_mnist_labels[i])
    axes[i].axis('off')

plt.tight_layout()
plt.show()

"""# 4. Normalization <a id=4></a>"""

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape the images to match the input shape of the model
X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

"""# 5. Model Definition <a id=5></a>

![](https://cdn.analyticsvidhya.com/wp-content/uploads/2021/03/Screenshot-from-2021-03-18-12-56-51.png)
"""

def LeNet5(input_shape, num_classes):
    # Define sequnetial
    model = tf.keras.Sequential()
    # Define Convolutional Layers
    # C1
    model.add(tf.keras.layers.Conv2D(
        filters=6,
        kernel_size=(5,5),
        strides=(1,1),
        activation='tanh',
        input_shape=input_shape,
        padding='same'
    ))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # C2
    model.add(tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(5,5),
        strides=(1, 1),
        activation='tanh'
    ))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten layer
    model.add(tf.keras.layers.Flatten())

    # Dense Layers
    # D1
    model.add(tf.keras.layers.Dense(
        units=120,
        activation='tanh'
    )),
    # D2
    model.add(tf.keras.layers.Dense(
        units=84,
        activation='tanh'
    )),

    # Output layer
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    return model

model = LeNet5((28, 28, 1), 10)
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Check Model
model.summary()

"""## 5.1 Model Training <a id=5.1></a>"""

history = model.fit(X_train,
                    y_train,
                    epochs=45,
                    batch_size=128,
                    validation_data=(X_test, y_test))

"""## 5.2 Model's Performance <a id=5.3></a>"""

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Check Accuracy of the Model
loss ,acc= model.evaluate(X_test, y_test)
print('Accuracy : ', acc)
print('Loss: ', loss)

