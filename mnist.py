import numpy as np
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from tkinter import filedialog, Canvas
from PIL import Image, ImageGrab
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.utils import shuffle

# Load MNIST dataset
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Shuffle data
train_x, train_y = shuffle(train_x, train_y)

# Reshape for CNN (28x28x1 for grayscale images)
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)

# Normalize pixel values (0-255) to (0-1)
train_x, test_x = train_x / 255.0, test_x / 255.0

# Convert labels to one-hot encoding
train_yOHE = to_categorical(train_y, num_classes=10)
test_yOHE = to_categorical(test_y, num_classes=10)

# Display sample images
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
axes = axes.flatten()
for i in range(9):
    axes[i].imshow(train_x[i].reshape(28,28), cmap="gray")
    axes[i].set_title(f"Label: {train_y[i]}")
plt.show()

# Build CNN model
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
# model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
# model.add(MaxPool2D(pool_size=(2,2), strides=2))
# model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid'))
# model.add(MaxPool2D(pool_size=(2,2), strides=2))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# # Compile the model
# model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# history = model.fit(train_x, train_yOHE, batch_size=32, epochs=5, validation_data=(test_x, test_yOHE))

# # Save the model
# model.save('mnist_cnn_model.h5')

# Load the model for testing
model = load_model('mnist_cnn_model.h5')

def predict_drawing():
    x=root.winfo_rootx()+canvas.winfo_x()
    y=root.winfo_rooty()+canvas.winfo_y()
    x1=x+canvas.winfo_width()
    y1=y+canvas.winfo_height()
    img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = cv2.bitwise_not(img)
    img = img.reshape(1, 28, 28, 1) / 255.0
    prediction = np.argmax(model.predict(img))
    result_label.config(text=f"Predicted Digit: {prediction}")

def clear_canvas():
    canvas.delete("all")

def draw(event):
    x, y = event.x, event.y
    canvas.create_oval(x-10, y-10, x+10, y+10, fill='black', width=5)

# Create GUI using Tkinter
root = tk.Tk()
root.title("MNIST Digit Recognition")

canvas = Canvas(root, width=280, height=280, bg='white')
canvas.pack()
canvas.bind("<B1-Motion>", draw)

btn_predict = tk.Button(root, text="Predict", command=predict_drawing)
btn_predict.pack()

btn_clear = tk.Button(root, text="Clear", command=clear_canvas)
btn_clear.pack()

result_label = tk.Label(root, text="Predicted Digit: ", font=("Arial", 14))
result_label.pack()

root.mainloop()

# Test predictions
# predictions = model.predict(test_x[:9])
# fig, axes = plt.subplots(3, 3, figsize=(8, 8))
# axes = axes.flatten()

# for i, ax in enumerate(axes):
#     ax.imshow(test_x[i].reshape(28, 28), cmap='gray')
#     ax.set_title(f"Pred: {np.argmax(predictions[i])}")
# plt.show()



