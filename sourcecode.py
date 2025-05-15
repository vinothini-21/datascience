from google.colab import files
import pandas as pd

df = pd.read_csv("mnist_test (1).csv")
df.head()
df.info()
df.describe()
print("Missing values:\n", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())
import matplotlib.pyplot as plt
import numpy as np

# Visualize one example digit
def plot_digit(index):
    pixels = df.drop('label', axis=1).iloc[index].values.reshape(28, 28)
    plt.imshow(pixels, cmap='gray')
    plt.title(f"Label: {df.iloc[index]['label']}")
    plt.axis('off')
    plt.show()

plot_digit(0)
X = df.drop('label', axis=1)
y = df['label']
from tensorflow.keras.utils import to_categorical

y_encoded = to_categorical(y, num_classes=10)
X = X / 255.0
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential([
    Flatten(input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
pred = model.predict(X_test[:5])
print("Predicted:", pred.argmax(axis=1))
import numpy as np
model.predict(X_test[0:1]).argmax()
!pip install gradio
import gradio as gr
def predict_digit(image):
    image = image.reshape(1, 784) / 255.0
    prediction = model.predict(image)
    return int(np.argmax(prediction))
inputs = [gr.Number(label=col) for col in X.columns]
interface = gr.Interface(fn=predict_digit, inputs=inputs, outputs="text", title="🎓Student Performance Predictor")
interface.launch()

