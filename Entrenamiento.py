import Dataset  # Importar el conjunto de datos
import tensorflow as tf  # Importar TensorFlow
import numpy as np
from tensorflow import keras

# Parámetros del modelo
classes = ['manzanas', 'platanos']
num_classes = len(classes)
validation_size = 0.2
img_size = 128
num_channels = 3
train_path = ".venv/training_data/"
data = Dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

# Crear el modelo con Keras
model = keras.Sequential([
    keras.layers.Input(shape=(img_size, img_size, num_channels)),
    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(data.train._images, data.train._labels, epochs=200, batch_size=32, 
          validation_data=(data.valid._images, data.valid._labels))

# Guardar el modelo en formato Keras
model.save("modelo.keras")
print("✅ Modelo guardado correctamente como modelo.keras")