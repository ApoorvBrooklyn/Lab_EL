import numpy as np
import librosa
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_and_preprocess_audio(file_path, sr=22050, duration=5):
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    if len(audio) > sr * duration:
        audio = audio[:sr * duration]
    else:
        audio = np.pad(audio, (0, sr * duration - len(audio)))
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfccs.T

def load_data(base_path):
    X = []
    y = []
    for label, animal in enumerate(['cat', 'dog']):
        animal_folder = os.path.join(base_path, animal)
        for file_name in os.listdir(animal_folder):
            if file_name.endswith('.wav'):
                file_path = os.path.join(animal_folder, file_name)
                features = load_and_preprocess_audio(file_path)
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)

# Load training data
X_train, y_train = load_data('Dataset/cats_dogs/train')

# Load testing data
X_test, y_test = load_data('Dataset/cats_dogs/test')

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Make predictions
predictions = model.predict(X_test)

# Save the model as .h5
model.save('cat_dog_audio_classifier.h5')