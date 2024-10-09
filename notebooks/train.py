import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import tensorflowjs as tfjs

# Read the data from CSV file
try:
    data = pd.read_csv("ICD10_openmed_UTF8.csv", delimiter=";", encoding="utf-8")
except FileNotFoundError:
    print("Error: CSV file not found. Please provide the correct path.")
    exit()
except pd.errors.ParserError:
    print("Error: CSV parsing error. Check the format of your data.")
    exit()

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['term'])
X = tokenizer.texts_to_sequences(data['term'])
X = pad_sequences(X, maxlen=None)  # Padding sequences dynamically

# Convert labels to numeric indices
X = data["term"]
y = data["icd10"]

# Split the data into training, test, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create the model architecture
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=X.shape[1]))
model.add(LSTM(100))
model.add(Dense(len(set(y)), activation='softmax'))  # Use the number of unique labels as the output dimension

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Save the model
model.save('nlp_model.h5')

# Convert the model to TensorFlow.js format
tfjs.converters.save_keras_model(model, 'tfjs_model')
