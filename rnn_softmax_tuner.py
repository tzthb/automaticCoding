from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from kerastuner import HyperModel, RandomSearch
import tensorflow as tf

# Hyperparameter für die Vorverarbeitung
max_vocab_size = 10000  # Maximalgröße des Vokabulars
max_sequence_length = 100  # Maximal zulässige Länge der Sequenzen
embedding_dim = 300  # Dimension des Embeddings

# Define the paths to your CSV files
csv_file_path_1 = './data/data.csv'
csv_file_path_2 = './data/AEcodiert240430_UTF8.csv'
csv_file_path_3 = './data/meddra_zkls.csv'
csv_file_path_4 = './data/meddra_zkls2.csv'

# Read the CSV files into DataFrames
df1 = pd.read_csv(csv_file_path_1, delimiter=';', encoding='utf-8')
df2 = pd.read_csv(csv_file_path_2, delimiter=';', encoding='utf-8')
df3 = pd.read_csv(csv_file_path_3, delimiter=';', encoding='utf-8')
df4 = pd.read_csv(csv_file_path_4, delimiter=';', encoding='utf-8')

# Concatenate all the DataFrames
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Function to split and expand DataFrame
def split_and_expand_df(df):
    def split_row(row):
        llt_code_list = str(row['llt_code']).split(',')
        return pd.DataFrame({'llt_code': llt_code_list, 'ae_description': row['ae_description']})

    df = df.apply(split_row, axis=1)
    df = pd.concat(df.tolist(), ignore_index=True)
    return df

# Preprocess the text data (with stopword removal and stemming)
stop_words = set(stopwords.words('german'))
stemmer = SnowballStemmer('german')

def preprocess_german_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text, language='german')
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return words

# Expand and preprocess data
expanded_df = split_and_expand_df(df)
expanded_df['ae_description'] = expanded_df['ae_description'].apply(preprocess_german_text)

print('Dataframe loaded and splitted.')

# Tokenization and Sequencing
# Tokenizer for text (convert text to sequences of integers)
tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(expanded_df['ae_description'])

# Convert texts to sequences of integers
sequences = tokenizer.texts_to_sequences(expanded_df['ae_description'])

# Pad sequences so all inputs have the same length
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Label Encoding (Integer Encoding)
label_encoder = MultiLabelBinarizer()
integer_encoded_labels = label_encoder.fit_transform(expanded_df['llt_code'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, integer_encoded_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print('Data splitted into train and test sets.')

# KerasTuner HyperModel
class LSTMHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
        
        # Hyperparameters for LSTM layers
        for i in range(hp.Int('num_lstm_layers', 1, 5)):
            model.add(LSTM(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32), 
                           return_sequences=(i < hp.get('num_lstm_layers') - 1), 
                           kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
            model.add(Dropout(hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1)))

        model.add(Dense(len(label_encoder.classes_), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

# Initialize KerasTuner
tuner = RandomSearch(
    LSTMHyperModel(),
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='lstm_tuner',
    project_name='lstm_hyperparameter_tuning'
)

# Early Stopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Run the hyperparameter tuning
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[early_stopping])
print('Hyperparameter tuning completed.')

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]
print(best_model.summary())

# Save the best model
best_model.save('best_rnn_softmax.h5')
print('Best model saved.')

# Evaluate the best model on test data
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Make predictions
y_test_predictions = best_model.predict(X_test)

# Calculate F1 Score using TensorFlow
y_test_predictions_labels = tf.argmax(y_test_predictions, axis=1)
y_test_labels = tf.argmax(y_test, axis=1)

# Use the confusion matrix to get true positives, false positives, and false negatives
conf_matrix = confusion_matrix(y_test_labels.numpy(), y_test_predictions_labels.numpy())

# Calculate precision and recall
TP = np.diag(conf_matrix)
FP = np.sum(conf_matrix, axis=0) - TP
FN = np.sum(conf_matrix, axis=1) - TP

precision = TP / (TP + FP)  # precision for each class
recall = TP / (TP + FN)  # recall for each class
f1_scores = 2 * (precision * recall) / (precision + recall)

# Calculate the macro F1 Score
macro_f1_score = np.mean(f1_scores)

print(f"Macro F1 Score = {round(macro_f1_score, 4)}")

# Save the best model
best_model.save('best_rnn_softmax.h5')
print('Best model saved.')

# Confusion Matrix
print("Confusion Matrix:")
print(conf_matrix)

# Model Plot
from tensorflow.keras.utils import plot_model
plot = plot_model(best_model, to_file='keras_model_plot.png', show_shapes=True, show_layer_names=True)
print(plot)
