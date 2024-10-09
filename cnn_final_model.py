from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras_tuner import Hyperband
import gensim
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import keras_tuner as kt

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
try:
    df1 = pd.read_csv(csv_file_path_1, delimiter=';', encoding='utf-8')
    df2 = pd.read_csv(csv_file_path_2, delimiter=';', encoding='utf-8')
    df3 = pd.read_csv(csv_file_path_3, delimiter=';', encoding='utf-8')
    df4 = pd.read_csv(csv_file_path_4, delimiter=';', encoding='utf-8')

except FileNotFoundError as e:
    print(e)
    exit()

# Concatenate all the DataFrames
df = pd.concat([ df1, df2, df3, df4], ignore_index=True)

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

# Load your dataframe

# Expand and preprocess data
expanded_df = split_and_expand_df(df)
expanded_df['ae_description'] = expanded_df['ae_description'].apply(preprocess_german_text)

print('Dataframe loaded and splitted.')
# Tokenization and Sequencing
# Beispiel: Ein Blick auf die ersten 10 Zeilen der ursprünglichen Labels
print(expanded_df['llt_code'].head(10))
# Anzahl der einzigartigen Labels (Kategorien)
unique_labels = expanded_df['llt_code'].nunique()
print(f"Anzahl einzigartiger Labels: {unique_labels}")
# Verteilung der Labels
label_distribution = expanded_df['llt_code'].value_counts()
print(label_distribution)

### Step 1: Text Tokenization
max_vocab_size = 10000  # Vocabulary size
max_sequence_length = 100  # Max sequence length

# Tokenizer for text (convert text to sequences of integers)
tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(expanded_df['ae_description'])

# Convert texts to sequences of integers
sequences = tokenizer.texts_to_sequences(expanded_df['ae_description'])

# Pad sequences so all inputs have the same length
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

### Step 2: Label Encoding (Integer Encoding)
# Use LabelEncoder to transform labels into integer-encoded format
label_encoder = MultiLabelBinarizer()
integer_encoded_labels = label_encoder.fit_transform(expanded_df['llt_code'])
# Überprüfen der kodierten Labels
print("Shape of Encoded Labels:", integer_encoded_labels.shape)
print("First 5 Encoded Labels:", integer_encoded_labels[:5])

import pickle

# After fitting the label encoder during training
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)
print('saved labels')
# Überprüfen, ob jede Klasse korrekt vertreten ist
class_labels = label_encoder.classes_
print("Klassennamen:", class_labels)
# Suche nach ähnlichen oder doppelten Label-Klassen
duplicate_labels = expanded_df[expanded_df.duplicated(['llt_code'], keep=False)]
print("Duplikate in den Labels:")
print(duplicate_labels)

### Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, integer_encoded_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print('Data splitted into train and test sets.')
# Prepare embedding matrix
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
num_classes = len(label_encoder.classes_)
import numpy as np
from tensorflow.keras.layers import Embedding
import gensim

# Load the pre-trained Word2Vec model (assuming you have it saved)
word2vec_model_path = 'word2vec.model'  # Replace with your Word2Vec model path
word2vec = gensim.models.Word2Vec.load(word2vec_model_path)
print(f"Word2Vec model loaded from {word2vec_model_path}")

# Prepare the embedding matrix
embedding_dim = word2vec.vector_size
vocab_size = len(tokenizer.word_index) + 1  # This needs to match the tokenizer used before

# Initialize the embedding matrix with zeros
embedding_matrix = np.zeros((vocab_size, embedding_dim))

# Populate the embedding matrix with Word2Vec embeddings
for word, index in tokenizer.word_index.items():
    if word in word2vec.wv:
        embedding_matrix[index] = word2vec.wv[word]
    else:
        # If word not found in Word2Vec, keep the corresponding row as zeros
        embedding_matrix[index] = np.zeros(embedding_dim)

# Now you can use the embedding matrix in the Keras Embedding layer
print(f"Embedding matrix created with shape: {embedding_matrix.shape}")

# Model Building with pre-trained Word2Vec embeddings
def build_model(hp):
    model = Sequential()

    # Use pre-trained Word2Vec embeddings
    model.add(Embedding(input_dim=vocab_size, 
                        output_dim=embedding_dim, 
                        input_length=max_sequence_length, 
                        weights=[embedding_matrix],  # Use the embedding matrix
                        trainable=False))  # Set to False to avoid changing pre-trained embeddings

    # Conv1D layers as in your existing code
    for i in range(hp.Int('num_conv_layers', 1, 3)):
        model.add(Conv1D(filters=hp.Choice('filters_' + str(i), values=[64, 128, 256]), kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize tuner
tuner = Hyperband(build_model, objective='val_accuracy', max_epochs=10, factor=3)

# Run the tuner search
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Save the model
best_model.save('best_cnn_model.h5')
best_model.save('best_cnn_model.keras')

print("Model saved as 'best_cnn_model.h5'")

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Predict on test data
y_pred_prob = best_model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate Precision, Recall, and F1-Score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1)

# Test the model on a sample text
sample_text = "Erkältung mit Husten und Halsschmerzen"
processed_sample = pad_sequences(tokenizer.texts_to_sequences([preprocess_german_text(sample_text)]), maxlen=max_sequence_length)

# Predict and convert to llt_code
predictions = best_model.predict(processed_sample)
print("Vorhergesagter llt_code für 'Erkältung mit Husten und Halsschmerzen':", predictions[0])
prediction = (best_model.predict(X_test) > 0.5).astype("int32")
print(prediction)
predicted_labels = np.argmax(predictions, axis=1)
print(predicted_labels)
print("Vorhergesagter llt_code:", predictions[0], "\n", label_encoder.inverse_transform(predictions[0]))
predicted_llt_code = label_encoder.inverse_transform(predictions)

print("Vorhergesagter llt_code für 'Husten und Fieber':", predicted_llt_code[0])

# Confusion Matrix
conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

# Plot the Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Print classification report
classification_report = tf.keras.metrics.classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(classification_report)