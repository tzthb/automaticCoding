from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Embedding, Dense, Dropout, Flatten, Input, Bidirectional, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras_tuner import Hyperband
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import Word2Vec

# Hyperparameters for preprocessing
max_vocab_size = 10000  # Maximum vocabulary size
max_sequence_length = 100  # Maximum sequence length

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

# Preprocess the text data
stop_words = set(stopwords.words('german'))
stemmer = SnowballStemmer('german')

def preprocess_german_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text, language='german')
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)  # Return as a single string

# Expand and preprocess data
expanded_df = split_and_expand_df(df)
expanded_df['ae_description'] = expanded_df['ae_description'].apply(preprocess_german_text)
embedding_dim = 128
# Tokenization and Sequencing
tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(expanded_df['ae_description'])
sequences = tokenizer.texts_to_sequences(expanded_df['ae_description'])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Label Encoding
label_encoder = MultiLabelBinarizer()
integer_encoded_labels = label_encoder.fit_transform(expanded_df['llt_code'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, integer_encoded_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
num_classes = len(label_encoder.classes_)


# Prepare text corpus (already preprocessed using your `preprocess_german_text`)
sentences = [word_tokenize(text) for text in expanded_df['ae_description']]

# Train Word2Vec model on your dataset
word2vec_model = Word2Vec(sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)

# Step 2: Create embedding matrix from Word2Vec model

# Vocabulary size must match the tokenizer vocabulary size
vocab_size = min(max_vocab_size, len(tokenizer.word_index) + 1)

# Create embedding matrix where each row index corresponds to a word index from the tokenizer
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]

# Step 3: Modify the model to use pre-trained Word2Vec embeddings

def build_model_with_word2vec(hp):
    input_layer = Input(shape=(max_sequence_length,))
    
    # Initialize the embedding layer with Word2Vec embeddings (non-trainable)
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length,
                                weights=[embedding_matrix], trainable=False)(input_layer)
    
    # CNN layers
    cnn_output = embedding_layer
    for i in range(hp.Int('num_conv_layers', 1, 3)):
        cnn_output = Conv1D(filters=hp.Choice('filters_' + str(i), values=[64, 128]), kernel_size=3, activation='relu')(cnn_output)
        cnn_output = MaxPooling1D(pool_size=2)(cnn_output)
        cnn_output = Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.1, max_value=0.5, step=0.1))(cnn_output)
    
    # RNN layers (LSTM or GRU)
    rnn_output = Bidirectional(LSTM(hp.Int('units', min_value=64, max_value=256, step=64), return_sequences=True))(cnn_output)
    
    # Attention mechanism
    attention_output = Attention()([rnn_output, rnn_output])
    
    # Flatten and Dense layers
    flat_output = Flatten()(attention_output)
    dense_output = Dense(num_classes, activation='sigmoid')(flat_output)
    
    model = Model(inputs=input_layer, outputs=dense_output)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize tuner with the new model that uses Word2Vec
tuner = Hyperband(build_model_with_word2vec, objective='val_accuracy', max_epochs=10, factor=3)

# Step 4: Run the tuner search and train the model
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
# Save the model
best_model.save('mixed.h5')
best_model.save('mixed.keras')
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