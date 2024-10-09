import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import os
from tensorflow.keras import regularizers
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re
import numpy as np
from gensim.models import Word2Vec

# Log parameters
params = {
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "epochs": 10,
    "batch_size": 128
}

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
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Function to split and expand DataFrame
def split_and_expand_df(df):
    def split_row(row):
        llt_code_list = str(row['llt_code']).split(',')
        return pd.DataFrame({'llt_code': llt_code_list, 'ae_description': row['ae_description']})

    df = df.apply(split_row, axis=1)
    df = pd.concat(df.tolist(), ignore_index=True)
    return df

expanded_df = split_and_expand_df(df)

stop_words = set(stopwords.words('german'))
stemmer = SnowballStemmer('german')

def preprocess_german_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text, language='german')
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return words

expanded_df['ae_description'] = expanded_df['ae_description'].apply(preprocess_german_text)

# Tokenization and Sequencing
max_words = 10000
max_len = 150

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(expanded_df['ae_description'])

sequences = tokenizer.texts_to_sequences(expanded_df['ae_description'])
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Prepare embedding matrix
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
embedding_matrix = np.zeros((vocab_size, embedding_dim))

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=expanded_df['ae_description'], vector_size=embedding_dim, window=5, min_count=1, workers=4)
word2vec_model.save("word2vec.model")

for word, i in tokenizer.word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

# One-hot encode multiclass feature
one_hot_multiclass = MultiLabelBinarizer()
multi_labels = one_hot_multiclass.fit_transform(expanded_df['llt_code'])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, multi_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

l1_l2_regularizer = regularizers.l1_l2(l1=0.01, l2=0.01)

# Define the updated LSTM model with Multi-Head Attention using Functional API
input_layer = Input(shape=(max_len,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(input_layer)

# LSTM layers
lstm_layer_1 = LSTM(units=128, kernel_regularizer=l1_l2_regularizer, return_sequences=True)(embedding_layer)
lstm_layer_2 = LSTM(units=128, return_sequences=True)(lstm_layer_1)

# Multi-Head Attention Layer
attention_output, attention_scores = MultiHeadAttention(num_heads=4, key_dim=128)(lstm_layer_2, lstm_layer_2, return_attention_scores=True)
attention_output = Add()([attention_output, lstm_layer_2])  # Residual connection
attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

# Further LSTM layers after attention
lstm_layer_3 = LSTM(units=128, kernel_regularizer=l1_l2_regularizer)(attention_output)
dropout_layer = Dropout(0.4)(lstm_layer_3)

# Output layer for multi-label classification
output_layer = Dense(units=len(multi_labels.classes_), activation='sigmoid')(dropout_layer)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=256, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)

# Log additional custom metrics
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

f1 = f1_score(y_test, y_pred_classes, average='weighted')
precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
print('F1 Score: ', f1)
print('Precision: ', precision)
print('Recall: ', recall)

# Confusion matrix
conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred_classes.argmax(axis=1))
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# Save the model
model.save('lstm_multilabel_model_with_attention.h5')

# Print model summary
print(model.summary())

# Define the preprocessing function for new input text
def preprocess_input_text(text):
    words = preprocess_german_text(text)  # Use the same preprocessing function
    return tokenizer.texts_to_sequences([words])[0]

# Define a function to make a prediction
def predict_llt_code(text):
    # Step 1: Preprocess the input text
    processed_text = preprocess_input_text(text)
    
    # Step 2: Pad the sequence
    padded_text = pad_sequences([processed_text], maxlen=max_len)
    
    # Step 3: Make prediction
    prediction = model.predict(padded_text)
    
    # Step 4: Convert probabilities to binary labels
    predicted_labels = (prediction > 0.5).astype(int)
    
    # Decode the predicted labels to llt_code
    predicted_codes = np.array(one_hot_multiclass.classes_)[predicted_labels[0] == 1]
    
    return predicted_codes

# Example usage
input_text = "Erkältung mit Husten und Halsschmerzen"
predicted_llt_codes = predict_llt_code(input_text)
print("Predicted LLT Codes:", predicted_llt_codes)
