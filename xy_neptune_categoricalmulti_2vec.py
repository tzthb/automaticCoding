import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization, LayerNormalization, MultiHeadAttention, Input, Bidirectional, Reshape
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
from sklearn.metrics import f1_score

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

one_hot_multiclass_classes = MultiLabelBinarizer()
expanded_df1 = split_and_expand_df(df1)
multi_labels_classes = one_hot_multiclass_classes.fit_transform(expanded_df1['llt_code'])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, multi_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the input layer
inputs = Input(shape=(max_len,))
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=True)(inputs)
# Define the MultiHeadAttention layer
x = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)

# Define the rest of the model
x = LSTM(units=128, return_sequences=True, recurrent_dropout=0.5)(x)
# Batch Normalization
x = LayerNormalization()(x)
# Another Bidirectional LSTM layer with dropout
x =LSTM(units=128, return_sequences=True, recurrent_dropout=0.5)(x)
# Batch Normalization
x = LayerNormalization()(x)
x=LSTM(units=128, recurrent_dropout=0.2)(x)
x = Dropout(0.4)(x)
# Batch Normalization
x = LayerNormalization()(x)
x = Reshape((1, 128))(x)
x= LSTM(units=128, return_sequences=True)(x)
x = LayerNormalization()(x)

# Final LSTM layers with dropout
x = LSTM(units=64, return_sequences=True)(x)
x = Dropout(0.6)(x)
x = LayerNormalization()(x)
x = LSTM(units=64, return_sequences=False)(x)
# Dense layer with regularization
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.5))(x)
# Dense layer with regularization
x =Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.5))(x)

# Output layer
outputs = Dense(units=len(one_hot_multiclass.classes_), activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train.flatten()), y=y_train.flatten())
class_weights_dict = dict(enumerate(class_weights))

history = model.fit(X_train, y_train, class_weight=class_weights_dict, epochs=5, batch_size=256, validation_data=(X_val, y_val), callbacks=[early_stopping])


# Train the model
#history = model.fit(X_train, y_train, epochs=5, batch_size=256, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)

# Log additional custom metrics
y_pred = model.predict_classes(X_test)
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
model.save('lstm_multilabel_model1.h5')

# Print model summary
print(model.summary())
# Plot training history
plt.figure(figsize=(10, 8))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Performance')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.savefig('model_performance.png')

# Predict value
text = "Husten und Halsschmerzen "
text = preprocess_german_text(text)
sequence = tokenizer.texts_to_sequences([text])
padded_sequence = pad_sequences(sequence, maxlen=max_len)
prediction = model.predict_classes(padded_sequence)
print(prediction)
#Convert prediction back to llt_code
print(one_hot_multiclass.inverse_transform(prediction))