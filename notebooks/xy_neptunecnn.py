import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D

# Initialize Neptune run
run = neptune.init_run(
    name="Keras text classification",
    tags=["keras", "script"],
    dependencies="requirements.txt",
    capture_stderr=True,
    capture_stdout=True,
    capture_hardware_metrics=True,
    project="tzthb/automatic-coding",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2YWMyZThkOS1iYTc3LTQ4YzEtYTE3Yi0xNTVmMTY2MjJiODIifQ==",
)

# Log data parameters
data_params = {
    "batch_size": 128,
    "validation_split": 0.3,
    "max_features": 10000,
    "embedding_dim": 128,
    "sequence_length": 150,
    "seed": 42,
}
run["data/params"] = data_params

# Define the paths to your CSV files
csv_file_path_1 = 'data.csv'
csv_file_path_2 = 'AEcodiert240430_UTF8.csv'

# Read the CSV files into DataFrames
try:
    df1 = pd.read_csv(csv_file_path_1, delimiter=';', encoding='utf-8')
    df2 = pd.read_csv(csv_file_path_2, delimiter=';', encoding='utf-8')
except FileNotFoundError as e:
    print(e)
    exit()

df = pd.concat([df1, df2], ignore_index=True)

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
    return ' '.join(words)

expanded_df['ae_description'] = expanded_df['ae_description'].apply(preprocess_german_text)

# Tokenization and Sequencing
max_words = 10000
max_len = 150

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(expanded_df['ae_description'])

sequences = tokenizer.texts_to_sequences(expanded_df['ae_description'])
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# One-hot encode multiclass feature
one_hot_multiclass = MultiLabelBinarizer()
multi_labels = one_hot_multiclass.fit_transform(expanded_df['llt_code'])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, multi_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

# Define the improved model
embedding_layer = Embedding(max_words, data_params["embedding_dim"], input_length=data_params["sequence_length"])
model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(50, return_sequences=True)))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(GlobalMaxPooling1D(128))
model.add(Dense(128, 64, activation='relu'))
model.add(GlobalMaxPooling1D(64))
model.add(Dense(64, 16, activation='relu'))
model.add(GlobalMaxPooling1D(16))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(one_hot_multiclass.classes_), activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Neptune callback
neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=data_params["batch_size"],
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, neptune_cbk]
)

# Evaluate the model on the test set
results = model.evaluate(X_test, y_test, batch_size=data_params["batch_size"])
loss, accuracy, precision, recall = results[0], results[1], results[2], results[3]

print('Test accuracy with improved model:', accuracy)

run["eval/test_loss"] = loss
run["eval/test_accuracy"] = accuracy
run["eval/test_precision"] = precision
run["eval/test_recall"] = recall

# Log additional custom metrics
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

f1 = f1_score(y_test, y_pred_classes, average='weighted')
precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')

run["eval/f1_score"] = f1
run["eval/precision"] = precision
run["eval/recall"] = recall

# Confusion matrix
conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred_classes.argmax(axis=1))
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
run["eval/confusion_matrix"].upload('confusion_matrix.png')

# Save the model
model.save('improved_model.h5')

# Initialize model version in Neptune
project_key = run["sys/id"].fetch().split('-')[0]
model_key = "IMPROVED_CNN"

try:
    model_entity = neptune.init_model(name="improved_cnn_model", key=model_key)
    model_entity.stop()
except neptune.exceptions.NeptuneModelKeyAlreadyExistsError:
    pass

model_version = neptune.init_model_version(model=f"{project_key}-{model_key}", name="improved_cnn_model")

# Log model and weights
model_version["serialized_model"] = model.to_json()
model.save_weights("model_weights.h5")
model_version["model_weights"].upload("model_weights.h5", wait=True)
model_version.change_stage("staging")

# Stop Neptune runs
run.stop()
model_version.stop()