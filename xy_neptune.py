import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
run = neptune.init_run(
    capture_stderr=True,
    capture_stdout=True,
    capture_hardware_metrics=True,
    project="tzthb/automatic-coding",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2YWMyZThkOS1iYTc3LTQ4YzEtYTE3Yi0xNTVmMTY2MjJiODIifQ==",
)

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
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

# Log parameters
params = {
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "epochs": 10,
    "batch_size": 128
}
run["parameters"] = params

# Define the paths to your CSV files
csv_file_path_1 = 'data.csv' # Replace with your actual first file path
csv_file_path_2 = 'AEcodiert240430_UTF8.csv' # Replace with your actual second file path

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

# Preprocessing text
nltk_data_path = '/path/to/nltk_data'
os.environ["NLTK_DATA"] = nltk_data_path

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
l1_l2_regularizer = regularizers.l1_l2(l1=0.01, l2=0.01)

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=128, kernel_regularizer=l1_l2_regularizer))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=len(one_hot_multiclass.classes_), activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Neptune callback
neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.5, callbacks=[early_stopping, neptune_cbk])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
run["eval/test_loss"] = loss
run["eval/test_accuracy"] = accuracy

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
model.save('lstm_multilabel_model1.h5')
run["model"].upload('lstm_multilabel_model1.h5')

# Print model summary
print(model.summary())
run["model_summary"].upload(model.summary())

model_version = neptune.init_model_version(
    model="AUT-MOD5",
    project="tzthb/automatic-coding",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2YWMyZThkOS1iYTc3LTQ4YzEtYTE3Yi0xNTVmMTY2MjJiODIifQ==", # your credentials
)
model_version["model"].upload('./lstm_multilabel_model1.h5')
model_version["train/dataset"].track_files("s3://datasets/train")
model_version["validation/acc"] = accuracy
model_version.change_stage("staging")
# Stop the Neptune run
run.stop()
