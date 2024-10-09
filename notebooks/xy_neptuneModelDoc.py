import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
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
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Concatenate, Flatten, Dense

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
    "batch_size": 64,
    "epochs": 10,
    "validation_split": 0.5,
    "max_features": 10000,
    "embedding_dim": 128,
    "learning-rate": 0.001
}
run["parameters"] = data_params

# Custom Callback to log the metrics in neptune for each epoch in training
class CustomNeptuneCallback(Callback):
  def __init__(self, run):
    super().__init__()
    self.run = run
  def on_epoch_end(self, epoch, logs=None):
    self.run['accuracy'].append( logs['accuracy'])
    self.run['loss'].append( logs['loss'])
    self.run['val_loss'].append( logs['val_loss'])
    self.run['val_accuracy'].append( logs['val_accuracy'])

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
max_len = 200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(expanded_df['ae_description'])

sequences = tokenizer.texts_to_sequences(expanded_df['ae_description'])
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# One-hot encode multiclass feature
one_hot_multiclass = MultiLabelBinarizer()
multi_labels = one_hot_multiclass.fit_transform(expanded_df['llt_code'])

np_callback = CustomNeptuneCallback(run)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, multi_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=data_params["validation_split"], random_state=42)

# Assuming max_words and embedding_dim are already defined
max_len = 200  # Set the sample length to 200

# Create the embedding layer
embedding_layer = Embedding(max_words, data_params["embedding_dim"], input_length=max_len)

# Define the input
inputs = Input(shape=(max_len,))
embedded = embedding_layer(inputs)

# Define multiple convolutional layers with different filter sizes
conv_1 = Conv1D(128, 1, activation='relu')(embedded)
conv_2 = Conv1D(128, 2, activation='relu')(embedded)
conv_3 = Conv1D(128, 3, activation='relu')(embedded)
conv_4 = Conv1D(128, 4, activation='relu')(embedded)
conv_5 = Conv1D(128, 5, activation='relu')(embedded)

# Apply max pooling to each convolutional layer
pool_1 = GlobalMaxPooling1D()(conv_1)
pool_2 = GlobalMaxPooling1D()(conv_2)
pool_3 = GlobalMaxPooling1D()(conv_3)
pool_4 = GlobalMaxPooling1D()(conv_4)
pool_5 = GlobalMaxPooling1D()(conv_5)

# Concatenate the pooled features
concatenated = Concatenate()([pool_1, pool_2, pool_3, pool_4, pool_5])

# Add a dense layer for classification
dense = Dense(128, activation='relu')(concatenated)
output = Dense(len(one_hot_multiclass.classes_), activation='softmax')(dense)

# Create the model
model = Model(inputs=inputs, outputs=output)

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=data_params["batch_size"],
    epochs=data_params["epochs"],
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, np_callback]
)

# Evaluate the model on the test set
results = model.evaluate(X_test, y_test, batch_size=data_params["batch_size"])
loss, accuracy, precision, recall = results[0], results[1], results[2], results[3]

print('Test accuracy with improved model:', accuracy)

# Log additional custom metrics
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

f1 = f1_score(y_test, y_pred_classes, average='weighted', zero_division=0)
precision = precision_score(y_test, y_pred_classes, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_classes, average='weighted', zero_division=0)
run["logs/accuracy"] = accuracy
run["logs/f1_score"] = f1
run["logs/precision_score"] = precision
run["logs/recall_score"] = recall

# Confusion matrix
conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred_classes.argmax(axis=1))
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()  # Add this line
run["val/conf_matrix"].upload("confusion_matrix.png")
# Save the model
model.save_weights("model_weights.weights.h5")
model_json = model.to_json()


modeln = neptune.init_model(
    model="AUT-CNN",
    project="tzthb/automatic-coding",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2YWMyZThkOS1iYTc3LTQ4YzEtYTE3Yi0xNTVmMTY2MjJiODIifQ==", # your credentials
)
model_version = neptune.init_model_version(
    model="AUT-CNN",
    project="tzthb/automatic-coding",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2YWMyZThkOS1iYTc3LTQ4YzEtYTE3Yi0xNTVmMTY2MjJiODIifQ==", # your credentials
)

with open("model_structure.json", "w") as json_file:
    json_file.write(model_json)
model_version["model/signature"].upload("model_structure.json")
model_version["validation/dataset/v0.1"].track_files("s3://datasets/validation")

# Stop Neptune runs
run.stop()
