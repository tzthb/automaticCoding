from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from kerastuner import HyperModel, RandomSearch
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

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

# Tokenization and Sequencing
max_vocab_size = 10000  # Vocabulary size
max_sequence_length = 100  # Max sequence length

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
num_classes = len(label_encoder.classes_)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, integer_encoded_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define HyperModel class for KerasTuner
class CNNHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Embedding(max_vocab_size, 128))
        
        for i in range(hp.Int('conv_layers', 1, 5)):  # Number of Conv1D layers
            model.add(Conv1D(filters=hp.Int('filters_' + str(i), 32, 512, step=32), kernel_size=3, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
        
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

# Set up KerasTuner
tuner = RandomSearch(
    CNNHyperModel(),
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='helloworld'
)

# Perform hyperparameter tuning
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Get the best model and hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Evaluate the best model
y_test_predictions = best_model.predict(X_test)
pred_test = np.argmax(y_test_predictions, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test_labels, pred_test, average='weighted')
recall = recall_score(y_test_labels, pred_test, average='weighted')
f1score = f1_score(y_test_labels, pred_test, average='weighted')

print(f"Precision = {round(precision, 4)}")
print(f"Recall = {round(recall, 4)}")
print(f"F1 Score = {round(f1score, 4)}")

# Save the best model
best_model.save('best_cnn_model.h5')
print("Best model saved.")

# Now calculate the confusion matrix
conf_matrix = confusion_matrix(y_test_labels, pred_test)
print(conf_matrix)