from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import pandas as pd
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

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

# Load your dataframe
# df = pd.read_csv('your_file.csv')

# Expand and preprocess data
expanded_df = split_and_expand_df(df)
expanded_df['ae_description'] = expanded_df['ae_description'].apply(preprocess_german_text)

# Tokenization and Sequencing
max_words = 10000  # Maximum number of words to consider
max_len = 150  # Maximum sequence length

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(expanded_df['ae_description'])

sequences = tokenizer.texts_to_sequences(expanded_df['ae_description'])
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Create a random embedding matrix
embedding_dim = 128  # Dimension of the embeddings
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = np.random.rand(vocab_size, embedding_dim)  # Random initialization

# MultiLabel Binarization of the llt_code column
one_hot_multiclass = MultiLabelBinarizer()
multi_labels = one_hot_multiclass.fit_transform(expanded_df['llt_code'])

# Compute class weights to handle imbalanced data
class_weights = compute_class_weight('balanced', classes=np.unique(expanded_df['llt_code']), y=expanded_df['llt_code'])
class_weights_dict = dict(enumerate(class_weights))

# Train-Test-Val split with stratified sampling
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(padded_sequences, multi_labels))

X_train, X_test = padded_sequences[train_idx], padded_sequences[test_idx]
y_train, y_test = multi_labels[train_idx], multi_labels[test_idx]

# Optionally, you can split the training set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Prepare embedding matrix
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
embedding_matrix = np.zeros((vocab_size, embedding_dim))
num_classes = len(set(y_train))


# MLP Modell - Version 2 (Binäre Klassifikation)
model_mlp_2 = Sequential()
model_mlp_2.add(Embedding(max_vocab_size, embedding_dim, input_length=max_sequence_length))
model_mlp_2.add(Flatten())
for _ in range(6):
    model_mlp_2.add(Dense(512, activation='relu'))
    model_mlp_2.add(Dropout(0.5))
model_mlp_2.add(Dense(1, activation='sigmoid'))

model_mlp_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
model_mlp_2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, callbacks=[early_stopping])

plot_model(model_mlp_2, to_file='model_architecture.png', show_shapes=True)

# Evaluate the model
y_pred = np.argmax(model_mlp_2.predict(X_val), axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_pred)

# Classification report
print(classification_report(y_val, y_pred))

# Evaluate the model on test data
test_loss, test_accuracy = model_mlp_2.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Save the model
model_mlp_2.save('mlp_sigmoid.h5')