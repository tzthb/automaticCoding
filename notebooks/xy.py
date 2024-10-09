import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping

# Define the paths to your CSV files
csv_file_path_1 = 'data.csv'   # Replace with your actual first file path
csv_file_path_2 = 'AEcodiert240430_UTF8.csv'  # Replace with your actual second file path

# Read the first CSV file into a DataFrame
try:
    df1 = pd.read_csv(csv_file_path_1, delimiter=';', encoding='utf-8')
except FileNotFoundError:
    print(f"Error: CSV file '{csv_file_path_1}' not found.")
    exit()

# Read the second CSV file into a DataFrame
try:
    df2 = pd.read_csv(csv_file_path_2, delimiter=';', encoding='utf-8')
except FileNotFoundError:
    print(f"Error: CSV file '{csv_file_path_2}' not found.")
    exit()
df = pd.concat([df1, df2], ignore_index=True)
df = pd.DataFrame(df)

print("First few rows of the concatenated DataFrame:")
print(df.head(5))
#add df2 to df 
# Combine both DataFrames
def split_and_expand_df(df):
  """Splits rows with multiple llt_codes into separate rows, preserving descriptions.

  Args:
    df: The input DataFrame.

  Returns:
    The expanded DataFrame.
  """

  def split_row(row):
    llt_code_list = str(row['llt_code']).split(',')
    return pd.DataFrame({'llt_code': llt_code_list, 'ae_description': row['ae_description']})

  # Apply split_row to each row using apply
  df = df.apply(split_row, axis=1)

  # Concatenate resulting DataFrames
  df = pd.concat(df.tolist(), ignore_index=True)

  return df

splitData = pd.DataFrame(df)
expanded_df = split_and_expand_df(splitData.copy())

print(expanded_df)

# Tokenization and Sequencing
max_words = 10000  # Adjust based on your vocabulary size
max_len = 150  # Adjust based on your text length

import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re

nltk_data_path = '/Users/tizander/Masterthesis/dl-code/myenv/lib/python3.12/site-packages/nltk'
os.environ["NLTK_DATA"] = nltk_data_path

# Set up stopwords and stemmer
stop_words = set(stopwords.words('german'))
stemmer = SnowballStemmer('german')

def preprocess_german_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    words = word_tokenize(text, language='german')
    # Remove stopwords and stem
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Preprocess 'ae_description' column
if 'ae_description' in expanded_df.columns:
    expanded_df['ae_description'] = expanded_df['ae_description'].apply(preprocess_german_text)
else:
    print("Error: 'ae_description' column not found in the DataFrame.")
    exit()

print("Text preprocessing completed successfully.")
print(expanded_df[['llt_code', 'ae_description']].head())

feature = expanded_df.llt_code
description = expanded_df.ae_description
print('Feature: ')
print(feature )
print('Description: ')
print(description )

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(description)

sequences = tokenizer.texts_to_sequences(description)
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Create multiclass one-hot encoder
one_hot_multiclass = MultiLabelBinarizer()

# One-hot encode multiclass feature
multi_labels = one_hot_multiclass.fit_transform(feature)

print(one_hot_multiclass.classes_)

unique_labels = one_hot_multiclass.classes_
num_unique_labels = len(unique_labels)

print("Unique labels:", unique_labels)
print("Number of unique labels:", num_unique_labels)

print(f"Length of padded_sequences: {len(padded_sequences)}")
print(f"Length of multi_labels: {len(multi_labels)}")
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, multi_labels, test_size=0.2, random_state=42)

print("Data tokenization and splitting completed successfully.")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128))
#Add multiHeadAttentionlayer
#model.add(MultiHeadAttention(num_heads=8, key_dim=128))
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=True))
model.add(LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(units=len(unique_labels), activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Save the model
model.save('lstm_multilabel_model.h5')

# Print model summary
model.summary()
