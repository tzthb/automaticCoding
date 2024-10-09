import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models import Word2Vec
import numpy as np
np.object = np.object_
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
# Model Definition and Training
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D, LSTM
from tensorflow.keras.optimizers import Adam
import tensorflowjs as tfjs
import nltk

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
    
# Combine both DataFrames
df = pd.concat([df1, df2], ignore_index=True)

# Data Cleaning
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('german'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(words):
    return [word for word in words if word.lower() not in stop_words]

def to_lowercase(words):
    return [word.lower() for word in words]

df['ae_description'] = df['ae_description'].astype(str)
df['cleaned_text'] = df['ae_description'].apply(clean_text)
df['tokenized_text'] = df['cleaned_text'].apply(tokenize_text)
df['filtered_text'] = df['tokenized_text'].apply(remove_stopwords)
df['lowercase_text'] = df['filtered_text'].apply(to_lowercase)

# EDA
word_freq = Counter([word for sublist in df['lowercase_text'] for word in sublist])
wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Text Representation
sentences = df['lowercase_text'].tolist()
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

def text_to_vector(words):
    vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

df['text_vector'] = df['lowercase_text'].apply(text_to_vector)
text_vectors = np.vstack(df['text_vector'].values)

# Tokenization and Sequencing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_text'])

sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
word_index = tokenizer.word_index
print(f'Found {len(word_index)} unique tokens.')

max_sequence_length = 100
data = pad_sequences(sequences, maxlen=max_sequence_length)

# Encoding Labels
mlb = MultiLabelBinarizer()

# Ensure labels are in string format and split correctly
df['llt_code'] = df['llt_code'].astype(str)
labels = df['llt_code'].apply(lambda x: x.split(','))

# Expand the DataFrame by splitting rows with multiple llt_codes
def split_and_expand_df(df):
    def split_row(row):
        llt_code_list = row['llt_code'].split(',')
        return pd.DataFrame({'llt_code': llt_code_list, 'ae_description': row['ae_description']})
    
    expanded_rows = [split_row(row) for _, row in df.iterrows()]
    expanded_df = pd.concat(expanded_rows, ignore_index=True)
    return expanded_df

expanded_df = split_and_expand_df(df)

# Re-process the expanded DataFrame
expanded_df['cleaned_text'] = expanded_df['ae_description'].apply(clean_text)
expanded_df['tokenized_text'] = expanded_df['cleaned_text'].apply(tokenize_text)
expanded_df['filtered_text'] = expanded_df['tokenized_text'].apply(remove_stopwords)
expanded_df['lowercase_text'] = expanded_df['filtered_text'].apply(to_lowercase)
expanded_df['text_vector'] = expanded_df['lowercase_text'].apply(text_to_vector)

# Get new text_vectors and labels
text_vectors = np.vstack(expanded_df['text_vector'].values)
binary_labels = mlb.fit_transform(expanded_df['llt_code'].apply(lambda x: [x]))

# Split the data into training, test, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(text_vectors, binary_labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define model parameters
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
num_heads = 4
ff_dim = 128
num_classes = binary_labels.shape[1]
max_len = max_sequence_length

# Model architecture with multi-head attention and causal masking
inputs = Input(shape=(max_len,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(inputs)
attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(embedding_layer, embedding_layer)
attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
attention_output = Dropout(0.1)(attention_output)

ff_output = Dense(ff_dim, activation='relu')(attention_output)
ff_output = LayerNormalization(epsilon=1e-6)(ff_output)
ff_output = Dropout(0.1)(ff_output)
flat_output = GlobalAveragePooling1D()(ff_output)
outputs = Dense(num_classes, activation='sigmoid')(flat_output)

# Compile the model
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=256)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Save the model
model.save('wordvec_model.h5')

# Convert the model to TensorFlow.js format
tfjs.converters.save_keras_model(model, 'wordvec_model')
