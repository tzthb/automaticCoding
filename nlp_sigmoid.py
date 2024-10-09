from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers



# Hyperparameter für die Vorverarbeitung
max_vocab_size = 10000  # Maximalgröße des Vokabulars
max_sequence_length = 100  # Maximal zulässige Länge der Sequenzen
embedding_dim = 300  # Dimension des Embeddings

# Define the paths to your CSV files
#csv_file_path_1 = './data/data.csv'
csv_file_path_2 = './data/AEcodiert240430_UTF8.csv'
csv_file_path_3 = './data/meddra_zkls.csv'
csv_file_path_4 = './data/meddra_zkls2.csv'

# Read the CSV files into DataFrames
try:
    #df1 = pd.read_csv(csv_file_path_1, delimiter=';', encoding='utf-8')
    df2 = pd.read_csv(csv_file_path_2, delimiter=';', encoding='utf-8')
    df3 = pd.read_csv(csv_file_path_3, delimiter=';', encoding='utf-8')
    df4 = pd.read_csv(csv_file_path_4, delimiter=';', encoding='utf-8')

except FileNotFoundError as e:
    print(e)
    exit()

# Concatenate all the DataFrames
df = pd.concat([ df2, df3, df4], ignore_index=True)

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

# Expand and preprocess data
expanded_df = split_and_expand_df(df)
expanded_df['ae_description'] = expanded_df['ae_description'].apply(preprocess_german_text)

print('Dataframe loaded and splitted.')
# Tokenization and Sequencing


### Step 1: Text Tokenization
max_vocab_size = 10000  # Vocabulary size
max_sequence_length = 100  # Max sequence length

# Tokenizer for text (convert text to sequences of integers)
tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(expanded_df['ae_description'])

# Convert texts to sequences of integers
sequences = tokenizer.texts_to_sequences(expanded_df['ae_description'])

# Pad sequences so all inputs have the same length
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

### Step 2: Label Encoding (Integer Encoding)
# Use LabelEncoder to transform labels into integer-encoded format
label_encoder = MultiLabelBinarizer()
integer_encoded_labels = label_encoder.fit_transform(expanded_df['llt_code'])

### Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, integer_encoded_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print('Data splitted into train and test sets.')
# Prepare embedding matrix
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
num_classes = len(label_encoder.classes_)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
l1_l2_regularizer = regularizers.l1_l2(l1=0.01, l2=0.01)

# NLP Model (Dense Network to mirror the number of layers and neurons as the RNN and CNN models)
model_nlp = Sequential()

# Embedding layer (same as in the original RNN and CNN models)
model_nlp.add(Embedding(max_vocab_size, embedding_dim, input_length=max_sequence_length))
# First Dense layer with 512 neurons
model_nlp.add(Dense(512, activation='relu', kernel_regularizer=l1_l2_regularizer))
model_nlp.add(Dropout(0.1)) 
model_nlp.add(BatchNormalization()) 
# Second Dense layer with 512 neurons
model_nlp.add(Dense(512, activation='relu'))
model_nlp.add(Dropout(0.1))
model_nlp.add(BatchNormalization()) 
# Third Dense layer with 512 neurons
model_nlp.add(Dense(512, activation='relu'))
model_nlp.add(Dropout(0.1))
# Fourth Dense layer with 512 neurons
model_nlp.add(Dense(512, activation='tanh'))
model_nlp.add(Dropout(0.1))
model_nlp.add(BatchNormalization()) 
model_nlp.add(Dense(512, activation='tanh'))
model_nlp.add(Dropout(0.1))
model_nlp.add(Flatten())# Output Dense layer with softmax activation for multi-class classification
model_nlp.add(Dense(num_classes, activation='sigmoid'))

# Compile the model
model_nlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early Stopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Training the NLP model
model_nlp.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=256, callbacks=[early_stopping])

# Model Summary
print(model_nlp.summary())

y_test_predictions = model_nlp.predict(X_test)
pred_test=np.argmax(y_test_predictions, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Classification Metrics
accuracy = accuracy_score(y_test_labels, pred_test)
precision = precision_score(y_test_labels, pred_test, average='weighted')
recall = recall_score(y_test_labels, pred_test, average='weighted')
f1score = f1_score(y_test_labels, pred_test, average='weighted')

print(f"Accuracy = {round(accuracy, 4)}")
print(f"Precision = {round(precision, 4)}")
print(f"Recall = {round(recall, 4)}")
print(f"F1 Score = {round(f1score, 4)}")

# Model Evaluation on test data
test_loss, test_accuracy = model_nlp.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the CNN model
model_nlp.save('nlp_sigmoid.keras')
print("Model saved.")

# Convert y_test to class labels if it's in one-hot encoded format
# Convert model predictions to class labels

# Now calculate the confusion matrix
conf_matrix = confusion_matrix(y_test_labels, pred_test)
print(conf_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
 
plt.figure(figsize=(8,8))
sns.set(font_scale = 1.5)
 
ax = sns.heatmap(
    conf_matrix, # confusion matrix 2D array 
    annot=True, # show numbers in the cells
    fmt='d', # show numbers as integers
    cbar=False, # don't show the color bar
    cmap='flag', # customize color map
    vmax=175 # to get better color contrast
)
 
ax.set_xlabel("Predicted", labelpad=20)
ax.set_ylabel("Actual", labelpad=20)
plt.show()
#save heatmap
plt.savefig('heatmap_nlpsigmoid.png')



 

#Predict llt_code
input_text = "Erkältung mit Husten und Halsschmerzen"

def preprocess_german_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text, language='german')  # Tokenize
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Remove stopwords and stem
    return words

# Assuming `input_text` is the new text you want to predict
input_text_processed = preprocess_german_text(input_text)
input_sequence = tokenizer.texts_to_sequences([input_text_processed])  # Use tokenizer from training
padded_input_sequence = pad_sequences(input_sequence, maxlen=max_len)  # Pad sequences

from tensorflow.keras.models import load_model
model = load_model('gemischtesModel.keras')
print('loaded model')
# Make prediction
predictions = model.predict(padded_input_sequence)

# Get predicted class index (for multiclass classification)
predicted_class_index = np.argmax(predictions)

# Map back to category label if necessary (using your LabelEncoder)
predicted_category = label_encoder.inverse_transform([predicted_class_index])

print(f"Predicted category: {predicted_category[0]}")

from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_architecture.png', show_shapes=True)