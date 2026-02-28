import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------
# 1. Load Dataset
# ----------------------

vocab_size = 10000
max_length = 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

# ----------------------
# 2. Pad Sequences
# ----------------------

x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

# ----------------------
# 3. Build Model
# ----------------------

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 64),
    keras.layers.LSTM(64),
    keras.layers.Dense(1, activation='sigmoid')
])

# ----------------------
# 4. Compile
# ----------------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ----------------------
# 5. Train
# ----------------------

model.fit(x_train, y_train, epochs=5, batch_size=64)

# ----------------------
# 6. Evaluate
# ----------------------

loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)

# ----------------------
# Function to Predict Custom Review
# ----------------------

# Get word index from dataset
word_index = keras.datasets.imdb.get_word_index()

def encode_review(text):
    tokens = text.lower().split()
    encoded = []
    for word in tokens:
        if word in word_index and word_index[word] < 10000:
            encoded.append(word_index[word] + 3)
        else:
            encoded.append(2)  # 2 = unknown word
    return encoded

def predict_review(text):
    encoded = encode_review(text)
    padded = pad_sequences([encoded], maxlen=200)
    prediction = model.predict(padded)[0][0]
    
    print("\nReview:", text)
    print("Prediction Score:", prediction)
    
    if prediction > 0.5:
        print("Sentiment: Positive")
    else:
        print("Sentiment: Negative")

# ----------------------
# User Input
# ----------------------

print("\nType a movie review to analyze sentiment.")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("Enter review: ")
    
    if user_input.lower() == "quit":
        print("Exiting program.")
        break
    
    predict_review(user_input)