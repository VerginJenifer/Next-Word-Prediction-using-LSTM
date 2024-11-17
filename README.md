# Next-Word-Prediction-using-LSTM

## AIM

To develop an LSTM-based model for predicting the next word in a text corpus.

## Problem Statement and Dataset


## DESIGN STEPS

### STEP 1:
Use fit_vectorizer to initialize and fit a TextVectorization layer on the corpus for word-to-integer tokenization.
### STEP 2:
Generate n-grams for each sentence using n_gram_seqs, creating sequential input data.
### STEP 3:
Pad these sequences to a uniform length with pad_seqs, enabling consistent input shapes for training.
### STEP 4:
Split each sequence into features and labels, where features contain all words except the last, and labels are the last word.
### STEP 5:
One-hot encode the labels with a vocabulary size from total_words for categorical prediction.
### STEP 6:
Construct a TensorFlow dataset with these features and labels, batching them for efficient processing.
### STEP 7:
Build the model with an Embedding layer, Bidirectional LSTM for sequence processing, and Dense layer with softmax for word prediction.
### STEP 8:
Compile and train the model using categorical cross-entropy loss and the Adam optimizer.
## PROGRAM
### Name: D Vergin Jenifer
### Register Number: 21223240174

### 1.fit_vectorizer function
def fit_vectorizer(corpus):
    """
    Instantiates the vectorizer class on the corpus
    
    Args:
        corpus (list): List with the sentences.
    
    Returns:
        (tf.keras.layers.TextVectorization): an instance of the TextVectorization class containing the word-index dictionary, adapted to the corpus sentences.
    """    

    tf.keras.utils.set_random_seed(65) # Do not change this line or you may have different expected outputs throughout the assignment

    ### START CODE HERE ###

    # Define the object
    vectorizer = tf.keras.layers.TextVectorization()

    # Adapt it to the corpus
    vectorizer.adapt(corpus)

    ### END CODE HERE ###
    
    return vectorizer
### 2. n_grams_seqs function
def n_gram_seqs(corpus, vectorizer):
    """
    Generates a list of n-gram sequences

    Args:
        corpus (list of string): lines of texts to generate n-grams for
        vectorizer (tf.keras.layers.TextVectorization): an instance of the TextVectorization class adapted in the corpus

    Returns:
        (list of tf.int64 tensors): the n-gram sequences for each line in the corpus
    """
    input_sequences = []

    ### START CODE HERE ###
    for sentence in corpus:
        # Vectorize the sentence to get the token indices
        vectorized_sentence =

        # Generate n-grams for the vectorized sentence
        for i in range( ):  # Start from 2 to avoid the first token
            n_gram =
            input_sequences.append( )

    ### END CODE HERE ###

    return input_sequences

### 3. pad_seqs function
def pad_seqs(input_sequences, max_sequence_len):
    """
    Pads tokenized sequences to the same length
    
    Args:
        input_sequences (list of int): tokenized sequences to pad
        maxlen (int): maximum length of the token sequences
    
    Returns:
        (np.array of int32): tokenized sequences padded to the same length
    """
    
    # Use tf.keras.preprocessing.sequence.pad_sequences to pad sequences to the desired length
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        input_sequences, 
        maxlen=max_sequence_len,       # Set the max length for padding
        padding='pre',                 # Add padding at the beginning of sequences
        truncating='post',             # Truncate sequences that exceed maxlen (if needed)
        value=0                        # Set the padding value to 0
    )
    
    return padded_sequences

### 4. features_and_labels_dataset function
def features_and_labels_dataset(input_sequences, total_words):
    """
    Generates features and labels from n-grams and returns a tensorflow dataset
    
    Args:
        input_sequences (list of int): sequences to split features and labels from
        total_words (int): vocabulary size
    
    Returns:
        (tf.data.Dataset): Dataset with elements in the form (sentence, label)
    """
    
    # Initialize features and labels lists
    features = []
    labels = []
    
    # Loop over all sequences in the input_sequences
    for sequence in input_sequences:
        # The last word is the label (the word we want to predict)
        label = sequence[-1]
        # The rest are the features (sequence without the last word)
        feature = sequence[:-1]
        
        features.append(feature)
        labels.append(label)
    
    # One-hot encode the labels
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)
    
    # Create a tensorflow dataset from features and one-hot encoded labels
    dataset = tf.data.Dataset.from_tensor_slices((features, one_hot_labels))
    
    # Batch the dataset with a batch size of 16 (can be adjusted if necessary)
    batched_dataset = dataset.batch(16)
    
    return batched_dataset
    
### 5.create_model function
def create_model(total_words, max_sequence_len):
    """
    Creates a text generator model capable of achieving at least 80% accuracy.
    
    Args:
        total_words (int): size of the vocabulary for the Embedding layer input
        max_sequence_len (int): length of the input sequences
    
    Returns:
       (tf.keras Model): the text generator model
    """
    model = tf.keras.Sequential()

    # Input layer: shape is (max_sequence_len - 1) since we remove the last word
    model.add(tf.keras.layers.Input(shape=(max_sequence_len - 1,)))  # Input without the last word

    # Embedding layer: 100 dimensions for the word embeddings
    model.add(tf.keras.layers.Embedding(input_dim=total_words, output_dim=100))

    # Bidirectional LSTM layer: 128 units for learning sequence patterns
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))

    # Dense layer: Output layer with the number of words in the vocabulary
    model.add(tf.keras.layers.Dense(total_words, activation='softmax'))  # Softmax activation for classification

    # Compile the model
    model.compile(loss='categorical_crossentropy',  # Loss for multi-class classification
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Optimizer with learning rate 0.001
                  metrics=['accuracy'])  # Metric to track during training
    
    return model

## OUTPUT
### 1. fit_vectorizer output
![image](https://github.com/user-attachments/assets/3831ad8e-50ab-4428-86a7-9b0e26aec3a6)
![image](https://github.com/user-attachments/assets/831efd29-218e-47e3-bae6-e6e7771913f4)

### 2. n_grams_seqs output
![image](https://github.com/user-attachments/assets/991984a3-3b31-4739-a73c-c5f8cd9a9d8d)

### 3. pad_seqs output
![image](https://github.com/user-attachments/assets/04976372-39a0-41bd-9ad5-2cadfe5edb71)

### 4. features_and_labels_dataset output
![image](https://github.com/user-attachments/assets/b07a0049-92f9-4a5f-91ad-ef07a85767fe)

### 5. Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/19d98db8-fe10-49de-99ae-5b8eaa1ecf5c)


### 6. Sample Text Prediction
![image](https://github.com/user-attachments/assets/b320dc5d-93cf-4beb-9425-bd51784c2bdf)


## RESULT
Thus, Next Word Prediction was executed Successfully.
