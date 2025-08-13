
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, GRU, Dense, Embedding,
                                     Bidirectional, RepeatVector, Concatenate,
                                     Dot, Lambda)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K

# =================================================================
# Helper Functions
# =================================================================

def softmax_over_time(x):
    """
    Custom softmax function that applies softmax over the time dimension.
    Required for the attention mechanism.
    """
    assert K.ndim(x) > 2
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e / s


def masked_accuracy(y_true, y_pred):
    """
    Custom accuracy metric that ignores padding (0) in the target sequences.
    """
    # Squeeze y_true if it's one-hot encoded, otherwise it's already integer-based
    targ = K.cast(y_true, 'int64')
    if K.ndim(y_true) > K.ndim(y_pred):
        targ = K.squeeze(targ, axis=-1)
        
    pred = K.cast(K.argmax(y_pred, axis=-1), 'int64')
    correct = K.cast(K.equal(targ, pred), K.floatx())

    # 0 is padding, so we don't count it
    mask = K.cast(K.greater(targ, 0), K.floatx())
    n_correct = K.sum(mask * correct)
    n_total = K.sum(mask)
    return n_correct / (n_total + K.epsilon())


# =================================================================
# Data Loading and Preprocessing
# =================================================================

def load_data(data_path, num_samples):
    """Loads and parses the translation data file."""
    input_texts = []
    target_texts = []
    target_texts_inputs = []

    t = 0
    for line in open(data_path, encoding='utf-8'):
        t += 1
        if t > num_samples:
            break
        if '\t' not in line:
            continue
        
        input_text, translation, *_ = line.rstrip().split('\t')
        
        target_text = translation + ' <eos>'
        target_text_input = '<sos> ' + translation
        
        input_texts.append(input_text)
        target_texts.append(target_text)
        target_texts_inputs.append(target_text_input)
        
    print(f"Loaded {len(input_texts)} samples.")
    return input_texts, target_texts, target_texts_inputs


def load_glove_embeddings(glove_path):
    """Loads GloVe word vectors into a dictionary."""
    print('Loading word vectors...')
    word2vec = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ', 1)
            word = values[0]
            if len(values) > 1:
                vec = np.asarray(values[1].split(), dtype='float32')
                word2vec[word] = vec
    print(f'Found {len(word2vec)} word vectors.')
    return word2vec


def create_embedding_matrix(word2idx, embedding_dim, word2vec, max_num_words):
    """Creates an embedding matrix from pre-trained GloVe vectors."""
    print('Filling pre-trained embeddings...')
    num_words = min(max_num_words, len(word2idx) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word2idx.items():
        if i < max_num_words:
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix, num_words


# =================================================================
# Model Building
# =================================================================

def build_model(config, embedding_matrix, num_words_input, num_words_output):
    """Builds and compiles the attention-based NMT model."""
    
    # Unpack config
    max_len_input = config['max_len_input']
    max_len_target = config['max_len_target']
    embedding_dim = config['embedding_dim']
    latent_dim = config['latent_dim']
    latent_dim_decoder = config['latent_dim_decoder']

    # --- ENCODER ---
    encoder_inputs_placeholder = Input(shape=(max_len_input,), name='encoder_input')
    
    embedding_layer = Embedding(
        num_words_input,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=max_len_input,
        trainable=False,
        name='encoder_embedding'
    )
    x = embedding_layer(encoder_inputs_placeholder)
    
    encoder = Bidirectional(LSTM(latent_dim, return_sequences=True), name='encoder_lstm')
    encoder_outputs = encoder(x)

    # --- DECODER with ATTENTION ---
    decoder_inputs_placeholder = Input(shape=(max_len_target,), name='decoder_input')
    
    decoder_embedding = Embedding(num_words_output, embedding_dim, name='decoder_embedding')
    decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

    # Attention layers
    attn_repeat_layer = RepeatVector(max_len_input)
    attn_concat_layer = Concatenate(axis=-1)
    attn_dense1 = Dense(10, activation='tanh')
    attn_dense2 = Dense(1, activation=softmax_over_time)
    attn_dot = Dot(axes=1)

    def one_step_attention(h, st_1):
        st_1 = attn_repeat_layer(st_1)
        x = attn_concat_layer([h, st_1])
        x = attn_dense1(x)
        alphas = attn_dense2(x)
        context = attn_dot([alphas, h])
        return context

    # Decoder LSTM and Dense layers
    decoder_lstm = LSTM(latent_dim_decoder, return_state=True)
    decoder_dense = Dense(num_words_output, activation='softmax')
    
    initial_s = Input(shape=(latent_dim_decoder,), name='s0')
    initial_c = Input(shape=(latent_dim_decoder,), name='c0')
    context_last_word_concat_layer = Concatenate(axis=2)
    
    s, c = initial_s, initial_c
    outputs = []
    
    for t in range(max_len_target):
        context = one_step_attention(encoder_outputs, s)
        selector = Lambda(lambda x: x[:, t:t + 1])
        xt = selector(decoder_inputs_x)
        
        decoder_lstm_input = context_last_word_concat_layer([context, xt])
        o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])
        
        decoder_outputs = decoder_dense(o)
        outputs.append(decoder_outputs)

    def stack_and_transpose(x):
        x = K.stack(x)
        return K.permute_dimensions(x, pattern=(1, 0, 2))

    stacker = Lambda(stack_and_transpose)
    outputs = stacker(outputs)

    model = Model(
        inputs=[encoder_inputs_placeholder, decoder_inputs_placeholder, initial_s, initial_c],
        outputs=outputs
    )

    # OPTIMIZATION: Using sparse cross-entropy to avoid one-hot encoding targets.
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[masked_accuracy]
    )
    
    return model


# =================================================================
# Inference Setup
# =================================================================

def build_inference_models(model, config):
    """Builds separate encoder and decoder models for inference."""
    
    max_len_input = config['max_len_input']
    latent_dim = config['latent_dim']
    latent_dim_decoder = config['latent_dim_decoder']

    # --- Encoder Model ---
    encoder_inputs_placeholder = model.get_layer('encoder_input').input
    encoder_outputs = model.get_layer('encoder_lstm').output[0] # Bidirectional output
    encoder_model = Model(encoder_inputs_placeholder, encoder_outputs)

    # --- Decoder Model ---
    encoder_outputs_as_input = Input(shape=(max_len_input, 2 * latent_dim,))
    decoder_inputs_single = Input(shape=(1,))
    
    # Get layers from the trained model
    decoder_embedding = model.get_layer('decoder_embedding')
    decoder_lstm = model.get_layer('lstm_1') # Adjust index if model changes
    decoder_dense = model.get_layer('dense_2')
    
    # Attention layers
    attn_repeat_layer = model.get_layer('repeat_vector')
    attn_concat_layer = model.get_layer('concatenate')
    attn_dense1 = model.get_layer('dense')
    attn_dense2 = model.get_layer('dense_1')
    attn_dot = model.get_layer('dot')
    context_last_word_concat_layer = model.get_layer('concatenate_1')

    def one_step_attention(h, st_1):
        st_1 = attn_repeat_layer(st_1)
        x = attn_concat_layer([h, st_1])
        x = attn_dense1(x)
        alphas = attn_dense2(x)
        context = attn_dot([alphas, h])
        return context
        
    initial_s = Input(shape=(latent_dim_decoder,), name='s0_inf')
    initial_c = Input(shape=(latent_dim_decoder,), name='c0_inf')

    context = one_step_attention(encoder_outputs_as_input, initial_s)
    
    decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
    decoder_lstm_input = context_last_word_concat_layer([context, decoder_inputs_single_x])
    
    o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
    decoder_outputs = decoder_dense(o)
    
    decoder_model = Model(
        inputs=[decoder_inputs_single, encoder_outputs_as_input, initial_s, initial_c],
        outputs=[decoder_outputs, s, c]
    )

    return encoder_model, decoder_model


def decode_sequence(input_seq, encoder_model, decoder_model, word2idx_outputs, idx2word_trans, config):
    """Decodes a sequence of text using the inference models."""
    
    enc_out = encoder_model.predict(input_seq)
    
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    
    eos = word2idx_outputs['<eos>']
    
    s = np.zeros((1, config['latent_dim_decoder']))
    c = np.zeros((1, config['latent_dim_decoder']))
    
    output_sentence = []
    for _ in range(config['max_len_target']):
        o, s, c = decoder_model.predict([target_seq, enc_out, s, c])
        
        idx = np.argmax(o.flatten())
        
        if eos == idx:
            break
        
        word = ''
        if idx > 0:
            word = idx2word_trans[idx]
            output_sentence.append(word)
            
        target_seq[0, 0] = idx
        
    return ' '.join(output_sentence)


# =================================================================
# Main Execution
# =================================================================

def main():
    # --- Configuration ---
    config = {
        'batch_size': 64,
        'epochs': 30,
        'latent_dim': 256,
        'latent_dim_decoder': 256,
        'num_samples': 20000,
        'max_num_words': 20000,
        'embedding_dim': 100,
        'data_path': 'spa.txt', # Download from http://www.manythings.org/anki/
        'glove_path': 'glove.6B.100d.txt' # Download from https://nlp.stanford.edu/projects/glove/
    }
    
    # --- Data Preparation ---
    input_texts, target_texts, target_texts_inputs = load_data(config['data_path'], config['num_samples'])
    
    # Tokenize inputs
    tokenizer_inputs = Tokenizer(num_words=config['max_num_words'])
    tokenizer_inputs.fit_on_texts(input_texts)
    input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
    word2idx_inputs = tokenizer_inputs.word_index
    print(f'Found {len(word2idx_inputs)} unique input tokens.')
    max_len_input = max(len(s) for s in input_sequences)
    config['max_len_input'] = max_len_input

    # Tokenize outputs
    tokenizer_outputs = Tokenizer(num_words=config['max_num_words'], filters='')
    tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)
    target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
    target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)
    word2idx_outputs = tokenizer_outputs.word_index
    print(f'Found {len(word2idx_outputs)} unique output tokens.')
    num_words_output = len(word2idx_outputs) + 1
    max_len_target = max(len(s) for s in target_sequences)
    config['max_len_target'] = max_len_target
    
    # Pad sequences
    encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
    decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
    decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')
    
    # --- Embedding ---
    word2vec = load_glove_embeddings(config['glove_path'])
    embedding_matrix, num_words_input = create_embedding_matrix(
        word2idx_inputs, config['embedding_dim'], word2vec, config['max_num_words']
    )
    
    # --- Model Training ---
    model = build_model(config, embedding_matrix, num_words_input, num_words_output)
    model.summary()
    
    z = np.zeros((len(encoder_inputs), config['latent_dim_decoder']))
    
    r = model.fit(
        [encoder_inputs, decoder_inputs, z, z],
        decoder_targets, # Using integer targets directly
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_split=0.2
    )
    
    # --- Plotting History ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(r.history['masked_accuracy'], label='accuracy')
    plt.plot(r.history['val_masked_accuracy'], label='val_accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    # --- Inference ---
    encoder_model, decoder_model = build_inference_models(model, config)
    idx2word_trans = {v: k for k, v in word2idx_outputs.items()}
    
    while True:
        i = np.random.choice(len(input_texts))
        input_seq = encoder_inputs[i:i + 1]
        translation = decode_sequence(
            input_seq, encoder_model, decoder_model, 
            word2idx_outputs, idx2word_trans, config
        )
        print('-' * 50)
        print('Input:', input_texts[i])
        print('Predicted:', translation)
        print('Actual:', target_texts[i].replace('<eos>', '').strip())
        
        ans = input("Continue? [Y/n]: ")
        if ans and ans.lower().startswith('n'):
            break

if __name__ == "__main__":
    main()