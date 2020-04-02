import pandas as pd
import glob
import re
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.preprocessing import sequence
from w2v import train_word2vec
import numpy as np

# ---------------------- Parameters start ---------------------
#

# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 4, 5)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 64
num_epochs = 10

# Prepossessing parameters
sequence_length = 400
max_words = 5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10

#
# ---------------------- Parameters end -----------------------

path = glob.glob('/Users/asi/connor_asi/spaff_data/spaff_grouped_transcripts/*.csv')
df_list = []
structured_utt_list = []
structured_labels = []
text_data = []

for i in path:
    df = pd.read_csv(i)
    df = df[df.spaff_category != 99]
    df = df[df.spaff_category != 10]
    df_list.append(df)
    utt_list = df.words_no_nan.tolist()
    text_data.append(utt_list)
    label_list = df.sentiment_category_int.tolist()
    structured_labels.append(label_list)

    new_utt_list = []
    for j in utt_list:
        new_utt_list.append([j])
    structured_utt_list.append(new_utt_list)

master_df = pd.concat(df_list)

###
# Tokenizing hierarchical strucutred text data
# Looks like:
                        #        |    corpus    |
                        #       /        |       \
                        #      c1       c2       c3
                        #    / | \     / | \    / | \
                        #   u1 u2 u3 u1 u2 u3  u1 u2 u3
                # where each node is a list (nested from bottom up)


tok_data = [y[0] for x in structured_utt_list for y in x]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tok_data)
word_index = tokenizer.word_index
vocabulary_inv = dict((v, k) for k, v in word_index.items())
vocabulary_inv[0] = "<PAD/>"
tok_sequences = []
text_sequences = []
for x in structured_utt_list:
    tok_tmp = []
    text_tmp = []
    for y in x:
        tok_tmp.append(tokenizer.texts_to_sequences(y)[0])
        text_tmp.append(y)
    tok_sequences.append(tok_tmp)
    text_sequences.append(text_tmp)

### 
# De-structuring (flattening) hierarchical text data to feed into embedding
# Looks like:
                        #    |        corpus        |
                        #    / | \     / | \    / | \
                        #   u1 u2 u3 u4 u5 u6  u7 u8 u9
                # not sure if this is the best approach...
                # loosing conversation level granularity,
                # but context may not exceed window function?

flat_tokens = [item for sublist in tok_sequences for item in sublist] # len 7910
flat_labels = [item for sublist in structured_labels for item in sublist] # len 7910

# # Set vocab size
flat_list = [item for sublist in text_sequences for item in sublist]
flat_list = [item for sublist in flat_list for item in sublist]
flat_words = ' '.join(flat_list)
distinct_words = set(m.group(0).lower() for m in re.finditer(r"\w+",flat_words))
len_distinct_words = len(distinct_words) + 1 # 8581

# pad sequences to 400 (convention set by OG model)
pad_flat_tokens = sequence.pad_sequences(flat_tokens, maxlen=sequence_length, padding="post", truncating="post")

# swt 80/20 train test split
x_train, y_train = pad_flat_tokens[:round(len(flat_tokens)*.8)], flat_labels[round(len(flat_labels)*.8):]
x_test, y_test = pad_flat_tokens[:round(len(flat_tokens)*.8)],flat_labels[round(len(flat_labels)*.8):]

# Prepare embedding layer weights and convert inputs for static model
x_train, x_test = np.array(x_train), np.array(x_test)
embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                    min_word_count=min_word_count, context=context)

input_shape = (sequence_length,)
model_input = Input(shape=input_shape)

z = Embedding(
    len_distinct_words,
    embedding_dim,
    input_length=sequence_length, 
    name="embedding")(model_input)

z = Dropout(dropout_prob[0])(z)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)
# Set Dense layer as output tensors to feed into DialogueGCN
model_output = Dense(hidden_dims, activation="relu")(z)

# Freeze last (classification) layer from OG model
# model_output = Dense(1, activation="sigmoid")(z)

# Define model
model = Model(model_input, model_output)

# Create output variable
output = model.layers[-1].get_weights()
print('output;', output)
print(len(output))
print(output[0].shape)
print(output[1].shape)
