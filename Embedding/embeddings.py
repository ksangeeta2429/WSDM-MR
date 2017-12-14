
# coding: utf-8

# # Generate Embeddings

# This script contains code snippets to outline the process of generating embeddings for various fields in the songs.csv - songs, artist, composer, lyricist etc. We generate embeddings using the following method: Taking the one of the columns as input, we try to predict the output of the other 3 columns. There are two ways we can take the individual rows of the input column, as input - (1) char-rnn (2) A one hot encoding with each unique input element will be considered different from each other. Advantage of (1) is that it will capture textual level similarity between the names whereas (2) will be faster to train and will avoid capturing misleading features

# In[1]:


from random import shuffle

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding
import pandas as pd
import numpy as np
import scipy.sparse
import tensorflow as tf
import pickle


# ## Loading Data

# In[105]:


cont = True
data = pd.read_csv('new_data/joined.csv').fillna('')
print("Data Loaded")


# Lets start by looking at number of distinct characters and number of distinct units in each columns. This will (hopefully) help in deciding which of the two approaches to choose from

# In[106]:


def get_unique_chars(data, column):
    char_set = set([c for (i, row) in data.iterrows() for c in str(row[column])])
    return len(char_set)

# Some of the rows corresponding to a column have multiple values separated by '|'
# character. We need to split and separate these multiple values

def get_unique_entities(data, column):
    data[column] = data[column].apply(str)
    return data[column].unique()

def generate_mapper(data, column):
    unique_elements = get_unique_entities(data, column)
    mapper = dict()
    mapper[''] = 0
    for u in unique_elements:
        if u is not '':
            mapper[u] = len(mapper)
    return mapper


# In[110]:


artist_mapper = generate_mapper(data, 'artist_name')
composer_mapper = generate_mapper(data, 'composer')
lyricist_mapper = generate_mapper(data, 'lyricist')
song_mapper = generate_mapper(data, 'song_id')
genre_mapper = generate_mapper(data, 'genre_id')

with open('artist_mapper.pkl', 'wb') as fw:
    pickle.dump(artist_mapper, fw)

with open('composer_mapper.pkl', 'wb') as fw:
    pickle.dump(composer_mapper, fw)

with open('lyricist_mapper.pkl', 'wb') as fw:
    pickle.dump(lyricist_mapper, fw)

with open('song_mapper.pkl', 'wb') as fw:
    pickle.dump(song_mapper, fw)

with open('genre_mapper.pkl', 'wb') as fw:
    pickle.dump(genre_mapper, fw)

def input_generator(data):
    num_rows = data.shape[0]
    X = np.empty(data.shape[0])
    Y0 = np.empty((data.shape[0], ))
    Y1 = np.empty((data.shape[0], ))
    Y2 = np.empty((data.shape[0], ))
    Y3 = np.empty((data.shape[0], ))

    count = 0
    for row_num, row in data.iterrows():
        X[count] = song_mapper[row['song_id']]
        Y0[count] = artist_mapper[row['artist_name']]
        Y1[count] = composer_mapper[row['composer']]
        Y2[count] = lyricist_mapper[row['lyricist']]
        Y3[count] = genre_mapper[str(row['genre_id'])]
        count += 1

    return (X, [Y0, Y1, Y2, Y3]) 



batch_size = 128
num_hidden_units = 100
hidden_activation = 'relu'

if not cont:
    input_features = Input(shape = (1,))
    embedding = keras.layers.Flatten()(
        Embedding(output_dim = num_hidden_units, input_dim = len(song_mapper), input_length = 1)(input_features))
    embedding = keras.layers.Activation(hidden_activation)(embedding)
    output_0 = Dense(len(artist_mapper), activation='softmax')(embedding)
    output_1 = Dense(len(composer_mapper), activation='softmax')(embedding)
    output_2 = Dense(len(lyricist_mapper), activation='softmax')(embedding)
    output_3 = Dense(len(genre_mapper), activation='softmax')(embedding)

    model = keras.models.Model(inputs = [input_features],
                               outputs = [output_0, output_1, output_2, output_3])


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

else:
    model = keras.models.load_model('songs_embeddings_100.h5')

X, Ys = input_generator(data)
model.fit(X, Ys, batch_size = batch_size, epochs = 5, verbose=2)
print(model.evaluate(X, Ys))
model.save('songs_embeddings_100.h5')
