
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import re
import IPython.display
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM, Flatten
from keras.utils import to_categorical
from keras.utils import plot_model
from nltk.tokenize import word_tokenize
import spacy
nlp = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')


# In[7]:

headers = ['song_id', 'translated_names']
songs = pd.read_csv('../New_Data/tr_songs.csv', usecols = headers, nrows=1000)
duplicated_idx = songs.duplicated(subset='song_id', keep='first')
songs = songs[~duplicated_idx]
songs['song_name'] = songs['translated_names'].map(str).apply(lambda x : ''.join([i for i in re.findall(r'[a-zA-Z_\s]', x)]))
songs['song_name'] = songs['song_name'].map(str).apply(lambda x : re.sub(r'\s+',' ',x))

headers_joined = ['song_id', 'artist_name', 'composer', 'lyricist', 'genre_id']
joined = pd.read_csv('../New_Data/joined.csv', usecols = headers_joined)

data = songs.merge(joined, on='song_id', how='left')


# In[8]:

data = data.fillna(-2)
#print data.head


# In[9]:

def onehot(column):
    label_encoder = LabelEncoder()
    integer_encoding = label_encoder.fit_transform([str(x) for x in column])

    #onehot_v = to_categorical(integer_encoding)
    #print onehot_encoding.shape[1] #45339
    return integer_encoding


# In[10]:

artist_mapper = dict()
artists_unique = data['artist_name'].unique()
composers_unique = data['composer'].unique()
lyricists_unique = data['lyricist'].unique()
genres_unique = data['genre_id'].unique()

artists_oh = onehot(artists_unique)
composers_oh = onehot(composers_unique)
lyricists_oh = onehot(lyricists_unique)
genres_oh = onehot(genres_unique)

artists_mapper = dict(zip(artists_unique, artists_oh))
composers_mapper = dict(zip(composers_unique, composers_oh))
lyricists_mapper = dict(zip(lyricists_unique, lyricists_oh))
genres_mapper = dict(zip(genres_unique, genres_oh))


# In[ ]:

import pickle

with open('artist_mapper.pkl', 'wb') as fw:
    pickle.dump(artists_mapper, fw)

with open('composer_mapper.pkl', 'wb') as fw:
    pickle.dump(composers_mapper, fw)

with open('lyricist_mapper.pkl', 'wb') as fw:
    pickle.dump(lyricists_mapper, fw)

with open('genre_mapper.pkl', 'wb') as fw:
    pickle.dump(genres_mapper, fw)


# In[11]:

seq_length = 25
cont = True


# In[12]:

def generate_songs_tensor(song_names, nlp, steps):
    #assert not isinstance(song_names, basestring)
    nb_samples = len(song_names)
    word_vec_dim = nlp(song_names[0].decode('utf8'))[0].vector.shape[0]
    song_tensor = np.zeros((nb_samples, steps, word_vec_dim))
    for i in xrange(len(song_names)):
        tokens = nlp(song_names[i].decode('utf8'))
        for j in xrange(len(tokens)):
            if j<steps:
                song_tensor[i,j,:] = tokens[j].vector

    return song_tensor


# In[13]:

def output_generator(data):
    num_rows = data.shape[0]
    #print X[10]
    Y0 = np.empty((data.shape[0], ))
    Y1 = np.empty((data.shape[0], ))
    Y2 = np.empty((data.shape[0], ))
    Y3 = np.empty((data.shape[0], ))

    count = 0
    for row_num, row in data.iterrows():
        Y0[count] = artists_mapper[row['artist_name']]
        Y1[count] = composers_mapper[row['composer']]
        Y2[count] = lyricists_mapper[row['lyricist']]
        Y3[count] = genres_mapper[row['genre_id']]
        count += 1

    return [Y0, Y1, Y2, Y3]


# In[ ]:

Ys = output_generator(data)


# In[ ]:

X = generate_songs_tensor(data['song_name'], nlp, seq_length)


# In[ ]:

input_dim = 300
hidden_units_1 = 128
hidden_units_mlp = 100
dropout_rate = 0.4
num_epochs = 30
batch_size = 256

if cont:
    input_features = Input(shape = (seq_length, input_dim))
    hidden = Dropout(dropout_rate)(LSTM(output_dim=hidden_units_1, return_sequences=False)(input_features))
    flatten = Flatten()(hidden)
    hidden_2 = Dense(hidden_units_mlp, activation='tanh')(flatten)
    output_0 = Dense(len(artists_mapper), activation='softmax')(hidden_2)
    output_1 = Dense(len(composers_mapper), activation='softmax')(hidden_2)
    output_2 = Dense(len(lyricists_mapper), activation='softmax')(hidden_2)
    output_3 = Dense(len(genres_mapper), activation='softmax')(hidden_2)
    model = keras.models.Model(inputs = [input_features],
                               outputs = [output_0, output_1, output_2, output_3])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
else:
    model = keras.models.load_model('../Models/songs_embeddings_100.h5')


# In[ ]:

model.fit(X, Ys, batch_size = batch_size, epochs = 30, verbose=2)
print(model.evaluate(X, Ys))
model.save('../Models/songs_embeddings_100.h5')

