
# coding: utf-8

# In[1]:


import numpy as np
import math
import pandas as pd

from tqdm import tqdm
import keras
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, Flatten
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomUniform
import cPickle as pickle
import progressbar
import os


# In[2]:


def load_dataset():
    return pd.read_csv('../Data/train.csv').fillna('')


# In[3]:


def load_songs_members_dataset():
    songs_dataset = pd.read_csv('../../new_data/Data/songs.csv')
    members_dataset = pd.read_csv('../Data/members.csv')
    return songs_dataset, members_dataset

# In[4]:

# In[5]:


# Some of the rows corresponding to a column have multiple values separated by '|'
# character. We need to split and separate these multiple values

def create_mapper(values):
    mapper = dict()
    for v in values:
        mapper[v] = len(v)
    return mapper


# In[6]:


def load_song_mapper_picke():
    song_mapper = pickle.load(open('../New_Data/model_song_embeddings/song_mapper_py2.pkl', 'rb'))
    return song_mapper

def load_user_mapper_pickle():
    user_mapper = pickle.load(open('../New_Data/model_user_embeddings/msno_mapper_py2.pkl', 'rb'))
    return user_mapper


# In[7]:


train_dataset = load_dataset()
songs_dataset, members_dataset = load_songs_members_dataset()


# In[87]:


train_dataset = train_dataset.merge(members_dataset, on='msno', how='left')


# In[88]:


train_dataset = train_dataset.merge(songs_dataset, on='song_id', how='left')


# In[89]:


def genre_id_count(x):
    if x == 'no_genre_id':
        return 0
    else:
        return x.count('|') + 1

train_dataset['genre_ids'].fillna('no_genre_id',inplace=True)
train_dataset['genre_ids_count'] = train_dataset['genre_ids'].apply(genre_id_count).astype(np.int32)
train_dataset['genre_ids_count'] = train_dataset['genre_ids_count'] / max(train_dataset['genre_ids_count'])


# In[90]:


def artist_count(x):
    if x == 'no_artist':
        return 0
    else:
        return x.count('and') + x.count(',') + x.count('feat') + x.count('&')
    
train_dataset['artist_name'].fillna('no_artist',inplace=True)
train_dataset['artist_count'] = train_dataset['artist_name'].map(str).apply(artist_count).astype(np.int32)
train_dataset['artist_count'] = train_dataset['artist_count'] / max(train_dataset['artist_count'])


# In[91]:


def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
    return sum(map(x.count, ['|', '/', '\\', ';']))

train_dataset['lyricist'].fillna('no_lyricist',inplace=True)
train_dataset['lyricists_count'] = train_dataset['lyricist'].map(str).apply(lyricist_count).astype(np.int32)
train_dataset['lyricists_count'] = train_dataset['lyricists_count'] / max(train_dataset['lyricists_count'])


# In[92]:


def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1

train_dataset['composer'].fillna('no_composer',inplace=True)
train_dataset['composer_count'] = train_dataset['composer'].map(str).apply(composer_count).astype(np.int8)
train_dataset['composer_count'] = train_dataset['composer_count'] / max(train_dataset['composer_count'])


# In[93]:


# number of times a song has been played before
_dict_count_song_played_train = {k: v for k, v in train_dataset['song_id'].value_counts().iteritems()}
def count_song_played(x):
    try:
        return _dict_count_song_played_train[x]
    except KeyError:
            return 0    

train_dataset['count_song_played'] = train_dataset['song_id'].map(str).apply(count_song_played).astype(np.int64)
train_dataset['count_song_played'] = train_dataset['count_song_played'] / max(train_dataset['count_song_played'])


# In[94]:


# number of times an artist has been played
_dict_count_artist_played_train = {k: v for k, v in train_dataset['artist_name'].value_counts().iteritems()}
def count_artist_played(x):
    try:
        return _dict_count_artist_played_train[x]
    except KeyError:
            return 0

train_dataset['count_artist_played'] = train_dataset['artist_name'].map(str).apply(count_artist_played).astype(np.int64)
train_dataset['count_artist_played'] = train_dataset['count_artist_played'] / max(train_dataset['count_artist_played'])



source_tab_mapper = create_mapper(train_dataset.source_system_tab.unique())
s_scr_name_mapper = create_mapper(train_dataset.source_screen_name.unique())
s_type_mapper = create_mapper(train_dataset.source_type.unique())

msno_mapper = load_user_mapper_pickle()
song_id_mapper = load_song_mapper_picke()


# In[8]:


def get_model():
    input_song_ids_layer = Input(shape=(1,))
    input_msno_layer = Input(shape=(1, ))
    input_source_system_tab = Input(shape=(1, ))
    input_source_screen_name = Input(shape=(1, ))
    input_source_type = Input(shape=(1, ))
    
    input_genre_ids_count = Input(shape=(1, ))
    input_artist_count = Input(shape=(1, ))
    input_lyricist_count = Input(shape=(1, ))
    input_composer_count = Input(shape=(1, ))
    input_song_count_played = Input(shape=(1, ))
    input_artist_count_played = Input(shape=(1, ))


    user_embeddings = Embedding(input_dim = len(msno_mapper) + 1,
            output_dim = 64,
            embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
            embeddings_regularizer=keras.regularizers.l2(1e-4),
            input_length=1,
            trainable=True)(input_msno_layer)
    
    song_embeddings = Embedding(input_dim=len(song_id_mapper) + 1, output_dim=64,           
            embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5), 
            embeddings_regularizer=keras.regularizers.l2(1e-4), input_length=1, trainable=True)(input_song_ids_layer)
    
    flatten_user = Flatten()(user_embeddings)
    flatten_songs = Flatten()(song_embeddings)
    
    sys_tab_embeddings = Embedding(output_dim=10, input_dim= max(source_tab_mapper.values()) + 1,input_length=1, trainable=True, embeddings_regularizer=keras.regularizers.l2(1e-3), embeddings_initializer='glorot_uniform') (input_source_system_tab)
    screen_name_embeddings = Embedding(output_dim=10, input_dim= max(s_scr_name_mapper.values()) + 1,input_length=1, trainable=True, embeddings_regularizer=keras.regularizers.l2(1e-3),embeddings_initializer='glorot_uniform' ) (input_source_screen_name)
    source_type_embeddings = Embedding(output_dim=10, input_dim= max(s_type_mapper.values()) + 1,input_length=1, trainable=True, embeddings_regularizer=keras.regularizers.l2(1e-3), embeddings_initializer='glorot_uniform') (input_source_type)
    
    flatten_tabs = Flatten()(sys_tab_embeddings)
    flatten_screen = Flatten()(screen_name_embeddings)
    flatten_type = Flatten()(source_type_embeddings)

    combined_meta = keras.layers.concatenate([flatten_tabs, flatten_type, flatten_screen])
    
    combined_counts = keras.layers.concatenate([input_genre_ids_count, input_artist_count, input_lyricist_count, input_composer_count, input_song_count_played, input_artist_count_played])
    
    dense_counts = Dense(40, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu')(combined_counts)

    dense_meta = Dense(40, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu')(combined_meta)

    
    dot_user_songs = keras.layers.dot([flatten_user, flatten_songs], axes = 1)
    combined_input = keras.layers.concatenate([flatten_user, flatten_songs, dot_user_songs, dense_meta, dense_counts])
    
    intermediate_0 = Dense(128, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu')(combined_input)
    dropout_0 = Dropout(0.5)(intermediate_0)
    
    intermediate_1 = Dense(64, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu')(dropout_0)
    dropout_1 = Dropout(0.5)(intermediate_1)
    
    intermediate_2 = Dense(16, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu')(dropout_1)
    
    output_0 = Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid')(intermediate_2)

    dnn_model = keras.models.Model(inputs = [input_msno_layer, input_song_ids_layer, input_source_system_tab, input_source_screen_name, input_source_type, input_genre_ids_count, input_artist_count, input_lyricist_count, input_composer_count, input_song_count_played, input_artist_count_played],
                               outputs = [output_0])

    dnn_model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    return dnn_model


# In[9]:


dnn_model = get_model()


# In[10]:


print dnn_model.summary()


# In[11]:


from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(dnn_model).create(prog='dot', format='svg'))


# In[12]:


# In[13]:


def gen_train_data(data):
    num_rows = data.shape[0]
    X_msno = np.empty(num_rows)
    X_song_id = np.empty(num_rows)
    X_source_system_tab = np.empty(num_rows)
    X_source_screen_name = np.empty(num_rows)
    X_source_type = np.empty(num_rows)
    Y = np.empty(num_rows)
    batch = num_rows
    input_genre_ids_count = np.empty(batch)
    input_artist_count = np.empty(batch)
    input_lyricist_count = np.empty(batch)
    
    input_composer_count = np.empty(batch)
    input_song_count_played = np.empty(batch)
    input_artist_count_played = np.empty(batch)
    
    count = 0
    bar = progressbar.ProgressBar()
    for _, row in bar(data.iterrows()):
        X_msno[count,] = msno_mapper[row['msno']]
        X_song_id[count, ] = song_id_mapper[row['song_id']]
        X_source_system_tab[count, ] = source_tab_mapper[row['source_system_tab']]
        X_source_screen_name[count, ] = s_scr_name_mapper[row['source_screen_name']]
        X_source_type[count, ] = s_type_mapper[row['source_type']]
        Y[count] = row['target']
        
        input_genre_ids_count[count, ] = row['genre_ids_count']
        input_artist_count[count, ] = row['artist_count']
        input_lyricist_count[count, ] = row['lyricists_count']
        
        input_composer_count[count, ] = row['composer_count']
        input_song_count_played[count, ] = row['count_song_played']
        input_artist_count_played[count, ] = row['count_artist_played']
        
        
        count += 1
    return ([X_msno, X_song_id, X_source_system_tab, X_source_screen_name, X_source_type, input_genre_ids_count, input_artist_count, input_lyricist_count, input_composer_count, input_song_count_played, input_artist_count_played], [Y]) 


# In[14]:


dst = train_dataset[train_dataset.song_id != 'lLsg/q8lurYYpIc3X526qxpWD6SY8Y4grHSCfXlO3mM=']
dst = dst[train_dataset.song_id != '5FXnI1sbD+lFFFGULwZhUAbZMa1P1eFMiigTjMZgW0I=']

train_dataset = dst


# In[15]:


mask_val = 0.8

len2 = len(train_dataset) * mask_val
len2 = int(len2)
temp = train_dataset
train_data = temp[0:len2]
val_data = temp[len2:len(temp)]

print 'Train Data --> ', len(train_data)
print 'Val Data --> ', len(val_data)


# tr = gen_train_data(train_data)


# In[16]:


trX, trY = gen_train_data(train_data)
valX, valY = gen_train_data(val_data)


# In[ ]:


# filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

# callbacks_list = [checkpoint]
# In[ ]:

batch_size = 32768
# generator = input_generator(train_data, batch_size)
# val_generator = input_generator(val_data, batch_size)
# print 'total iterations -- ' , len(train_data)/batch_size

dnn_model.fit(x=trX, y=trY, batch_size = batch_size, epochs = 100, verbose=2, validation_data=(valX, valY),
             shuffle=True)
# dnn_model.fit_generator(generator=generator, steps_per_epoch = (len(train_data)/batch_size), validation_data = val_generator, validation_steps = (len(val_data)/batch_size), epochs=2)

# In[ ]:

model_file = 'lysto_model_no_pre_train_embeds_md_counts_v1_epochs_100.h5'
dnn_model.save(model_file)
print model_file, 'Model saved'

