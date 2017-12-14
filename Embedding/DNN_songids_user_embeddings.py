
# coding: utf-8

# In[124]:


''' This notebook contains the DNN model for song_embeddings + user_embeddings '''


# In[125]:


import numpy as np
import math
import pandas as pd

from tqdm import tqdm
import keras
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, Flatten
from keras.callbacks import ModelCheckpoint
import cPickle as pickle
import progressbar
import os


# In[126]:


def load_song_model():
    '''This method loads the song embeddings model into memory'''
    model = keras.models.load_model('../New_Data/model_song_embeddings/songs_embeddings_100.h5')
    return model


# In[127]:


def load_dataset():
    embedded_dataset = pd.read_csv('../../new_data/New_Data/embedded_data.csv')
    return embedded_dataset


# In[128]:


embedded_dataset = load_dataset()


# In[129]:


print embedded_dataset.shape


# In[130]:


print embedded_dataset.head()


# In[131]:


song_orig_model = load_song_model()


# In[132]:


song_orig_embedding_layer_model = Model(inputs=song_orig_model.input,
                                 outputs=song_orig_model.get_layer('embedding_1').output)


# In[133]:


unique_song_ids = embedded_dataset['song_id'].unique()


# In[134]:


index = np.argwhere(unique_song_ids >= 419868)
unique_song_ids = np.delete(unique_song_ids, index)


# In[135]:


song_weights = song_orig_embedding_layer_model.predict(unique_song_ids)


# In[136]:


def save_to_pickle(data, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(data, fw, protocol=2) #Python 2

def read_from_pickle(filename):
    return pickle.load(open(filename, 'r'))


# In[137]:


song_weights = np.squeeze(song_weights)
print song_weights.shape


# In[138]:


song_weights_dict = {}
bar = progressbar.ProgressBar()
for i in bar(range(len(unique_song_ids))):
    weight = song_weights[i]
    id_ = unique_song_ids[i]
    #Save to dict
    song_weights_dict[id_] = weight

#Save the dict
save_to_pickle(song_weights_dict, '../New_Data/unique_song_weights.pkl')


# In[139]:


### Now let's get the user embeddings
def load_user_model():
    model = keras.models.load_model('../New_Data/model_user_embeddings/user_embeddings_100.h5')
    return model


# In[140]:


user_model = load_user_model()


# In[141]:


user_model.summary()


# In[142]:


user_embeddings_layer = Model(inputs=user_model.input,outputs=user_model.get_layer('embedding_1').output) 


# In[143]:


unique_user_ids = embedded_dataset['msno'].unique()


# In[144]:


user_weights = user_embeddings_layer.predict(unique_user_ids)


# In[145]:


user_weights = np.squeeze(user_weights)
print user_weights.shape


# In[146]:


user_weights_dict = {}
bar = progressbar.ProgressBar()
for i in bar(range(len(unique_user_ids))):
    weight = user_weights[i]
    id_ = unique_user_ids[i]
    ## Save to the dict
    user_weights_dict[id_] = weight

#Save the dict
save_to_pickle(user_weights_dict, '../New_Data/unique_user_weights.pkl')


# In[147]:


### Now that we have the song_embeddings, let's just try to train the model ### 


# In[148]:


input_song_ids_layer = Input(shape=(100,))
input_msno_layer = Input(shape=(50, ))


# In[149]:


combined_input = keras.layers.concatenate([input_msno_layer, input_song_ids_layer])


# In[150]:


intermediate_0 = Dense(25)(combined_input)
intermediate_1 = Dense(10)(intermediate_0)
output_0 = Dense(1, activation='sigmoid')(intermediate_1)

print output_0.shape


# In[151]:


dnn_model = keras.models.Model(inputs = [input_msno_layer, input_song_ids_layer],
                               outputs = [output_0])



from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(dnn_model).create(prog='dot', format='svg'))


# In[1]:
mask_val = np.random.rand(len(embedded_dataset)) < 0.9
train_data = embedded_dataset[mask_val]
val_data = embedded_dataset[~mask_val]

print 'Train Data --> ', len(train_data)
print 'Val Data --> ', len(val_data)

def batch_gen(data, index_low, index_high):
    batch = index_high - index_low
    X_msno = np.zeros((batch, 50), dtype='float32')
    X_song_id = np.zeros((batch, 100), dtype='float32')
    Y = np.empty(batch)
    count = 0
    for row_num in (range(index_low, index_high)):
        curr_song_id = data.iloc[row_num]['song_id']
        curr_msno = data.iloc[row_num]['msno']
        curr_target = data.iloc[row_num]['target']
        if curr_song_id in song_weights_dict:
            X_msno[count,] = user_weights_dict[curr_msno]
            X_song_id[count, ] = song_weights_dict[curr_song_id]
            Y[count] = curr_target
            count += 1
    return ([X_msno, X_song_id], [Y]) 

def input_generator(data, batch_size):
    num_rows = len(data)
    
    if os.path.isfile('../New_Data/unique_song_weights.pkl'):
        song_weights_dict = read_from_pickle('../New_Data/unique_song_weights.pkl')
    if os.path.isfile('../New_Data/unique_user_weights.pkl'):
        user_weights_dict = read_from_pickle('../New_Data/unique_user_weights.pkl')
    while True:
        count = 0
        while count < num_rows/batch_size:
            index_low = count * batch_size
            index_high = (count + 1) * batch_size
            count += 1
            yield batch_gen(data, index_low, index_high)


# In[ ]:


dnn_model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:

batch_size = 128
generator = input_generator(train_data, batch_size)
val_generator = input_generator(val_data, batch_size)
print 'total iterations -- ' , len(train_data)/batch_size
dnn_model.fit_generator(generator=generator, steps_per_epoch = (len(train_data)/batch_size), validation_data = val_generator, validation_steps = (len(val_data)/batch_size), epochs=5)

# In[ ]:


dnn_model.save('dnn_with_users_song_embeddings_with_batch_128_5_epoch.h5')


# In[ ]:


#!ipython nbconvert --to script DNN_songs_user_embeddings.ipynb

