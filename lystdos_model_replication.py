
# coding: utf-8

# # Simple NN with Pretrained Embeddings
# 
# This is our first attempt model. We use pretrained embeddings for user and songs. The embeddings of other columns are computed within this model. All the embeddings are concatenated and go through a softmax before predicting the final output

# In[1]:


import keras
import pandas as pd
from keras.layers import Dense, Activation, Embedding, Input, Concatenate, Flatten
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score
import pickle
import datetime
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--song_emb", type=int, default=64)
parser.add_argument("--user_emb", type=int, default=64)
parser.add_argument("--src_emb", type=int, default=16)
parser.add_argument("--dense", type=int, default=128)

features = ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type']
data = pd.read_csv('./new_data/embedded_data.csv')
language_mapper = pickle.load(open('new_data/model_song_embeddings/language_mapper.pkl', 'rb'))
#data = data[features + ['target']].drop_duplicates(inplace = False).sample(frac=1)
data = data.drop_duplicates(inplace = False)
data['language'] = [language_mapper[s] if s in language_mapper.keys() else 5 for s in data['song_id']]
train_size = int(0.8*data.shape[0])
train_data = data.iloc[:train_size].sample(frac=1)
val_data = data.iloc[train_size:]
test_data = pd.read_csv('./new_data/embedded_test.csv')
#test_data = test_data[features+['id']]


args = parser.parse_args()

song_embedding_size = args.song_emb
user_embedding_size = args.user_emb
source_embedding_size = args.src_emb
batch_size = 32768
num_epochs = 100
extra_dense = args.dense

directory = 'trained_models/{}'.format(datetime.date.today().strftime('%d%b'))
if not os.path.exists(directory):
    print("Creating directory {}".format(directory))
    os.makedirs(directory)

save_path = '{}/full_train_reg_{}_{}_{}_{}.h5'.format(directory, extra_dense, song_embedding_size, user_embedding_size, source_embedding_size)
cont = False
# In[49]:

input_sizes = {
    'song_id': len(pickle.load(open('new_data/model_song_embeddings/song_mapper.pkl', 'rb'))) +1,
    'msno': len(pickle.load(open('new_data/model_user_embeddings/msno_mapper.pkl', 'rb'))) +1,
    'source_system_tab': max(test_data.source_system_tab.max(), data.source_system_tab.max())+1,
    'source_screen_name': max(test_data.source_screen_name.max(), data.source_screen_name.max())+1,
    'source_type': max(data.source_type.max(), test_data.source_type.max())+1,
    'language': max(language_mapper.values())+1,
#    'artist_name': len(pickle.load(open('new_data/model_song_embeddings/artist_mapper.pkl', 'rb'))) +1,
#    'composer': len(pickle.load(open('new_data/model_song_embeddings/composer_mapper.pkl', 'rb'))) +1,
#    'genre_id': len(pickle.load(open('new_data/model_song_embeddings/genre_mapper.pkl', 'rb'))) +1,
#    'lyricist': len(pickle.load(open('new_data/model_song_embeddings/lyricist_mapper.pkl', 'rb'))) +1,
#    'city': len(pickle.load(open('new_data/model_user_embeddings/city_mapper.pkl', 'rb'))) +1,
#    'registered_via': len(pickle.load(open('new_data/model_user_embeddings/reg_via_mapper.pkl', 'rb'))) +1,
#    'registration_year': len(pickle.load(open('new_data/model_user_embeddings/reg_year_mapper.pkl', 'rb'))) +1,
}


# In[50]:




# In[51]:


#song_emb = song_embeddings_model.predict(data.apply(lambda row: song_mapper[row.song_id], axis=1))
#user_emb = user_embeddings_model.predict(data.apply(lambda row: msno_mapper[row.msno], axis=1))
song_input = Input(shape = (1, ))
user_input = Input(shape = (1, ))
s_sys_tab_input = Input(shape = (1, ))
s_scr_name_input = Input(shape = (1, ))
s_type_input = Input(shape = (1, ))
#language = Input(shape=(1,))

song_emb = Flatten()(Embedding(output_dim = song_embedding_size, input_dim=input_sizes['song_id'], embeddings_regularizer=l2(1e-4), embeddings_initializer='glorot_uniform')(song_input))
user_emb = Flatten()(Embedding(output_dim = user_embedding_size, input_dim=input_sizes['msno'], embeddings_regularizer=l2(1e-4), embeddings_initializer='glorot_uniform')(user_input))
s_sys_tab_emb = Flatten()(Embedding(output_dim = source_embedding_size, input_dim=input_sizes['source_system_tab'], embeddings_initializer='glorot_uniform')(s_sys_tab_input))
s_scr_name_emb = Flatten()(Embedding(output_dim = source_embedding_size, input_dim=input_sizes['source_screen_name'], embeddings_initializer='glorot_uniform')(s_scr_name_input))
s_type_emb = Flatten()(Embedding(output_dim = source_embedding_size, input_dim=input_sizes['source_type'], embeddings_initializer='glorot_uniform')(s_type_input))
#lang_emb = Flatten()(Embedding(output_dim = source_embedding_size, input_dim=input_sizes['language'], embeddings_initializer='glorot_uniform')(language))


#embedding_layer = Concatenate(axis=-1)([song_emb, user_emb])

embedding_layer = Concatenate(axis=-1)([song_emb, user_emb,
                                  s_sys_tab_emb, s_scr_name_emb, s_type_emb, lang_emb])
if extra_dense > 0:
    embedding_layer = keras.layers.Dropout(0.5)(
        Dense(extra_dense, activation = 'relu', kernel_initializer = 'glorot_normal')(embedding_layer))

# In[52]:


prediction = Dense(1, activation='sigmoid')(embedding_layer)

#artist_output = Dense(input_sizes['artist_name'], activation='softmax', kernel_initializer='glorot_normal')(embedding_layer)
#composer_output = Dense(input_sizes['composer'], activation='softmax', kernel_initializer='glorot_normal')(embedding_layer)
#genre_output = Dense(input_sizes['genre_id'], activation='softmax', kernel_initializer='glorot_normal')(embedding_layer)
#lyricist_output = Dense(input_sizes['lyricist'], activation='softmax', kernel_initializer='glorot_normal')(embedding_layer)
#
#city_output = Dense(input_sizes['city'], activation='softmax', kernel_initializer='glorot_normal')(embedding_layer)
#reg_via_output = Dense(input_sizes['registered_via'], activation='softmax', kernel_initializer='glorot_normal')(embedding_layer)
#reg_year_output = Dense(input_sizes['registration_year'], activation='softmax', kernel_initializer='glorot_normal')(embedding_layer)

if not cont:


    model = keras.models.Model(inputs=[song_input, user_input, s_sys_tab_input, s_scr_name_input, s_type_input, language],
#                               inputs = [song_input, user_input],
                               outputs = [prediction])
                               #outputs=[prediction, artist_output, composer_output, lyricist_output, genre_output,
                               #        city_output, reg_via_output, reg_year_output])
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

else:
    model = keras.models.load_model(save_path)

early_stopping = EarlyStopping(monitor='val_acc', patience = 5)
model_checkpoint = ModelCheckpoint(save_path, save_best_only = True, save_weights_only=False)
model.fit([train_data.song_id, train_data.msno, train_data.source_system_tab, train_data.source_screen_name, train_data.source_type, train_data.language],
          #[data.song_id, data.msno],
          [train_data.target], epochs = num_epochs, batch_size = batch_size, verbose=1,
          validation_data = ([val_data.song_id, val_data.msno, val_data.source_system_tab, val_data.source_screen_name, val_data.source_type, val_data.language],
                            [val_data.target]),
          callbacks = [early_stopping, model_checkpoint],
          shuffle=True)
          #[data.target, data.artist_name, data.composer, data.lyricist, data.genre_id, data.city,
          #data.registered_via, data.registration_year], epochs = num_epochs, batch_size = batch_size, validation_split=0.1,
          #verbose=1)
model.save(save_path)
preds_val = model.predict([val_data.song_id, val_data.msno, val_data.source_system_tab, val_data.source_screen_name, val_data.source_type], batch_size=batch_size, verbose=2)
print("ROC AUC Score: {}".format(roc_auc_score(val_data.target, preds_val)))
preds_test = model.predict([test_data.song_id, test_data.msno, test_data.source_system_tab, test_data.source_screen_name, test_data.source_type], batch_size=batch_size, verbose=2)
sub = pd.DataFrame({'id': test_data.id, 'target': preds_test.ravel()})
sub.to_csv('./submission_{}_{}_{}_{}.csv'.format(song_embedding_size, user_embedding_size, source_embedding_size, extra_dense), index=False)
