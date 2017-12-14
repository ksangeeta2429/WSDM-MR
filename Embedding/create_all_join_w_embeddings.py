
# coding: utf-8

# In[1]:


import pandas as pd
import pickle


# In[2]:


data = pd.read_csv('./data/train.csv').fillna('')


# In[3]:


def create_mapper(values):
    mapper = dict()
    for v in values:
        mapper[v] = len(v)
    return mapper


# In[4]:


source_system_mapper = create_mapper(data.source_system_tab.unique())
s_scr_name_mapper = create_mapper(data.source_screen_name.unique())
s_type_mapper = create_mapper(data.source_type.unique())

pickle.dump(source_system_mapper, open('source_system_mapper.pkl', 'wb'))
pickle.dump(s_scr_name_mapper, open('s_scr_name_mapper.pkl', 'wb'))
pickle.dump(s_type_mapper, open('s_type_mapper.pkl', 'wb'))


# In[5]:


songs_mapper = pickle.load(open('./new_data/model_song_embeddings/song_mapper.pkl', 'rb'))
artist_mapper = pickle.load(open('./new_data/model_song_embeddings/artist_mapper.pkl', 'rb'))
composer_mapper = pickle.load(open('./new_data/model_song_embeddings/composer_mapper.pkl', 'rb'))
genre_mapper = pickle.load(open('./new_data/model_song_embeddings/genre_mapper.pkl', 'rb'))
lyricist_mapper = pickle.load(open('./new_data/model_song_embeddings/lyricist_mapper.pkl', 'rb'))


# In[6]:


user_mapper = pickle.load(open('./new_data/model_user_embeddings/msno_mapper_py2.pkl', 'rb'))
city_mapper = pickle.load(open('./new_data/model_user_embeddings/city_mapper_py2.pkl', 'rb'))
expiry_mapper = pickle.load(open('./new_data/model_user_embeddings/expiry_year_mapper_py2.pkl', 'rb'))
reg_via_mapper = pickle.load(open('./new_data/model_user_embeddings/reg_via_mapper_py2.pkl', 'rb'))
reg_year_mapper = pickle.load(open('./new_data/model_user_embeddings/reg_year_mapper_py2.pkl', 'rb'))


# In[7]:


songs_data = pd.read_csv('./new_data/joined.csv')
users_data = pd.read_csv('./new_data/mem_shortlist.csv')


# In[8]:


data = data.merge(songs_data, on='song_id', how='left').merge(users_data, on='msno', how='left').fillna('')


# In[ ]:



def transform(row):
    if transform.count % 1000000 == 0:
        print(transform.count)

    if row.song_id in songs_mapper:
        row.song_id = songs_mapper[row.song_id]
    else:
        row.song_id = songs_mapper['']
    row.artist_name = artist_mapper[row.artist_name]
    row.composer = composer_mapper[row.composer]
    row.genre_id = genre_mapper[str(row.genre_id)]
    row.lyricist = lyricist_mapper[row.lyricist]
    
    row.msno = user_mapper[row.msno]
    row.city = city_mapper[str(row.city)]
    row.registered_via = reg_via_mapper[str(row.registered_via)]
    row['registration_year'] = reg_year_mapper[str(row.registration_init_time)[:4]] if row.registration_init_time is not '' else ''
    
    row.source_system_tab = source_system_mapper[row.source_system_tab]
    row.source_screen_name = s_scr_name_mapper[row.source_screen_name]
    row.source_type = s_type_mapper[row.source_type]
    
    del row['bd']
    del row['gender']
    del row['registration_init_time']
    del row['expiration_date']
    
    transform.count += 1
    return row

#transform.count = 0
#
#import ipdb
#ipdb.set_trace()
#new_data = data.apply(transform, axis=1)
#
#
#new_data.head()
#
#
## In[ ]:
#
#
#new_data.to_csv('./compute_ready.csv', index=False)


# In[ ]:

del data['bd']
del data['gender']
del data['expiration_date']

data.song_id = data.song_id.apply(lambda x: songs_mapper[x] if x in songs_mapper else songs_mapper[''])
data.artist_name = data.artist_name.apply(lambda x: artist_mapper[str(x).lower()] if str(x).lower() in artist_mapper else artist_mapper[''])
data.composer = data.composer.apply(lambda x: composer_mapper[str(x).lower()] if str(x).lower() in composer_mapper else composer_mapper[''])
data.genre_id = data.genre_id.apply(lambda x: genre_mapper[str(int(x))] if x != '' else genre_mapper[''])
data.lyricist = data.lyricist.apply(lambda x: lyricist_mapper[str(x).lower()] if str(x).lower() in lyricist_mapper else lyricist_mapper[''])

data.msno = data.msno.apply(lambda x: user_mapper[x] if x in user_mapper else user_mapper[''])
data.city = data.city.apply(lambda x: city_mapper[str(int(x))] if x != '' else city_mapper[''])
data.registered_via = data.registered_via.apply(lambda x: reg_via_mapper[str(int(x))] if x != '' else reg_via_mapper[''])
data['registration_year'] = data.registration_init_time.apply(lambda x: reg_year_mapper[str(x)[:4]] if data.registration_init_time is not '' else 0)
del data['registration_init_time']

data.source_system_tab = data.source_system_tab.apply(lambda x: source_system_mapper[x])
data.source_screen_name = data.source_screen_name.apply(lambda x: s_scr_name_mapper[x])
data.source_type = data.source_type.apply(lambda x: s_type_mapper[x])

data.to_csv('./embedded_data.csv', index=False)
