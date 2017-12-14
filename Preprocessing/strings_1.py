
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt


# In[2]:

rcolumn = 'composer'
headers = ['song_id', rcolumn]
new_songs = pd.read_csv('../Data/shortlisted_song.csv', usecols = headers, na_filter=True)
new_songs = new_songs.dropna(axis = 0)
#print new_songs


# In[3]:

def split_str(string):
    multiple = re.split('/|,|\|', string)
    return multiple

def brack_entry(string):
    a_1 = string[string.find("(")+1:string.find(")")]
    a_2 = string[0:string.find("(")-1]
    a_1 = ''.join([i for i in a_1 if not i.isdigit()])
    a_2 = ''.join([i for i in a_2 if not i.isdigit()])
    return a_1, a_2


# In[4]:

df_new = pd.DataFrame(columns=['index', 'song_id', rcolumn])
artists = new_songs[rcolumn]

for row_index, row in new_songs.iterrows():
    artist = split_str(row[rcolumn])
    if len(artist) != 0:
        for i in range(len(artist)):
            df_new.loc[df_new.shape[0]] = [df_new.shape[0], row['song_id'], artist[i]]
    else:
        df_new.loc[df_new.shape[0]] = [df_new.shape[0], row['song_id'], row[rcolumn]]


# In[5]:

#print len(df_new)
#print df_new


# In[6]:

length = len(df_new)
for row_index, row in df_new.iterrows():
    string = row[rcolumn]
    if string.find('(') <> -1:
        a1, a2 = brack_entry(string)
        df_new.loc[length] = [length, row['song_id'], a1]
        df_new.loc[length+1] = [length+1, row['song_id'], a2]
        length = length + 2
        df_new = df_new.drop(row['index'])


# In[7]:

#print df_new


# In[8]:

df_new[rcolumn] = df_new[rcolumn].map(lambda x: x.lstrip(" "))


# In[9]:

#print df_new


# In[10]:

df_new.to_csv(rcolumn+'.csv', index=False)


# In[ ]:



