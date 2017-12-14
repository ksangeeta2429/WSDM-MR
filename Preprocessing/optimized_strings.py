import numpy as np
import pandas as pd
import re
import progressbar

# In[2]:

rcolumn = 'name'
headers = ['song_id', rcolumn]
new_songs = pd.read_csv('./songnames.csv', usecols = headers, na_filter=True)
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

artists = new_songs[rcolumn]

bar = progressbar.ProgressBar()
data = [None]*new_songs.shape[0]
extra_data = []
count = 0

for row_index, row in bar(new_songs.iterrows()):
    artist = split_str(row[rcolumn])
    if len(artist) == 0:
        data[count] = [row['song_id'], row[rcolumn]]
    elif len(artist) == 1:
        data[count] = [row['song_id'], artist[0]]
    else:
        data[count] = [row['song_id'], artist[0]]
        for i in range(1, len(artist)):
            extra_data.append([row['song_id'], artist[i]])
    count += 1

data = data[:count]
data.extend(extra_data)

# In[5]:

#print len(df_new)
#print df_new


# In[6]:

new_data = [None]*int(len(data)*1.2)
count = 0
bar = progressbar.ProgressBar()
for i in bar(range(len(data))):
    row = data[i]
    string = row[1]
    if not string.find('(') == -1:
        a1, a2 = brack_entry(string)
        new_data[count] = [row[0], a1]
        new_data[count+1] = [row[0], a2]
        count += 2
    else:
        new_data[count] = [row[0], row[1]]
        count += 1

df_new = pd.DataFrame(new_data[:count], columns=['song_id', rcolumn])

# In[7]:

#print df_new


# In[8]:

df_new[rcolumn] = df_new[rcolumn].map(lambda x: x.strip())


# In[9]:

#print df_new


# In[10]:

df_new.to_csv(rcolumn+'.csv', index=False)