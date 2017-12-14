# coding: utf-8

# In[6]:

import numpy as np
import pandas as pd
from langid.langid import LanguageIdentifier, model
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
from googletrans import Translator
translator = Translator()
import time
import sys
import re
default_stdout = sys.stdout
default_stderr = sys.stderr

reload(sys)
sys.setdefaultencoding('utf8')

sys.stdout = default_stdout
sys.stderr = default_stderr

import progressbar


# In[4]:

fileName = 'extra_song_names.csv'
outputFileName = 'tr_extra_song_names.csv'
col_to_be_modified = 'name'
pattern = '\s\(.+\)'


# In[371]:

artists = pd.read_csv(fileName, index_col=None)


# In[373]:

artists['translated_names'] = artists[col_to_be_modified]
# Adding a new column to the existing table
dict_names = {}


# In[2]:

def translate_sent(sent,i, dest='en'):
    count = 0
    while count <= 5 :
        try :
            y = translator.translate(sent.decode('utf8'), dest='en')
            text = y.text
            dict_names[sent] = text
            time.sleep(1)
            return text
        except Exception as e:
            print 'Caught exception, retrying for ', sent, i
            time.sleep(pow(2, count))
        count += 1
    return sent


# In[377]:

print 'Starting translation'
bar = progressbar.ProgressBar()
for i in bar(range(0,len(artists))):
    currName = str(artists.iloc[i][col_to_be_modified])
    # Check if current name contains brackets. If it does, drop the parts in brackets...
    currName = re.sub(pattern, '', currName)
    #print currName
    lang, pred = identifier.classify(currName)  # Using LangId
    #     print lang, pred, artists.iloc[i]['artist_name'], i
    if (lang != 'en'):
        if currName in dict_names:
            text = dict_names[currName]
        else:
            text = translate_sent(currName, i, dest='en')
        print text, i
        artists.set_value(i, 'translated_names', text)
    else:
    #Copy the reduced name, i.e. bracket removed name
        artists.set_value(i, 'translated_names', currName)  
print 'Translation finished'


# In[5]:

def write_to_csv(dataframe):
    dataframe.to_csv(outputFileName, index=False)


# In[379]:

write_to_csv(artists)

