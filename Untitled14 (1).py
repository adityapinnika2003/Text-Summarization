#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding


# In[3]:


df = pd.read_parquet("cnn_dailymail-test.parquet")


# In[4]:


df


# In[5]:


train_data, test_data = train_test_split(df, test_size=0.2)


# In[6]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['article'])
vocab_size = len(tokenizer.word_index) + 1


# In[7]:


train_sequences = tokenizer.texts_to_sequences(train_data['article'])
test_sequences = tokenizer.texts_to_sequences(test_data['article'])


# In[8]:


max_len = 100
train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post')


# In[9]:


train_highlights = tokenizer.texts_to_sequences(train_data['highlights'])
test_highlights = tokenizer.texts_to_sequences(test_data['highlights'])


# In[10]:


train_highlights_padded = pad_sequences(train_highlights, maxlen=max_len, padding='post')
test_highlights_padded = pad_sequences(test_highlights, maxlen=max_len, padding='post')


# In[11]:


model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_len))
model.add(LSTM(128, return_sequences=True))
model.add(Dense(64, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))


# In[12]:


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.fit(train_padded,train_highlights_padded,validation_data=(test_padded,test_highlights_padded),epochs=1, batch_size=32)


# In[ ]:




