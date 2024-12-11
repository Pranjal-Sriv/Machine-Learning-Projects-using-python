#!/usr/bin/env python
# coding: utf-8

# # Recommandation system project

# ## Types of recommandation systems
# 1. Content based recommandation system
# 2. Popularity based recommandation system
# 3. Collaborative recommandation system

# ## Steps to be followed
# 1. Data collection
# 2. Data preprocessing
# 3. Feature extraction #to convert text data to numerical
# 4. Similarity score 
# 5. user input
# 6. consine similarity to find similar movies
# 

# ## Key learnings
# 1. Cosine similarity
# 2. TfidfVectorizer
# 3. difflib
# 4. fillna fn
# 5. tolist fn
# 6. enumerate

# In[27]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ## Data collection and preprocessing

# In[1]:


movies_data=pd.read_csv("movies.csv")


# In[2]:


movies_data.tail()


# In[3]:


movies_data.shape


# In[4]:


#selecting the relevent features for recommandation
selected_features=['genres','keywords','tagline','cast','director']
print(selected_features)


# In[10]:


#replacing the null values with null string
for feature in selected_features:
    movies_data[feature]=movies_data[feature].fillna('')


# In[20]:


# Combining all the 5 selected features
combined_features=movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']


# In[21]:


print(combined_features)


# In[22]:


#Converting the text data into numerical
vectorizer=TfidfVectorizer()


# In[23]:


feature_vectors=vectorizer.fit_transform(combined_features)


# In[24]:


print(feature_vectors)


# ## Cosine Similarity

# In[28]:


# getting the similarity scores using cosine similarity
similarity=cosine_similarity(feature_vectors)


# In[29]:


print(similarity)


# In[30]:


print(similarity.shape)


# In[31]:


#Getting the movie name by user
movie_name=input('Enter your fav movie name')


# In[33]:


# Creating a list with all the movie names given in the dataset
list_of_all_titles=movies_data['title'].tolist()
print(list_of_all_titles)


# In[34]:


#Finding the close match for the movie name given by user
find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)
print(find_close_match)


# In[35]:


close_match=find_close_match[0]
print(close_match)


# In[38]:


# Find the index of the movie with title to find similarity score later
index_of_the_movie=movies_data[movies_data.title==close_match]['index'].values[0]
print(index_of_the_movie)


# In[40]:


#Getting the list of similar movies
similarity_score=list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[41]:


len(similarity_score)


# In[42]:


# sorting the movies based on their similarity score
sorted_similar_movies=sorted(similarity_score,key=lambda x:x[1],reverse=True)
print(sorted_similar_movies)


# In[48]:


# print the name of similar movies based on the index
print('Movies suggested:\n')
i=1
for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index=movies_data[movies_data.index==index]['title'].values[0]
    if(i<=10): #recommanding 10 movies
        print(i,'.',title_from_index)
        i+=1
    


# ## Movie Recommandation System

# In[51]:


movie_name=input('Enter your fav movie name')
list_of_all_titles=movies_data['title'].tolist()
find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)
close_match=find_close_match[0]
index_of_the_movie=movies_data[movies_data.title==close_match]['index'].values[0]
similarity_score=list(enumerate(similarity[index_of_the_movie]))
sorted_similar_movies=sorted(similarity_score,key=lambda x:x[1],reverse=True)
print('Movies suggested:\n')
i=1
for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index=movies_data[movies_data.index==index]['title'].values[0]
    if(i<=10): #recommanding 10 movies
        print(i,'.',title_from_index)
        i+=1

