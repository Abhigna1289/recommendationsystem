#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the required libraries
import numpy as np
import pandas as pd
import pickle
import matrix_factorization_utilities
import scipy.sparse as sp
from scipy.sparse.linalg import svds


# In[2]:


# Reading the ratings data
ratings = pd.read_csv(r'C:\Users\kumari\Desktop\fintech\AI & ML\Movie-Recommendation-System-master\Dataset\ratings.csv')


# In[3]:


#Just taking the required columns
ratings = ratings[['userId', 'movieId','rating']]


# In[6]:


ratings


# In[4]:


# Checking if the user has rated the same movie twice, in that case we just take max of them
ratings_df = ratings.groupby(['userId','movieId']).aggregate(np.max)


# In[5]:


#Getting the percentage count of each rating value 
count_ratings = ratings.groupby('rating').count()
count_ratings['perc_total']=round(count_ratings['userId']*100/count_ratings['userId'].sum(),1)


# In[9]:


count_ratings


# In[74]:


#Visualising the percentage total for each rating
count_ratings['perc_total'].plot.bar()


# In[6]:


#reading the movies dataset
movie_list = pd.read_csv(r'C:\Users\kumari\Desktop\fintech\AI & ML\Movie-Recommendation-System-master\Dataset\movies.csv')


# In[8]:


# reading the tags datast
tags = pd.read_csv(r'C:\Users\kumari\Desktop\fintech\AI & ML\Movie-Recommendation-System-master\Dataset\tags.csv')


# In[9]:


# inspecting various genres
genres = movie_list['genres']


# In[10]:


genre_list = ""
for index,row in movie_list.iterrows():
        genre_list += row.genres + "|"
#split the string into a list of values
genre_list_split = genre_list.split('|')
#de-duplicate values
new_list = list(set(genre_list_split))
#remove the value that is blank
new_list.remove('')
#inspect list of genres
new_list


# In[84]:


#Enriching the movies dataset by adding the various genres columns.
movies_with_genres = movie_list.copy()

for genre in new_list :
    movies_with_genres[genre] = movies_with_genres.apply(lambda _:int(genre in _.genres), axis = 1)


# In[85]:


movies_with_genres.head()


# In[91]:


# Finding the average rating for movie and the number of ratings for each movie
avg_movie_rating = pd.DataFrame(ratings.groupby('movieId')['rating'].agg(['mean','count']))
avg_movie_rating['movieId']= avg_movie_rating.index


# In[92]:


avg_movie_rating.reset_index(drop=True, inplace=True)


# In[93]:


avg_movie_rating.head()


# In[14]:


#calculate the percentile count. It gives the no of ratings at least 70% of the movies have
np.percentile(avg_movie_rating['count'],70)


# In[94]:


#Get the average movie rating across all movies 
avg_rating_all=ratings['rating'].mean()
avg_rating_all
#set a minimum threshold for number of reviews that the movie has to have
min_reviews=30
min_reviews
movie_score = avg_movie_rating.loc[avg_movie_rating['count']>min_reviews]
movie_score.head()


# In[16]:


#create a function for weighted rating score based off count of reviews
def weighted_rating(x, m=min_reviews, C=avg_rating_all):
    v = x['count']
    R = x['mean']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[96]:


#Calculating the weighted score for each movie
movie_score['weighted_score'] = movie_score.apply(weighted_rating, axis=1)


# In[97]:


movie_score.head()


# In[98]:


movie_score = pd.merge(movie_score,movies_with_genres,how ='inner',on='movieId')


# In[99]:


movie_score


# In[100]:


#list top scored movies over the whole range of movies
pd.DataFrame(movie_score.sort_values(['weighted_score'],ascending=False)[['title','count','mean','weighted_score','genres']][:10])


# In[101]:


# Gives the best movies according to genre based on weighted score which is calculated using IMDB formula
def best_movies_by_genre(genre,top_n):
    return pd.DataFrame(movie_score.loc[(movie_score[genre]==1)].sort_values(['weighted_score'],ascending=False)[['title','count','mean','weighted_score']][:top_n])


# In[39]:


#run function to return top recommended movies by genre
best_movies_by_genre('Musical',10)  


# In[22]:


#merging ratings and movies dataframes
ratings_movies = pd.merge(ratings,movie_list, on = 'movieId')


# In[105]:


ratings_movies.head()


# In[23]:


#Gets the other top 10 movies which are watched by the people who saw this particular movie
def get_other_movies(movie_name):
    #get all users who watched a specific movie
    df_movie_users_series = ratings_movies.loc[ratings_movies['title']==movie_name]['userId']
    #convert to a data frame
    df_movie_users = pd.DataFrame(df_movie_users_series,columns=['userId'])
    #get a list of all other movies watched by these users
    other_movies = pd.merge(df_movie_users,ratings_movies,on='userId')
    #get a list of the most commonly watched movies by these other user
    other_users_watched = pd.DataFrame(other_movies.groupby('title')['userId'].count()).sort_values('userId',ascending=False)
    other_users_watched['perc_who_watched'] = round(other_users_watched['userId']*100/other_users_watched['userId'][0],1)
    return other_users_watched[:10]


# In[26]:


# Getting other top 10 movies which are watched by the people who saw 'Gone Girl'
get_other_movies('Gone Girl (2014)')


# In[25]:


#get ordered list of movieIds
item_indices = pd.DataFrame(sorted(list(set(ratings['movieId']))),columns=['movieId'])
#add in data frame index value to data frame
item_indices['movie_index']=item_indices.index
#inspect data frame
item_indices.head()


# In[26]:


#get ordered list of movieIds
user_indices = pd.DataFrame(sorted(list(set(ratings['userId']))),columns=['userId'])
#add in data frame index value to data frame
user_indices['user_index']=user_indices.index
#inspect data frame
user_indices.head()


# In[106]:


#join the movie indices
df_with_index = pd.merge(ratings,item_indices,on='movieId')
#join the user indices
df_with_index=pd.merge(df_with_index,user_indices,on='userId')
#inspec the data frame
df_with_index.head()


# In[28]:


#import train_test_split module
from sklearn.model_selection import train_test_split
#take 80% as the training set and 20% as the test set
df_train, df_test= train_test_split(df_with_index,test_size=0.2)
print(len(df_train))
print(len(df_test))


# In[327]:


df_train.head()


# In[33]:


df_test.head()


# In[29]:


n_users = ratings.userId.unique().shape[0]
n_items = ratings.movieId.unique().shape[0]
print(n_users)
print(n_items)


# In[30]:


#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
    #for every line in the data
for line in df_train.itertuples():
    #set the value in the column and row to 
    #line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
    train_data_matrix[line[5], line[4]] = line[3]
train_data_matrix.shape


# In[31]:


#Create two user-item matrices, one for training and another for testing
test_data_matrix = np.zeros((n_users, n_items))
    #for every line in the data
for line in df_test[:1].itertuples():
    #set the value in the column and row to 
    #line[1] is userId, line[2] is movieId and line[3] is rating, line[4] is movie_index and line[5] is user_index
    #print(line[2])
    test_data_matrix[line[5], line[4]] = line[3]
    #train_data_matrix[line['movieId'], line['userId']] = line['rating']
test_data_matrix.shape


# In[107]:


pd.DataFrame(train_data_matrix).head()


# In[37]:


df_train['rating'].max()


# In[32]:


#from sklearn.metrics import mean_squared_error
#from math import sqrt
#def rmse(prediction, ground_truth):
    #select prediction values that are non-zero and flatten into 1 array
    #prediction = prediction[ground_truth.nonzero()].flatten() 
    #select test values that are non-zero and flatten into 1 array
    #ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    #return RMSE between values
    #return sqrt(mean_squared_error(prediction, ground_truth))


# In[33]:


#Calculate the rmse sscore of SVD using different values of k (latent features)
rmse_list = []
#for i in [1,2,5,20,40,60,100,200]:
    #apply svd to the test data
    u,s,vt = svds(train_data_matrix,k=i)
    #get diagonal matrix
    s_diag_matrix=np.diag(s)
    #predict x with dot product of u s_diag and vt
    X_pred = np.dot(np.dot(u,s_diag_matrix),vt)
    #calculate rmse score of matrix factorisation predictions


# In[34]:


#Convert predictions to a DataFrame
mf_pred = pd.DataFrame(X_pred)
mf_pred.head()


# In[108]:


df_names = pd.merge(ratings,movie_list,on='movieId')
df_names.head()


# In[227]:


#choose a user ID
user_id = int(input('Enter User Id'))
#get movies rated by this user id
users_movies = df_names.loc[df_names["userId"]==user_id]
#print how many ratings user has made 
print("User ID : " + str(user_id) + " has already rated " + str(len(users_movies)) + " movies")
#list movies that have been rated
users_movies


# In[38]:


user_index = df_train.loc[df_train["userId"]==user_id]['user_index'][:1].values[0]
#get movie ratings predicted for this user and sort by highest rating prediction
sorted_user_predictions = pd.DataFrame(mf_pred.iloc[user_index].sort_values(ascending=False))
#rename the columns
sorted_user_predictions.columns=['ratings']
#save the index values as movie id
sorted_user_predictions['movieId']=sorted_user_predictions.index
print("Top 10 predictions for User " + str(user_id))
#display the top 10 predictions for this user
pd.merge(sorted_user_predictions,movie_list, on = 'movieId')[:10]


# In[39]:


#count number of unique users
numUsers = df_train.userId.unique().shape[0]
#count number of unitque movies
numMovies = df_train.movieId.unique().shape[0]
print(len(df_train))
print(numUsers) 
print(numMovies) 


# In[40]:


#import libraries
import keras
from keras.layers import Embedding, Reshape, concatenate
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[41]:


from keras.utils import plot_model


# In[69]:


# Couting no of unique users and movies
len(ratings.userId.unique()), len(ratings.movieId.unique())


# In[42]:


# Assigning a unique value to each user and movie in range 0,no_of_users and 0,no_of_movies respectively.
ratings.userId = ratings.userId.astype('category').cat.codes.values
ratings.movieId = ratings.movieId.astype('category').cat.codes.values


# In[48]:


# Splitting the data into train and test.
train, test = train_test_split(ratings, test_size=0.2)


# In[56]:


train.head()


# In[57]:


test.head()


# In[44]:


n_users = len(ratings.userId.unique())
n_users


# In[45]:


n_movies =  len(ratings.movieId.unique())
n_movies


# In[183]:


# Returns a neural network model which performs matrix factorisation
from keras.layers import Input, Flatten, Dot
from keras.models import Model

movie_input = Input(shape=[1],name='Item')
movie_embedding = Embedding(n_movies+1, 5, name='Movie-Embedding')(movie_input)
movie_vec = Flatten(name='Flatten-Movies')(movie_embedding)

user_input = Input(shape=[1],name='users')
user_embedding = Embedding(n_users+1, 5, name='User-Embedding')(user_input)
user_vec = Flatten(name='Flatten-Users')(user_embedding)

prod = Dot(name='Dot-Product', axes=1)([movie_vec, user_vec])
model = Model([user_input, movie_input], prod)
model.compile('adam', 'mean_squared_error',metrics=['acc'])


# In[229]:


import os
from keras.models import load_model
if os.path.exists('regression_model.h5'):
    model = load_model('regression_model.h5')
else:
    history = model.fit([train.userId, train.movieId], train.rating, epochs=10,verbose=1,batch_size=500,validation_split=0.2)


# In[220]:


from matplotlib import pyplot 
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('model accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.legend(['training', 'validation'], loc='lower right')
pyplot.show()


# In[221]:


epochs=range(1, 11)
pyplot.clf()
acc_values = history.history['acc']
val_acc_values = history.history['val_acc']
pyplot.plot(epochs, acc_values, 'bo', label='Training acc')
pyplot.plot(epochs, val_acc_values, 'b', label='Validation acc')
pyplot.title('Training and validation accuracy')
pyplot.xlabel('epochs')
pyplot.ylabel('loss')
pyplot.legend()
pyplot.show()


# In[222]:


pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['training', 'validation'], loc='upper right')
pyplot.show()


# In[223]:


loss_values = history.history['loss']
val_loss_values = history.history['val_loss']
pyplot.plot(epochs, loss_values, 'bo', label='Training loss')
pyplot.plot(epochs, val_loss_values, 'b', label='Validation loss')
pyplot.title('Training and validation loss')
pyplot.xlabel('Epochs')
pyplot.ylabel('Loss')
pyplot.legend()
pyplot.show()


# In[196]:


model.summary()


# In[216]:


y_hat = np.round(model.predict([test.userId, test.movieId]),0)
y_true = test.rating


# In[174]:


print(y_hat)
print(y_true)


# In[217]:


from sklearn.metrics import mean_absolute_error
mbe = mean_absolute_error(y_true, y_hat)
print(mbe)


# In[218]:


from sklearn.metrics import mean_squared_error


mse = mean_squared_error(y_hat, y_true)

print(mse)


# In[219]:


from math import *
import math
rmse = math.sqrt(mse)
print(rmse)


# In[207]:


from sklearn.metrics import r2_score
r2_score(y_true, y_hat)


# In[230]:


from keras.layers import Input, Flatten, Dot, Concatenate, Dense
from keras.models import Model

movie_input = Input(shape=[1],name='Item')
movie_embedding = Embedding(n_movies+1, 5, name='Movie-Embedding')(movie_input)
movie_vec = Flatten(name='Flatten-Movies')(movie_embedding)

user_input = Input(shape=[1],name='users')
user_embedding = Embedding(n_users+1, 5, name='User-Embedding')(user_input)
user_vec = Flatten(name='Flatten-Users')(user_embedding)

conc = Concatenate()([movie_vec, user_vec])

# add fully-connected-layers
fc1 = Dense(128, activation='relu')(conc)
fc2 = Dense(32, activation='relu')(fc1)
out = Dense(1)(fc2)

# Create model and compile it
model2 = Model([user_input, movie_input], out)
model2.compile('adam', 'mean_squared_error',metrics=['acc'])


# In[232]:


from keras.models import load_model
if os.path.exists('regression_model2.h5'):
    model2 = load_model('regression_model2.h5')
else:
    history = model2.fit([train.userId, train.movieId], train.rating, epochs=10, verbose=1,batch_size=500,validation_split=0.2)


# In[78]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Training Error")


# In[233]:


model2.summary()


# In[234]:


from matplotlib import pyplot 
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('model accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.legend(['training', 'validation'], loc='lower right')
pyplot.show()


# In[235]:


epochs=range(1, 11)
pyplot.clf()
acc_values = history.history['acc']
val_acc_values = history.history['val_acc']
pyplot.plot(epochs, acc_values, 'bo', label='Training acc')
pyplot.plot(epochs, val_acc_values, 'b', label='Validation acc')
pyplot.title('Training and validation accuracy')
pyplot.xlabel('epochs')
pyplot.ylabel('loss')
pyplot.legend()
pyplot.show()


# In[236]:


pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['training', 'validation'], loc='upper right')
pyplot.show()


# In[237]:


loss_values = history.history['loss']
val_loss_values = history.history['val_loss']
pyplot.plot(epochs, loss_values, 'bo', label='Training loss')
pyplot.plot(epochs, val_loss_values, 'b', label='Validation loss')
pyplot.title('Training and validation loss')
pyplot.xlabel('Epochs')
pyplot.ylabel('Loss')
pyplot.legend()
pyplot.show()


# In[238]:


y_hat = np.round(model2.predict([test.userId, test.movieId]),0)
y_true = test.rating


# In[241]:


from sklearn.metrics import mean_squared_error


mse = mean_squared_error(y_hat, y_true)

print(mse)


# In[242]:


from sklearn.metrics import mean_absolute_error
mbe = mean_absolute_error(y_true, y_hat)
print(mbe)


# In[243]:


from math import *
import math
rmse = math.sqrt(mse)
print(rmse)


# In[ ]:




