#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 23:17:55 2019

@author: Afrochemist
"""

#Common Libraries
import pandas as pd
import numpy as np

#for machine learning
from sklearn.metrics.pairwise import pairwise_distances
import scipy.sparse as sp
from scipy.sparse.linalg import svds

#for calculating the RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt

#Uploading the data
df =pd.read_csv("Data/movie_ratings.csv")

#Getting to know the data
df.head()

df.describe()

n_users = df.userID.unique().shape[0]
n_items = df.itemID.unique().shape[0]
print('\nNumber of users = ' + str(n_users) + ' | Numbe of movies = ' + str(n_items))

#Create user-item similarity matrices
df_matrix = np.zeros((n_users, n_items))
for line in df.itertuples():
    df_matrix[line[1]-1, line[2],-1] = line[3]
    
#For cosine similarity
user_similarity = pairwise_distances(df_matrix, metric='cosine')
item_similarity = pairwise_distances(df_matrix.T, metric='cosine')

#Top 3 similar users for user id 7
print("Similar users for users id 7: \n",pd.DataFrame(user_similarity).loc[6,pd.DataFrame(user_similarity).loc[6,:] > 0 ].sort_values(ascending=False)[0:3])

#Top 3 similar users for user id 6
print("Similar users for users id 6: \n",pd.DataFrame(user_similairty).loc[5,pd.DataFrame(user_similarity).loc[5,:] > 0 ].sort_values(ascending=False)[0:3])

#Function for item based rating prediction
def item_based_prediction(rating_matrix, simialrity_matrix):
    return rating_matrix.dot(similarity) / np.array([np.abs(similarity_matrix).sum(axis=1)])

#Function for user based rating prediction
    def user_based_prediction(rating_matrix, similarity_matrix):
        mean_user_rating = rating_matrix.mean(axis=1)
        raintgs_diff = (rating_matrix - mean_user_rating[:, np.newaxis])
        return mean_user_rating[:, np.newaxis] + similarity_matrix.dot(ratings_diff) / np.array([np.abs(similarity_matrix).sum(axis=1)]).T

#Now utilizing the functions
item_based_prediction = item_based_prediction(df_,atrix, item_similarity)
user_based_prediction = user_based_prediction(df_matrix, user_similarity)

#Calculating the RMSE
def rmse(prediction, actual):
    prediction = prediction[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, actual))

#Printing RMSE for both functions
print("User-based CF RMSE: " + str(rmse(user_based_prediction, df_matrix)))
print("Item-based CF RMSE: " + str(rmse(item_based_prediction, df_matrix)))


#Now storing data in a pandas Dataframe
y_used_based = pd.DataFrame(user_based_prediction)

#Predictions for mobies that the user id 6 hasn't rated yet
predictions = y_used_based.loc[6,pd.DataFrame(df_matrix).loc[6,:] == 0]
top = predictions.sort_values(ascending=False).head(n=1)
recommendations = pd.DataFrame(data=top)
recommendations.columns = ['Predicted Rating']
print(recommendations)

#Now fror the item-based recommender
y_item_based = pd.DataFrame(item_based_prediction)

#Predictions for movies that the user user 6 hasn't rated yet
predictions = y_item_based.loc[6,pd.DataFrame(df_matrix).loc[6,:] == 0]
top = predictions.sort_values(ascending=False).head(n=1)
recommendations = pd.DataFrame(data=top)
recommendations.columns = ['Predicted Rating']
print(recommendations)


#Now implementing SVD
#comment explaining what SVD is

#Calculate sparsity level
sparsity = round(1.0-len(df)/float(n_users*n_items),3)
print("The sparsity level of is ") + str(sparsity*100) + '%'

#Get SVD Components from train matrix. Choose K.
u, s, vt = svds(df_matrix, k = 5)
s_diag_matrix = np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print("User-based CF MSE: " + str(rmse(X_pred, df_matrix)))


