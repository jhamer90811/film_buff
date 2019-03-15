#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:43:53 2019

Author: Jesse Hamer

EDA
"""

import numpy as np
import pandas as pd

ratings = pd.read_csv('datasets/ml-latest/ratings.csv')

ratings.drop(columns='timestamp', inplace=True)

# Get num_ratings for each user

user_ratings = ratings.groupby('userId').size()
user_ratings.name = 'num_ratings_user'
user_ratings = user_ratings.reset_index()

movie_ratings = ratings.groupby('movieId').size()
movie_ratings.name = 'num_ratings_movie'
movie_ratings = movie_ratings.reset_index()

ratings = ratings.merge(movie_ratings, on='movieId')
ratings = ratings.merge(user_ratings, on='userId')

total_ratings = ratings.shape[0]
# 27753444
total_users = ratings.userId.unique().size
# 283228
total_movies = ratings.movieId.unique().size
# 53889

# Distributions of numbers of ratings per user and movie:

ratings.groupby('userId').num_ratings_user.mean().describe(np.arange(0,1,0.05))
# 50% = 30, 95% = 405, 1%=1, 2%=2, 5%=4, 15%=10

ratings.groupby('movieId').num_ratings_movie.mean().describe(np.arange(0,1,0.05))
# 15%=1, 30%=2, 50%=7, 95%=1855, max=97999

# How many users have rated a movie that nobody else has?

unique_reviews_users = ratings.groupby('userId')\
                              .apply(lambda x: (x.num_ratings_movie==1).sum())
print((unique_reviews_users>0).sum()/total_users)
# About 0.67%
print(unique_reviews_users[unique_reviews_users>0].describe(np.arange(0,1,0.05)))
# median is 1; 75% is 3; 95% is 16, 99% is 61
unique_reviews_users[(unique_reviews_users>0)&(unique_reviews_users<=100)]\
                    .hist(bins=50)

# Assumptions: a new user would have at least 10 movies in their collection
# to consider using Film Buff. So only keep users with at least 10 ratings to
# produce a better training set.
# What about movies that have been rated by only one person? The ALS algorithm
# will continue to update its latent factors, but it only receives information
# from this user. It will be deemed similar to other movies viewed by this one
# user, and no one else.
                    
# Question: what is the relationship between the number of ratings for a user,
# and the number of ratings of movies that the user has rated?
                    
avg_movie_ratings_per_user = ratings.groupby('userId').agg([np.mean, np.std])
avg_movie_ratings_per_user = avg_movie_ratings_per_user\
                             .loc[:, [('num_ratings_user', 'mean'), 
                                      ('num_ratings_movie', 'mean'), 
                                      ('num_ratings_movie', 'std')]]
                             
avg_movie_ratings_per_user.columns = ['num_ratings_user_mean',
                                      'num_ratings_movie_mean',
                                      'num_ratings_movie_std']

avg_movie_ratings_per_user.plot.scatter(x='num_ratings_user_mean',
                                        y='num_ratings_movie_mean')
# Shows that: if a user rates movies with high numbers of ratings on average,
# then that user probably has few ratings (converse not necessarily true);
# moreover, if a user has many ratings, they're more likely to rate movies
# with fewer overall ratings (kind of obvious really: if movies were randomly
# assigned to users, then users who receive a larger sample of movies would
# be more likely to rate less-rated films).             
avg_movie_ratings_per_user.plot.scatter(x='num_ratings_user_mean',
                                        y='num_ratings_movie_std')
# Standard deviations are large for low-rating users, but tend to level off
# at about 10K as users rate more films.

# Conclusion: for now, I will keep in users and movies with few ratings,
# given the mild evidence above that users with fewer ratings tend to review
# movies that have many reviews. This should be returned to though and a more
# precise conclusion met as to how minimum user ratings/movie ratings should
# be thresholded.

all_users = ratings.userId.unique()
random_users_1000 = np.random.choice(all_users, 1000, False)

ratings_1000_users = ratings[np.isin(ratings.userId, random_users_1000)]

# Must recompute num_ratings_movie for subsample of users.
ratings_1000_users.drop(columns='num_ratings_movie', inplace=True)

movie_ratings_1000 = ratings_1000_users.groupby('movieId').size()
movie_ratings_1000.name = 'num_ratings_movie'
movie_ratings_1000 = movie_ratings_1000.reset_index()

ratings_1000_users = ratings_1000_users.merge(movie_ratings_1000, on='movieId')
ratings_1000_users.sort_values(['userId','movieId'], inplace=True)

# Conduct same analysis for this to see if it's representative.
total_ratings_1000 = ratings_1000_users.shape[0]
# 94535
total_movies_1000 = ratings_1000_users.movieId.unique().size
# 9597

# Distributions of numbers of ratings per user and movie:

ratings_1000_users.groupby('userId').num_ratings_user.mean().describe(np.arange(0,1,0.05))
# 50% = 29, 95% = 381.3, 1%=1, 2%=2, 5%=4, 15%=11 --> Quite close to whole set!

ratings_1000_users.groupby('movieId').num_ratings_movie.mean().describe(np.arange(0,1,0.05))
# 35%=1, 40%=2, 50%=3, 95%=46, max=335; this sample has more movies with
# only one rating. In general it is populated by movies with fewer ratings.

# How many users have rated a movie that nobody else has?

unique_reviews_users_1000 = ratings_1000_users.groupby('userId')\
                              .apply(lambda x: (x.num_ratings_movie==1).sum())
print((unique_reviews_users_1000>0).sum()/total_users)
# About 0.1%; slightly less than the whole dataset
print(unique_reviews_users_1000[unique_reviews_users_1000>0].describe(np.arange(0,1,0.05)))
# median is 2; 75% is 6; 95% is 45, 99% is 104; max=858. The distribution is
# similar to the overall dataset. 288 total users with unique ratings.
unique_reviews_users_1000[(unique_reviews_users_1000>0)&(unique_reviews_users_1000<=100)]\
                    .hist(bins=50)
                    
# The 1000-user sample should be fairly representative of the overall dataset.
ratings_1000_users.to_csv('datasets/ml-latest/ratings_1000_users.csv',
                          index=False,
                          columns=['userId', 'movieId', 'rating'])                  

                  


                             
                             
                             
                             
                             
                             
                             