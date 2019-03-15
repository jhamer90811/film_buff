#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:38:38 2019

@author: Jesse Hamer

Contains scratch work for custom-built recommendation engine. The engine will
implement the collaborative filtering algorithm outlined in "Online-Updating
Regularized Kernel Matrix Factorization Models for Large Scale Recommender
Systems". Functionality is needed for online-updating as well as proper
validation techniques (i.e, must validate on unseen test users).

This basic collaborative filtering can be extended later using the tag-genome
features from the MovieLens dataset, as well as movie features obtained via
OMDB. Specifically, collaborative recommendations can be tuned/filtered based
on user-specified desires, such as "more action", or "less romance", or by
specifying that all recommendations satisfy certain criteria like "director=
Quentin Tarantino", or "language=French".
"""

import numpy as np
import pandas as pd

from pyspark import SparkContext
from pyspark.sql import SparkSession
spark_home = '/usr/local/spark'
sc = SparkContext('local', 'film_buff', spark_home)
spark = SparkSession(sc)

from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql.types import *
import pyspark.sql.functions as sqlfun

from scipy.optimize import nnls




class collectionRecommender(object):
    """
        NOTE: Requires an active Spark session named spark to be running.
        
        :param userCol: user column name for Spark ALS. Default 'userId'
        :type userCol: str
        :param itemCol: item column name for Spark ALS. Default 'movieId'
        :type itemCol: str
        :param ratingCol: rating column name for Spark ALS. Default 'rating'
        :type ratingCol: str
        :param update_threshold: determines user or movie profile size beyond
            which online updating is deemed unnecessary
        :type update_threshold: int
        :param ownership_conversion: When users enter new movies from their
            collections, this number determines what ownership is worth on
            a star rating scale. Users can always enter their custom ratings
            via the addRatings feature.
        :type ownership_conversion: float
        :params `**kwargs`: keyword arguments passed to Spark ALS. Note that
            userCol, itemCol, and ratingCol are all set by abovementioned
            parameters, and nonnegative is set to True by default
        """
    def __init__(self, userCol='userId', itemCol='movieId',
                 ratingCol='rating', update_threshold=20, 
                 ownership_conversion=4.0, **kwargs):
        self.als_params = kwargs
        self.userCol=userCol
        self.itemCol=itemCol
        self.ratingCol=ratingCol
        self.ownership_conversion=ownership_conversion
        self.update_threshold = update_threshold 
        self.ratings=None
        self.als = ALS(**self.als_params, userCol=self.userCol, 
                       itemCol=self.itemCol, ratingCol=self.ratingCol,
                       nonnegative=True) # nonnegative set to True by default
        self.userFactors=None
        self.itemFactors=None
        self.predictions = spark\
                           .createDataFrame([], 
                                            schema=StructType([StructField('userId',
                                                                          IntegerType(), 
                                                                          True),
                                                              StructField('movieId', 
                                                                          IntegerType(), 
                                                                          True),
                                                              StructField('predictedRating', 
                                                                          FloatType(),
                                                                          True)]))
        self.predicted_users=[] # modified on prediction
        self.updated_users=[] # modified in online updating
        self.updated_movies=[] # modified in online updating
        
    def fit(self, train=None, reset_predictions=False):
        """
        Fits recommender on provided training set. If no train set is provided,
        will refit on internal "ratings" attribute. If no ratings attribute
        present, will return None.
        
        :param train: Spark DataFrame with same schema as expected by internal
            als parameter
        :param reset_predictions: Determines whether or not list of predictions
            should be reset upon fitting the recommender. If no fit has yet
            been processed, this should be set to False.
        :type reset_predictions: bool
        
        :returns: None
        """
        # Until training data is stored in a SQL or Parquet database,
        # need to keep a copy internally. Can switch to database path later.
        if train:
            self.ratings = train
        else:
            if not self.ratings:
                print('No internal ratings. Need to supply training data.')
                return None
        self.train_users = list(self.ratings\
                           .select(self.userCol)\
                           .distinct()\
                           .toPandas()[self.userCol])
        self.train_movies = list(self.ratings\
                            .select(self.itemCol)\
                            .distinct()\
                            .toPandas()[self.itemCol])
        
        self.updated_users=[]
        self.updated_movies=[]
        if reset_predictions:
            self.predictions = spark\
                               .createDataFrame([], 
                                            schema=StructType([StructField('userId',
                                                                          IntegerType(), 
                                                                          True),
                                                              StructField('movieId', 
                                                                          IntegerType(), 
                                                                          True),
                                                              StructField('predictedRating', 
                                                                          FloatType(),
                                                                          True)]))
            self.predicted_users=[]
        model = self.als.fit(self.ratings)
        self.userFactors = model.userFactors
        self.itemFactors = model.itemFactors
        
    def predict(self, user, keep=1000):
        """
        Will make predictions for user on all unrated movies. Updates internal
        predictions attribute.
        
        :param user: The id of the user to update. Throws error if the model
            has not yet been trained on this user.
        :type user: int
        :param keep: The number of predictions to retain in the predictions
            attribute. If None, will keep all.
        :type keep: int
        
        :returns: None
        """
        if (not self.userFactors) or (not self.itemFactors):
            print('No model trained!')
            return None
        uf = self.userFactors\
                 .filter(self.userFactors.id==user)\
                 .collect()
        if uf:
            uf=uf[0]
        else:
            print('Model has not been trained on user {}.'.format(user))
            return None
        self.predicted_users = list(set(self.predicted_users+[user]))
        already_rated = self.ratings.filter(self.ratings.userId==user)\
                                    .select('movieId')\
                                    .collect()
        already_rated = sorted([row[0] for row in already_rated])
        to_predict = self.itemFactors\
                         .filter(~self.itemFactors.id.isin(already_rated))\
                         .toLocalIterator()
        del already_rated
        # _get_predictions returns a spark DF
        self._update_predictions(self._get_predictions(uf, to_predict, 
                                                       keep=keep))
        
    def _get_predictions(self, uf, to_predict, keep=None):
        """
        Computes predicted ratings given single user factors and item factors.
        
        :param uf: Single user's factors, formatted as a Spark Row object
            with attributes 'id' and 'features'. Extracted from internal
            userFactors DataFrame.
        :param to_predict: An iterator produced by filtering the internal
            itemFactors DataFrame for movies not yet rated by the user given
            in uf.
        
        :returns: a Spark DataFrame with schema: ('userId_new', int),
            ('movieId_new', int), ('predictedRating_new', float).
        """
        preds = []
        userId = uf[0]
        userFactors = np.array(uf[1])
        for movie in to_predict:
            movieId = movie[0]
            movieFactors = np.array(movie[1])
            preds.append((userId,
                          movieId,
                          float(userFactors.dot(movieFactors))))
        
        if keep:
            preds = sorted(preds, key=lambda x: x[2], reverse=True)[:keep]
        
        schema = StructType([StructField('userId_new', IntegerType(),True),
                             StructField('movieId_new', IntegerType(), True),
                             StructField('predictedRating_new', FloatType(),True)])
        return(spark.createDataFrame(preds, schema=schema))
        
    def _update_predictions(self, new_predictions):
        """
        Takes Spark DataFrame of new_predictions and updates the internal
        Spark DataFrame of predictions. Will override existing predictions,
        if they exist.
        
        :param new_predictions: a Spark DataFrame with schema: ('userId_new', int),
            ('movieId_new', int), ('predictedRating_new', float).
        
        :returns: None
        """
        join_condition = [self.predictions.userId==new_predictions.userId_new,
                          self.predictions.movieId==new_predictions.movieId_new]
        self.predictions = self.predictions.join(new_predictions,
                                                 join_condition,
                                                 'outer')
        
        self.predictions = self.predictions\
                               .withColumn('predictedRating_new',
                                           sqlfun.when(self.predictions\
                                                           .predictedRating_new\
                                                           .isNotNull(),
                                                       self.predictions\
                                                           .predictedRating_new)\
                                                   .otherwise(self.predictions\
                                                                  .predictedRating))
        self.predictions = self.predictions\
                               .select(sqlfun.coalesce(self.predictions.userId,
                                                       self.predictions.userId_new)\
                                           .alias('userId'),
                                       sqlfun.coalesce(self.predictions.movieId,
                                                       self.predictions.movieId_new)\
                                           .alias('movieId'),
                                       self.predictions.predictedRating_new\
                                           .alias('predictedRating'))
                                       
    def recommend(self, user, num_recs=20):
        """
        Returns top recommendations for specified user. If user is not found
        in predictions DataFrame, then predict is called to generate predictions.
        
        :param user: userId of user for whom to make recommendations
        :type user: int
        :param num_recs: The number of recommendations to return. Cannot
            exceed total number of predictions made for user in the predictions
            DataFrame. If more recommendations are desired, recall predict
            method on this user and increase the value of keep.
        
        :returns: A list of Spark Row objects of the form (userId, movieId, 
                  predictedRating), sorted in descending order by predictedRating.
        
        """
        if user in self.predicted_users:
            recommendations = self.predictions\
                                  .filter(self.predictions.userId==user)\
                                  .sort(self.predictions.predictedRating.desc())\
                                  .take(num_recs)
            return recommendations
        elif user in self.train_users+self.updated_users:
            self.predict(user)
            recommendations = self.predictions\
                                  .filter(self.predictions.userId==user)\
                                  .sort(self.predictions.predictedRating.desc())\
                                  .take(num_recs)
            return recommendations
        else:
            print('User {} not in ratings.'.format(user))
            return None
            
    def _online_update(self, new_ratings):
        """
        This method should only be called by addRating or removeRating.
        If the profiles of users or movies in new_ratings are still below
        update_threshold, then this function will call updateUser or updateMovie
        as appropriate. Will also update the updated_users and updated_movies
        lists to keep track of who has been modified since last fit.
        
        :param new_ratings: list of tuples of new ratings which have already
            been added to ratings DataFrame.
        
        :returns: None
        """
        users_to_update = list(set([nr[0] for nr in new_ratings]))
        movies_to_update = list(set([nr[1] for nr in new_ratings]))
        
        for user in users_to_update:
            profile_size = self.ratings.filter(self.ratings.userId==user).count()
            if profile_size <= self.update_threshold:
                self.updated_users = list(set(self.updated_users+[user]))
                self._updateUser(user)
        for movie in movies_to_update:
            profile_size = self.ratings.filter(self.ratings.movieId==movie).count()
            if profile_size <= self.update_threshold:
                self.updated_movies = list(set(self.updated_movies+[movie]))
                self._updateMovie(movie)
        
    def addRatings(self, new_ratings):
        """
        Adds new ratings to the ratings DataFrame. Calls _onlineUpdate to
        update latent factors if need be. Will override old ratings if there
        is a conflict.
        
        :param new_ratings: list of tuples of form (userId, movieId, new_rating)
        
        :returns: None
        """
        new_ratings_schema = StructType([StructField('new_userId', IntegerType(),True),
                                         StructField('new_movieId', IntegerType(), True),
                                         StructField('new_rating', FloatType(), True)])
        new_ratings_df = spark.createDataFrame(new_ratings, 
                                               schema=new_ratings_schema)
        self._update_ratings(new_ratings_df)
        self._online_update(new_ratings)
    
    def addToCollection(self, new_movies):
        """
        This should be called when a user adds a movie to his or her collection
        and does not specify a rating. Ownership of a film is converted into
        a star rating via the attribute ownership_conversion. If the user wishes
        to update the rating later, addRatings may be called.
        
        :param new_movies: list of tuples of the form (userId, movieId)
        
        :returns: None
        """
        new_ratings = [t + (self.ownership_conversion,) for t in new_movies]
        self.addRatings(new_ratings)
        
    def removeRatings(self, old_ratings):
        """
        This removes old ratings from the internal ratings DataFrame and calls
        _onlineUpdate to retrain users or movies if necessary.
        
        :param old_ratings: list of tuples of the form (userId, movieId)
        
        :returns: None
        """
        users = [t[0] for t in old_ratings]
        for user in users:
            movies = [t[1] for t in old_ratings if t[0]==user]
            cond = (self.ratings.userId==user)&(self.ratings.movieId.isin(movies))
            self.ratings=self.ratings.filter(~cond)
        self._online_update(old_ratings)
    
    def _update_ratings(self, new_ratings):
        """
        Updates internal ratings DataFrame with new_ratings.
        
        :param new_ratings: Spark DataFrame with schema 
            StructType([StructField('new_userId', IntegerType(),True),
                        StructField('new_movieId', IntegerType(), True),
                        StructField('new_rating', FloatType(), True)])
        :returns: None
        """
        join_condition = [self.ratings.userId==new_ratings.new_userId,
                          self.ratings.movieId==new_ratings.new_movieId]
        self.ratings = self.ratings.join(new_ratings,
                                         join_condition,
                                         'outer')
        
        self.ratings = self.ratings\
                           .withColumn('new_rating',
                                       sqlfun.when(self.ratings\
                                                       .new_rating\
                                                       .isNotNull(),
                                                   self.ratings\
                                                       .new_rating)\
                                              .otherwise(self.ratings\
                                                             .rating))
        self.ratings = self.ratings\
                           .select(sqlfun.coalesce(self.ratings.userId,
                                                   self.ratings.new_userId)\
                                         .alias('userId'),
                                   sqlfun.coalesce(self.ratings.movieId,
                                                   self.ratings.new_movieId)\
                                       .alias('movieId'),
                                   self.ratings.new_rating\
                                       .alias('rating'))
        
    def _updateUser(self,user):
        """
        Performs regularized nonnegative least-squares optimization to update
        latent factors for user.
        
        :param user: userId of user in ratings DataFrame
        :type user: int
        
        :returns: None
        """
        rank = self.als.getRank()
        regParam = self.als.getRegParam()
        # maxIter = self.als.getMaxIter()
        
        user_ratings = self.ratings.filter(self.ratings.userId==user)
        user_ratings = user_ratings.join(self.itemFactors,
                                         user_ratings.movieId==self.itemFactors.id,
                                         how='inner')
        
        b = user_ratings.select('rating').collect()
        b = np.array([r[0] for r in b])
        b = np.concatenate((b, np.zeros(rank)))
        
        A = user_ratings.select('features').collect()
        A = np.array([r[0] for r in A])
        A = np.concatenate((A, np.sqrt(regParam)*np.eye(rank)))
        
        w = nnls(A, b)[0]
        w = [float(f) for f in w]
        
        new_userFactor_schema = StructType([StructField('new_id',
                                                        IntegerType(),
                                                        True),
                                            StructField('new_features', 
                                                        ArrayType(FloatType(),True),
                                                        True)])
        new_userFactors = spark.createDataFrame([(user, w)], 
                                                schema=new_userFactor_schema)
        self._update_userFactors(new_userFactors)
    
    def _update_userFactors(self, new_factors):
        """
        Helper function for updating the userFactors matrix.
        
        :param new_factors: Spark DataFrame with schema
            new_userFactor_schema = StructType([StructField('new_id',
                                                        IntegerType(),
                                                        True),
                                            StructField('new_features', 
                                                        ArrayType(FloatType(),True),
                                                        True)])
        """
        join_condition = self.userFactors.id==new_factors.new_id
        
        self.userFactors = self.userFactors.join(new_factors,
                                                 on=join_condition,
                                                 how='outer')
        self.userFactors = self.userFactors\
                               .withColumn('new_features',
                                           sqlfun.when(self.userFactors\
                                                           .new_features\
                                                           .isNotNull(),
                                                       self.userFactors\
                                                           .new_features)\
                                                  .otherwise(self.userFactors\
                                                                 .features))
        self.userFactors = self.userFactors\
                               .select(sqlfun.coalesce(self.userFactors.id,
                                                       self.userFactors.new_id)\
                                             .alias('id'),
                                       self.userFactors.new_features.alias('features'))
    def _updateMovie(self, movie):
        """
        Performs regularized nonnegative least-squares optimization to update
        latent factors for movie.
        
        :param movie: movieId of movie in ratings DataFrame
        :type movie: int
        
        :returns: None
        """
        rank = self.als.getRank()
        regParam = self.als.getRegParam()
        # maxIter = self.als.getMaxIter()
        
        movie_ratings = self.ratings.filter(self.ratings.movieId==movie)
        movie_ratings = movie_ratings.join(self.userFactors,
                                         movie_ratings.userId==self.userFactors.id,
                                         how='inner')
        
        b = movie_ratings.select('rating').collect()
        b = np.array([r[0] for r in b])
        b = np.concatenate((b, np.zeros(rank)))
        
        A = movie_ratings.select('features').collect()
        A = np.array([r[0] for r in A])
        A = np.concatenate((A, np.sqrt(regParam)*np.eye(rank)))
        
        h = nnls(A, b)[0]
        h = [float(f) for f in h]
        
        new_itemFactor_schema = StructType([StructField('new_id',
                                                        IntegerType(),
                                                        True),
                                            StructField('new_features', 
                                                        ArrayType(FloatType(),True),
                                                        True)])
        new_itemFactors = spark.createDataFrame([(movie, h)], 
                                                schema=new_itemFactor_schema)
        self._update_itemFactors(new_itemFactors)
    def _update_itemFactors(self, new_factors):
        """
        Helper function for updating the itemFactors matrix.
        
        :param new_factors: Spark DataFrame with schema
            new_movieFactor_schema = StructType([StructField('new_id',
                                                        IntegerType(),
                                                        True),
                                            StructField('new_features', 
                                                        ArrayType(FloatType(),True),
                                                        True)])
        """
        join_condition = self.itemFactors.id==new_factors.new_id
        
        self.itemFactors = self.itemFactors.join(new_factors,
                                                 on=join_condition,
                                                 how='outer')
        self.itemFactors = self.itemFactors\
                               .withColumn('new_features',
                                           sqlfun.when(self.itemFactors\
                                                           .new_features\
                                                           .isNotNull(),
                                                       self.itemFactors.\
                                                           new_features)\
                                                  .otherwise(self.itemFactors\
                                                                 .features))
        self.itemFactors = self.itemFactors\
                               .select(sqlfun.coalesce(self.itemFactors.id,
                                                       self.itemFactors.new_id)\
                                             .alias('id'),
                                       self.itemFactors.new_features.alias('features'))
    def evaluate(self):
        pass
    def crossValidate(self):
        pass
    def save(self):
        pass
    def load(self):
        pass
    def updateParams(self):
        pass
    
        

