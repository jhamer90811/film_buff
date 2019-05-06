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

from pyspark import StorageLevel
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql.types import StructType,StructField,ArrayType,\
                            IntegerType,FloatType
import pyspark.sql.functions as sqlfun
from pyspark.sql.functions import col

from recommender_functions import *

import time, os, json
from pathlib import Path



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
        self.ratings = self.ratings.withColumn('full_train', sqlfun.lit(True))
        
    def _predict(self, user):
        """
        Will make predictions for user on all unrated movies. Updates internal
        predictions attribute.
        
        :param user: The id of the user to update. Throws error if the model
            has not yet been trained on this user.
        :type user: int
        
        :returns: None
        """
        user_factors = self.userFactors.filter(col('id')==user)
        user_factors = user_factors.withColumnRenamed('features',
                                                      'userFactors')\
                                   .withColumnRenamed('id',
                                                      'userId')
        
        to_predict = self.ratings.filter(col('userId')==user)\
                                 .select('movieId')
        to_predict = to_predict.join(self.itemFactors,
                                     to_predict.movieId==self.itemFactors.id,
                                     how='outer')\
                               .filter(to_predict.movieId.isNull())\
                               .select(col('id').alias('movieId'),
                                       col('features').alias('itemFactors'))
        to_predict = to_predict.crossJoin(user_factors)\
                               .select('userId', 'movieId',
                                       sqlfun.arrays_zip('userFactors', 
                                                         'itemFactors')\
                                       .alias('paired_factors'))
        to_predict = to_predict.withColumn('predictedRating_new',
                                           dot_product('paired_factors'))
        to_predict = to_predict.select(col('userId').alias('userId_new'),
                                       col('movieId').alias('movieId_new'),
                                       'predictedRating_new')
        
        self.predicted_users = list(set(self.predicted_users + [user]))
        self.predictions = update_predictions(self.predictions, to_predict)
        
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
                                  .sort(col('predictedRating').desc())\
                                  .take(num_recs)
            return recommendations
        elif user in self.train_users+self.updated_users:
            self._predict(user)
            recommendations = self.predictions\
                                  .filter(self.predictions.userId==user)\
                                  .sort(col('predictedRating').desc())\
                                  .take(num_recs)
            return recommendations
        else:
            print('User {} not in ratings.'.format(user))
            return None
                
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
        self.ratings = update_ratings(self.ratings, new_ratings_df)
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
        user_list = list(set([nr[0] for nr in new_ratings]))
        item_list = list(set([nr[1] for nr in new_ratings]))
        
        users_to_update = self.ratings.filter(col('userId').isin(user_list))
        items_to_update = self.ratings.filter(col('movieId').isin(item_list))
        
        user_profiles = users_to_update.groupby('userId').agg({'full_train':'count'})
        user_profiles = user_profiles.select(col('userId').alias('userId2'),
                                             col('count(full_train)').alias('user_profile'))
        
        item_profiles = items_to_update.groupby('movieId').agg({'full_train':'count'})
        item_profiles = item_profiles.select(col('movieId').alias('movieId2'),
                                             col('count(full_train)').alias('item_profile'))
        
        users_to_update = users_to_update.join(user_profiles,
                                               users_to_update.userId==user_profiles.userId2,
                                               how='left')\
                                         .join(item_profiles,
                                               users_to_update.movieId==item_profiles.movieId2,
                                               how='left')
        items_to_update = items_to_update.join(item_profiles,
                                               items_to_update.movieId==item_profiles.movieId2,
                                               how='left')\
                                         .join(user_profiles,
                                               items_to_update.userId==user_profiles.userId2,
                                               how='left')
        users_to_update = users_to_update.filter(col('user_profile')<=self.update_threshold)
        users_to_update = users_to_update.select('userId', 'movieId', 'rating')
        items_to_update = items_to_update.filter(col('item_profile')<=self.update_threshold)
        items_to_update = items_to_update.select('userId', 'movieId', 'rating')
        
        self._updateUsers(users_to_update)
        self._updateItems(items_to_update)
        
    def _updateUsers(self, users_to_update):
        """
        Performs regularized nonnegative least-squares optimization to update
        latent factors for users. If there are movies in the users' profile
        which have only been rated by the user, then a partial ALS will be
        called in order to generate latent factors for these movies.
        
        :param users_to_update: Spark DataFrame with schema
            StructType([StructField('userId', IntegerType(),True),
                        StructField('movieId', IntegerType(), True),
                        StructField('rating', FloatType(), True)])
        
        :returns: None
        """
        rank = self.als.getRank()
        maxIter = self.als.getMaxIter()
        regParam = self.als.getRegParam()
        
        users_to_update = users_to_update.join(self.itemFactors,
                                               users_to_update.movieId==self.itemFactors.id,
                                               'left')
        users_to_update = users_to_update.select('userId',
                                                 sqlfun.struct('movieId', 'features')\
                                                       .alias('itemFactors'),
                                                 'rating')
        users_to_update = users_to_update.groupby('userId')\
                                         .agg(sqlfun.collect_list('itemFactors')\
                                                  .alias('itemFactors'),
                                              sqlfun.collect_list('rating')\
                                                  .alias('rating'))
        
        users_to_update = users_to_update.withColumn('new_latent_factors',
                                                     new_lf('itemFactors',
                                                            'rating',
                                                            sqlfun.lit(rank),
                                                            sqlfun.lit(maxIter),
                                                            sqlfun.lit(regParam)))
        users_to_update = users_to_update.withColumn('new_userFactors',
                                                     extract_first('new_latent_factors'))
        users_to_update = users_to_update.withColumn('new_itemFactors',
                                                     extract_second('new_latent_factors'))
        
        items_to_update = users_to_update.select('new_itemFactors')\
                                         .filter(col('new_itemFactors').isNotNull())
        
        users_to_update = users_to_update.select(col('userId').alias('new_id'), 
                                                 col('new_userFactors').alias('new_features'))
        
        items_to_update = items_to_update.select(sqlfun.explode('new_itemFactors')\
                                                       .alias('new_itemFactors'))
        items_to_update = items_to_update.select(extract_id('new_itemFactors')\
                                                         .alias('new_id'),
                                                 extract_features('new_itemFactors')\
                                                         .alias('new_features'))
        items_to_update = items_to_update.groupby('new_id')\
                                         .agg(sqlfun.collect_list('new_features')\
                                                    .alias('new_features_list'))
        items_to_update = items_to_update.select('new_id',
                                                 get_avg('new_features_list')\
                                                     .alias('new_features'))
        
        self.userFactors = update_latentFactors(self.userFactors, users_to_update)
        self.itemFactors = update_latentFactors(self.itemFactors, items_to_update)
        
        updated_users = [x[0] for x in users_to_update.select('new_id').collect()]
        updated_items = [x[0] for x in items_to_update.select('new_id').collect()]
        
        self.updated_users = list(set(self.updated_users + updated_users))
        self.updated_movies = list(set(self.updated_movies + updated_items))
    
    def _updateItems(self, items_to_update):
        """
        Performs regularized nonnegative least-squares optimization to update
        latent factors for multiple items.
        
        :param items_to_update: Spark DataFrame with schema 
            StructType([StructField('userId', IntegerType(),True),
                        StructField('movieId', IntegerType(), True),
                        StructField('rating', FloatType(), True)])
        
        :returns: None
        """
        rank = self.als.getRank()
        maxIter = self.als.getMaxIter()
        regParam = self.als.getRegParam()
        
        items_to_update = items_to_update.join(self.userFactors,
                                               items_to_update.userId==self.userFactors.id,
                                               'left')
        items_to_update = items_to_update.select('movieId',
                                                 sqlfun.struct('userId', 'features')\
                                                       .alias('userFactors'),
                                                 'rating')
        items_to_update = items_to_update.groupby('movieId')\
                                         .agg(sqlfun.collect_list('userFactors')\
                                                  .alias('userFactors'),
                                              sqlfun.collect_list('rating')\
                                                  .alias('rating'))
        
        items_to_update = items_to_update.withColumn('new_latent_factors',
                                                     new_lf('userFactors',
                                                            'rating',
                                                            sqlfun.lit(rank),
                                                            sqlfun.lit(maxIter),
                                                            sqlfun.lit(regParam)))
        items_to_update = items_to_update.withColumn('new_itemFactors',
                                                     extract_first('new_latent_factors'))
        
        items_to_update = items_to_update.select(col('movieId').alias('new_id'), 
                                                 col('new_itemFactors').alias('new_features'))
        self.itemFactors = update_latentFactors(self.itemFactors, items_to_update)
        
        updated_items = [x[0] for x in items_to_update.select('new_id').collect()]
        
        self.updated_movies = list(set(self.updated_movies + updated_items))
        
    def evaluate_withRatings(self, test, itemFactors, alpha=0.9):
        """
        Evaluates model on previously unseen users. Users are assumed to have
        provided ratings for each movie.
        
        Evaluation is performed by splitting each user profile into a train
        set and a validation set. The train set is a random selection of
        half of the user's profile, or update_threshold ratings from the user's
        profile, whichever is smaller.
        
        On each iteration, all users are trained on a random rating from their
        training sets (while they yet have ratings in their training set). They
        are then scored on their validation sets and the RMSE is computed over
        all users who were updated this iteration. This yields s sequence of
        scores, RMSE_j, for j=1, ..., update_threshold.
        
        After all iterations are completed, the elbow index is computed. If
        M = update_threshold, then the elbow index is defined as the minimal
        index j such that (RMSE_1-RMSE_j)>alpha*(RMSE_1-RMSE_M) and RMSE_k < RMSE_j
        for all k > j. Informally, this is the number of training samples
        required to account for at least alpha*100% of the reduction in error due to
        a larger user profile. A smaller elbow index is generally better.
        
        Finally, an adjusted RMSE is computed as\n
        RMSE_adj = (1 + (j_e-1)/M)*RMSE_M,\n
        where j_e is the elbow index.
        
        :param test: A DataFrame of users on which the model has not yet trained.
        
            :type test: Spark DataFrame with schema=StructType([StructField('userId', IntegerType(),True),
                                         StructField('movieId', IntegerType(), True),
                                         StructField('rating', FloatType(), True)])
        :param itemFactors: A DataFrame of latent item factors.
            :type itemFactors: Spark DataFrame with schema=StructType([StructField('new_id',
                                                        IntegerType(),
                                                        True),
                                            StructField('new_features', 
                                                        ArrayType(FloatType(),True),
                                                        True)])
        :param alpha: Determines percent of error that must be accounted for
        in order to define the elbow index. Should be between 0 and 1, non-
        inclusive. A larger alpha will produce a larger elbow index.
            :type alpha: float
            
        :returns: A tuple of the form ([list of RMSE_j], elbow index, RMSE_adj)
        """
        # First set up training and validation sets
#        print('Initial number of persistent RDDs: {}'\
#              .format(len(spark.sparkContext._jsc.getPersistentRDDs())))
        M = self.update_threshold
        
        test = test.withColumn('movie_rating', sqlfun.struct('userId', 
                                                             'movieId', 
                                                             'rating'))
        test = test.select('userId', 'movie_rating')
        test = test.groupby('userId').agg(sqlfun.collect_list('movie_rating')\
                                                .alias('movie_rating_list'))
        all_users = test.select('userId')
        test = test.withColumn('movie_rating_list', 
                               sqlfun.shuffle(col('movie_rating_list')))
        test = test.withColumn('profile_size', sqlfun.size('movie_rating_list'))
        
        # Will not evaluate on users half of whose profile is obscure (i.e.
        # movies on which the model has not been trained).
        # This step also gaurantees that the validation sets will be at least
        # 5 items, or 1/8 the total profile, whichever is smaller.
        test = test.withColumn('obscurity', get_obscurity('movie_rating_list',
                                                          sqlfun.array([sqlfun.lit(x)\
                                                                        for x in\
                                                                        self.train_movies]),
                                                          sqlfun.lit(M)))
        test = test.dropna(subset='obscurity').drop('obscurity')
        obscure_users = all_users.exceptAll(test.select('userId'))
        
        test = test.withColumn('train_size', sqlfun.when(sqlfun.ceil(col('profile_size')/2)<M,
                                                         sqlfun.ceil(col('profile_size')/2))\
                                                    .otherwise(M))
        test = test.withColumn('validation_set',
                               extract_slice('movie_rating_list',
                                             'train_size',
                                             sqlfun.lit(None)))
        test = test.withColumn('validation_set',
                               get_val_set('validation_set',
                                            sqlfun.array([sqlfun.lit(x)\
                                                          for x in\
                                                          self.train_movies])))
        test = test.withColumn('validation_size', 
                               sqlfun.size('validation_set'))\
                   .cache().localCheckpoint()
        test.count()
        
        # Extract and cache DFs for training and validation itemFactors
        
        val_items = test.select(sqlfun.explode('validation_set').alias('val_sample'))
        val_items = val_items.select(col('val_sample')\
                                     .getField('movieId').alias('movieId'))\
                             .distinct()
        val_itemFactors = val_items.join(itemFactors,
                                         val_items.movieId==itemFactors.id,
                                         how='left')\
                                   .select('id', 'features')\
                                   .cache().localCheckpoint()
        val_itemFactors.count()
#        print('Caching validation itemFactors...')
#        print('Number of validation items: {}'.format(val_itemFactors.count()))
        
        train_items = test.select(extract_slice('movie_rating_list',
                                                sqlfun.lit(0),
                                                'train_size')\
                                  .alias('train_set'))
        train_items = train_items.select(sqlfun.explode('train_set')\
                                         .alias('train_sample'))
        train_items = train_items.select(col('train_sample')\
                                         .getField('movieId')\
                                         .alias('movieId'))
        train_itemFactors = train_items.join(itemFactors,
                                             train_items.movieId==itemFactors.id,
                                             how='inner')\
                                       .select('id', 'features')\
                                       .cache().localCheckpoint()
        train_itemFactors.count()
#        print('Caching training itemFactors...')
#        print('Number of training items: {}'.format(train_itemFactors.count()))
        
        latentFactor_schema = StructType([StructField('id', IntegerType()),
                                          StructField('features', 
                                                        ArrayType(FloatType()))])
        userFactors = spark.createDataFrame([], schema=latentFactor_schema)
        
        scores = test.select('userId', 'validation_size')
        
        rmse_list = []
        
#        print('***********************')
#        print('BEGINNING EVAL ITERATIONS')
#        print('***********************')
        for j in range(M):
            
            iteration_time = time.time()
#            print('Validation Iteration {}/{}'.format(j+1, M))
            new_ratings = test.filter(j<col('train_size'))
            if not new_ratings.take(1):
                M = j+1
                print('No training samples left, stopping early at iteration {}'\
                      .format(M))
                break
            validation_users = new_ratings.select('validation_set')
            
            train_ratings = new_ratings.select(extract_slice('movie_rating_list',
                                                             sqlfun.lit(0),
                                                             sqlfun.lit(j+1))\
                                                .alias('train_set'))
            train_ratings = train_ratings.select(sqlfun.explode('train_set')\
                                                 .alias('train_sample'))
            train_ratings = train_ratings.select(col('train_sample')\
                                                 .getField('userId')\
                                                 .alias('userId'),
                                                 col('train_sample')\
                                                 .getField('movieId')\
                                                 .alias('movieId'),
                                                 col('train_sample')\
                                                 .getField('rating')\
                                                 .alias('rating'))
            
            # Persistence of previous DataFrames survives the name change
            # This allows us to unpersist immediately after we're done with them
            old_userFactors = userFactors
            old_train_itemFactors = train_itemFactors
            
            userFactors, train_itemFactors =self._eval_updateUsers(train_ratings,
                                                             old_userFactors,
                                                             old_train_itemFactors)
            new_scores = self._eval_get_scores(scores, validation_users,
                                               userFactors, val_itemFactors)
            
            
            rmse_time = time.time()
            rmse_j = new_scores.filter(col('sse').isNotNull())\
                           .select(get_sqrt(sqlfun.sum('sse')/\
                                            sqlfun.sum('validation_size')))\
                           .collect()[0][0]
#            print('RMSE_{} = {}; update time: {}'\
#                  .format(j+1, round(rmse_j,2), round(time.time()-rmse_time,2)))
            rmse_list.append(rmse_j)
            
#            print('Total time spent on iteration {}: {}'\
#                  .format(j+1, round(time.time()-iteration_time,2)))
#            print('*******************************\n')
            
        
        elbow_index = self._eval_get_elbow_index(rmse_list, alpha)
        rmse_adj = self._eval_get_rmse_adj(rmse_list, elbow_index)
        
#        print("The following users' profiles were too obscure for eval:")
#        obscure_users.show(truncate=100)
        
#        print('Unpersisting userFactors...')
        userFactors.unpersist()
#        print('Unpersisting training itemFactors...')
        train_itemFactors.unpersist()
#        print('Unpersisting validation itemFactors...')
        val_itemFactors.unpersist()
#        print('Unpersisting modified test set...')
        test.unpersist()
#        spark.catalogue.clearCache()
        
#        print('Final number of persistent RDDs: {}'\
#              .format(len(spark.sparkContext._jsc.getPersistentRDDs())))
        
        return rmse_list, elbow_index, rmse_adj
        
    def evaluate_withoutRatings(self, test, itemFactors, alpha=0.9):
        """
        Evaluates on new users who do not supply ratings with their movies.
        The test set is given a 'rating' column consisting of the 
        ownership_conversion attribute of the collectionRecommender object.
        This augmented test set is then sent to evaluate_withRatings.
        """
        test = test.withColumn('rating', sqlfun.lit(self.ownership_conversion))
        return self.eval_withRatings(test, itemFactors, alpha)
    
    def _eval_updateUsers(self, users_to_update, userFactors, itemFactors):
        """
        Helper function for evaluation. Updates the latent factors for user.
        If user has rated movies which have been rated by nobody else, will
        instead call partial_als to update user's latent factors as well as
        item factors for these new movies.
        
        :returns: updated_userFactors, updated_itemFactors
        """
        rank = self.als.getRank()
        maxIter = self.als.getMaxIter()
        regParam = self.als.getRegParam()
        
        users_to_update = users_to_update.join(itemFactors,
                                               users_to_update.movieId==itemFactors.id,
                                               'left')
        users_to_update = users_to_update.select('userId',
                                                 sqlfun.struct('movieId', 'features')\
                                                       .alias('itemFactors'),
                                                 'rating')
        users_to_update = users_to_update.groupby('userId')\
                                         .agg(sqlfun.collect_list('itemFactors')\
                                                  .alias('itemFactors'),
                                              sqlfun.collect_list('rating')\
                                                  .alias('rating'))
        
        users_to_update = users_to_update.withColumn('new_latent_factors',
                                                     new_lf('itemFactors',
                                                            'rating',
                                                            sqlfun.lit(rank),
                                                            sqlfun.lit(maxIter),
                                                            sqlfun.lit(regParam)))
        users_to_update = users_to_update.withColumn('new_userFactors',
                                                     col('new_latent_factors')\
                                                     .getField('first_features'))
        users_to_update = users_to_update.withColumn('new_itemFactors',
                                                     col('new_latent_factors')\
                                                     .getField('second_features'))
        
        items_to_update = users_to_update.select('new_itemFactors')\
                                         .filter(col('new_itemFactors').isNotNull())
        
        users_to_update = users_to_update.select(col('userId').alias('new_id'), 
                                                 col('new_userFactors').alias('new_features'))
        
        items_to_update = items_to_update.select(sqlfun.explode('new_itemFactors')\
                                                       .alias('new_itemFactors'))
        items_to_update = items_to_update.select(col('new_itemFactors').getField('id')\
                                                         .alias('new_id'),
                                                 col('new_itemFactors').getField('features')\
                                                         .alias('new_features'))
        items_to_update = items_to_update.groupby('new_id')\
                                         .agg(sqlfun.collect_list('new_features')\
                                                    .alias('new_features_list'))
        items_to_update = items_to_update.select('new_id',
                                                 get_avg('new_features_list')\
                                                     .alias('new_features'))
        
        new_userFactors = update_latentFactors(userFactors, users_to_update)
        timestamp = time.time()
#        print('Caching new userFactors DF and unpersisting old...')
        new_userFactors = new_userFactors.cache().localCheckpoint()
        new_userFactors.count()
        userFactors.unpersist()
#        print('Time: {}'.format(round(time.time()-timestamp, 2)))
#        print('Number of persistent RDDs: {}'\
#              .format(len(spark.sparkContext._jsc.getPersistentRDDs())))
        
        new_itemFactors = update_latentFactors(itemFactors, items_to_update)
        timestamp = time.time()
#        print('Caching new train_itemFactors DF and unpersisting old...')
        new_itemFactors = new_itemFactors.cache().localCheckpoint()
        new_itemFactors.count()
#        print('Count of new train_itemFactors: ')
#        print(new_itemFactors.count())
        itemFactors.unpersist()
#        print('Time: {}'.format(round(time.time()-timestamp, 2)))
#        print('Number of persistent RDDs: {}'\
#              .format(len(spark.sparkContext._jsc.getPersistentRDDs())))
        
        return new_userFactors, new_itemFactors
    
    def _eval_get_scores(self, scores, validation_sets, userFactors,
                         itemFactors):
        """
        Helper function for evaluation. 
        Computes SSE for each user in validation_sets and appends as a new
        column to scores.
        
        :returns: updated_scores
        """
        scoring_df = validation_sets.select(sqlfun.explode('validation_set')\
                                                  .alias('val_set'))
        scoring_df = scoring_df.select(col('val_set').getField('userId').alias('userId2'),
                                       col('val_set').getField('movieId').alias('movieId'),
                                       col('val_set').getField('rating').alias('rating'))
        
        scoring_df = scoring_df.join(userFactors,
                                     scoring_df.userId2==userFactors.id,
                                     how='left')\
                               .select('userId2',
                                       'movieId',
                                       'rating',
                                       col('features').alias('userFactors'))
        scoring_df = scoring_df.join(itemFactors,
                                     scoring_df.movieId==itemFactors.id,
                                     how='left')\
                               .select('userId2',
                                       'rating',
                                       'userFactors',
                                       col('features').alias('itemFactors'))
        scoring_df = scoring_df.select('userId2',
                                       'rating',
                                       sqlfun.arrays_zip('userFactors',
                                                         'itemFactors')\
                                              .alias('paired_factors'))
        scoring_df = scoring_df.select('userId2',
                                       'rating',
                                       dot_product('paired_factors')\
                                       .alias('pred'))
        scoring_df = scoring_df.select('userId2',
                                       ((col('rating')-col('pred'))**2)\
                                       .alias('se'))
        scoring_df = scoring_df.groupby('userId2').agg(sqlfun.sum(col('se'))\
                                                       .alias('sse'))
        updated_scores = scores.join(scoring_df,
                                     scores.userId==scoring_df.userId2,
                                     how='left')
        updated_scores = updated_scores.drop('userId2')
        
        return updated_scores
    
    def _eval_get_elbow_index(self, rmse_list, alpha):
        """
        Helper function for evaluation. Computes elbow index given list of
        rmse and alpha.
        
        :returns: integer elbow index
        """
        M = len(rmse_list)
        first = rmse_list[0]
        last = rmse_list[-1]
        if last > first:
            return M
        threshold = alpha*(first-last)
        for j in range(M):
            current = rmse_list[j]
            if current<max(rmse_list[j:]):
                continue
            else:
                if (first-current)>=threshold:
                    return j+1
        return M
        
    def _eval_get_rmse_adj(self, rmse_list, elbow_index):
        """
        Helper function for evaluation. Computes adjusted rmse.
        
        :returns: adjusted rmse as float.
        """
        return (1+(elbow_index-1)/len(rmse_list))*rmse_list[-1]
    
    def save(self, save_dir, refit = False, save_ratings = False,
             save_preds = False):
        """
        Saves a collectionRecommender object to a user-specified directory. Any
        Spark DataFrames are serialized as JSON objects, and all other model
        attributes are collected into a Python dict and serialized as a JSON
        called metadata. The data is saved in the directory with the following
        names:
        
        collectionRecommender.userFactors --> userFactors
        collectionRecommender.itemFactors --> itemFactors
        collectionRecommender.ratings --> ratings
        collectionRecommender.predictions --> predictions
        All other attributes --> metadata
        
        :param save_dir: A string specifying the name of the directory to which 
            model will be saved.
        :param refit: A boolean specifying whether or not to refit the model
            on internal ratings DataFrame prior to saving.
        :param save_ratings: A boolean specifying whether or not to save the
            internal ratings DataFrame. WARNING: if this option is not set to false
            and the ratings data is not saved elsewhere, then future refitting of
            the model will be impossible.
        :param save_preds: A boolean specifying whether or not to save the
            predictions DataFrame.
        """
        if not os.path.exists(save_dir):
            print('Specified directory does not exist.')
            return None
        if refit:
            print('Refitting on full training data...')
            self.fit()
        print('Saving latent user factors...')
        self.userFactors.write.json(os.path.join(save_dir, 'userFactors'))
        print('Saving latent item factors...')
        self.itemFactors.write.json(os.path.join(save_dir, 'itemFactors'))
        if save_ratings:
            print('Saving ratings matrix...')
            self.ratings.write.json(os.path.join(save_dir, 'ratings'))
        if save_preds:
            print('Saving predictions...')
            self.predictions.write.json(os.path.join(save_dir, 'predictions'))
        print('Saving metadata...')
        metadata = {}
        metadata['als_params'] = self.als_params
        metadata['userCol'] = self.userCol
        metadata['itemCol'] = self.itemCol
        metadata['ratingCol'] = self.ratingCol
        metadata['ownership_conversion'] = self.ownership_conversion
        metadata['update_threshold'] = self.update_threshold
        metadata['predicted_users'] = self.predicted_users
        metadata['updated_users'] = self.updated_users
        metadata['updated_movies'] = self.updated_movies
        metadata['train_users'] = self.train_users
        metadata['train_movies'] = self.train_movies
        if not os.path.exists(os.path.join(save_dir, 'metadata')):
            Path(os.path.join(save_dir, 'metadata')).touch()
        json.dump(metadata, open(os.path.join(save_dir, 'metadata'), 'w'))
        
    def load(self, load_dir):
        """
        Loads collectionRecommender model from directory specified by load_dir.
        Note that the contents of load_dir must have the names and formats
        specified by the collectionRecommender.save method (see that
        documentation for further information).
        
        :param load_dir: A string specifying the name of directory from which 
            to load the collectionRecommender object
        """
        if not os.path.exists(load_dir):
            print('Specified load directory does not exist.')
            return None
        print('Loading metadata...')
        metadata = json.load(open(os.path.join(load_dir, 'metadata'), 'r'))
        meta_keys = list(metadata.keys())
        if 'als_params' in meta_keys:
            self.als_params = metadata['als_params']
        if 'userCol' in meta_keys:
            self.userCol = metadata['userCol']
        if 'itemCol' in meta_keys:
            self.itemCol = metadata['itemCol']
        if 'ratingCol' in meta_keys:
            self.ratingCol = metadata['ratingCol']
        if 'ownership_conversion' in meta_keys:
            self.ownership_conversion = metadata['ownership_conversion']
        if 'update_threshold' in meta_keys:
            self.update_threshold = metadata['update_threshold']
        if 'predicted_users' in meta_keys:
            self.predicted_users = metadata['predicted_users']
        if 'updated_users' in meta_keys:
            self.updated_users = metadata['updated_users']
        if 'updated_movies' in meta_keys:
            self.updated_movies = metadata['updated_movies']
        if 'train_users' in meta_keys:
            self.train_users = metadata['train_users']
        if 'train_movies' in meta_keys:
            self.train_movies = metadata['train_movies']
        self.als = ALS(**self.als_params, userCol=self.userCol, 
                       itemCol=self.itemCol, ratingCol=self.ratingCol,
                       nonnegative=True)
        
        latentFactor_schema = StructType([StructField('id', IntegerType()),
                                          StructField('features', 
                                                        ArrayType(FloatType()))])
        print('Loading latent factors...')
        self.userFactors = spark.read.json(os.path.join(load_dir, 'userFactors'),
                                           schema=latentFactor_schema)
        self.itemFactors = spark.read.json(os.path.join(load_dir, 'itemFactors'),
                                           schema=latentFactor_schema)
        
        
    
        if os.path.exists(os.path.join(load_dir, 'ratings')):
            print('Loading ratings...')
            ratings_schema = StructType([StructField('userId',IntegerType(), True),
                                         StructField('movieId',IntegerType(), True),
                                         StructField('rating',FloatType(), True)])
            self.ratings = spark.read.json(os.path.join(load_dir, 'ratings'),
                                           schema=ratings_schema)
        
        
        if os.path.exists(os.path.join(load_dir, 'predictions')):
            print('Loading predictions...')
            preds_schema = StructType([StructField('userId',IntegerType(), True),
                                       StructField('movieId',IntegerType(), True),
                                       StructField('predictedRating',FloatType(), True)])
            self.predictions = spark.read.json(os.path.join(load_dir, 'predictions'),
                                               schema=preds_schema)
#    def updateParams(self):
#        pass
 
def recommender_crossValidate(ratings, param_grids, num_folds=3, seed=9,
                              max_training_samples = 20):
    """
    This method performs grid-search cross-validation for hyperparameter
    tuning. Any hyperparameters to be passed to Spark's ALS model may be
    tuned. CV splits are done on users, so that evaluation is done on users
    without any pre-exising latent factors. This evaluation method is consistent
    with the collectionRecommender's built-in evaluation methods.
    
    :param ratings: Spark DataFrame of schema userId/movieId/rating to be split
    and used for CV.
    :param param_grids: List of Python dicts. The keys of each dict should be
    names of ALS model hyperparameters, and the values should be lists of 
    appropriate values.
    :param num_folds: An integer specifying the number of folds for CV.
    :param seed: Currently not implemented. In future will be used to set the
    seed for cross validation shuffling.
    :param max_training_samples: An integer specifying the maximal number of
    training samples to use for evaluation.
    
    :returns: List of Pandas DataFrames. Each DataFrame corresponds to a 
    parameter dict passed via param_grids. The rows of each DataFrame correspond
    to a specific combination of parameters and CV folds. The sequence of RMSE_j
    is reported for each parameter/fold combination.
    """
    users = [u[0] for u in ratings.select('userId').distinct().collect()]
    users = list(int(x) for x in np.random.permutation(users))
    n_users = len(users)
    print('Total users: {}'.format(n_users))
    fold_size = int(n_users/num_folds)
    print('Fold size: {}'.format(fold_size))
    
    results_list = []
    
    rmse_names = ['rmse{}'.format(i) for i in range(max_training_samples)]
    
    overall_time = time.time()
    for pg in param_grids:
        param_names = list(pg.keys())
        result_columns = param_names + ['fold'] + rmse_names
        result_df = pd.DataFrame(columns=result_columns)
        
        for p in product_dict(**pg):
            param_timestamp = time.time()
            print('Current Params:')
            print(p)
            cr = collectionRecommender(**p)
            for i in range(num_folds):
                fold_timestamp = time.time()
                print('Fold {}...'.format(i))
                fold_dict = {'fold': i}
                test_users = users[i*fold_size:(i+1)*fold_size]
                train_users = list(set(users).difference(set(test_users)))
                test = ratings.filter(col('userId').isin(test_users))
                train = ratings.filter(col('userId').isin(train_users))
                print('Training model...')
                cr.fit(train)
                print('Evaluating model...')
                scores = cr.evaluate_withRatings(test, cr.itemFactors)[0]
                scores = dict(zip(rmse_names, scores))
                p.update(fold_dict)
                p.update(scores)
                result_df = result_df.append(p, ignore_index=True)
                spark.catalog.clearCache()
                print('Fold finished. Time: {}'.format(round(time.time()-fold_timestamp,2)))
                print(result_df.to_string())
            del cr
            print('Params finished. Time: {}'.format(round(time.time()-param_timestamp,2)))
        results_list.append(result_df)
    print('Total time: {}'.format(time.time()-overall_time, 2))
    return results_list