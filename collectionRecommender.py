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
from pyspark.sql.types import StructType,StructField,ArrayType,\
                            IntegerType,FloatType
import pyspark.sql.functions as sqlfun
from pyspark.sql.functions import col

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
                self._updateUser(user, profile_size)
        for movie in movies_to_update:
            profile_size = self.ratings.filter(self.ratings.movieId==movie).count()
            if profile_size <= self.update_threshold:
                self._updateMovie(movie)
        
    def _updateUser(self,user, profile_size):
        """
        Performs regularized nonnegative least-squares optimization to update
        latent factors for user. If there are movies in the user's profile
        which have only been rated by the user, then a partial ALS will be
        called in order to generate latent factors for these movies.
        
        :param user: userId of user in ratings DataFrame
        :type user: int
        
        :param profile_size: the total number of ratings from the user
        :type profile_size: int
        
        :returns: None
        """
        rank = self.als.getRank()
        regParam = self.als.getRegParam()
        # maxIter = self.als.getMaxIter()
        
        user_ratings = self.ratings.filter(self.ratings.userId==user)
        user_ratings = user_ratings.join(self.itemFactors,
                                         user_ratings.movieId==self.itemFactors.id,
                                         how='inner')
        if profile_size > user_ratings.count():
            self._partial_als(user)
            return None
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
        self.updated_users = list(set(self.updated_users+[user]))
    
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
        self.updated_movies = list(set(self.updated_movies+[movie]))
        
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
                                           
    def _partial_als(self, user):
        """
        Performs ALS on the subset of ratings with userId user. Latent factors
        for films which have been reviewed by other users will not be updated,
        but latent factors for films only reviewed by user will be updated.
        
        :param user: userId of user on whom to perform partial ALS.
        :type user: int
        
        :returns: None
        """
        rank = self.als.getRank()
        maxIter = self.als.getMaxIter()
        regParam = self.als.getRegParam()
        
        user_profile = self.ratings.filter(self.ratings.userId==user)
        user_profile = user_profile.join(self.itemFactors,
                                         user_profile.movieId==self.itemFactors.id,
                                         'left')
        items_to_update = user_profile.filter(col('features').isNull())\
                                      .select('movieId')\
                                      .collect()
        items_to_update = [r[0] for r in items_to_update]
        
        # initialize missing item features
        user_profile = user_profile.withColumn('features',
                                               sqlfun.when(col('features').isNull(),
                                                           sqlfun.array_repeat(\
                                                               sqlfun.sqrt(col('rating'))\
                                                               +0.1*sqlfun.randn(),
                                                               rank))\
                                                     .otherwise(col('features')))
        
        factors_ratings = user_profile.select('movieId', 'features', 'rating')\
                                      .toPandas()
        
        # begin ALS
        for j in range(maxIter):
            H = np.array([features for features in factors_ratings.features.values])
            H = np.concatenate((H, np.sqrt(regParam)*np.eye(rank)))
            
            b = factors_ratings.rating.values
            b = np.concatenate((b, np.zeros(rank)))
            
            w = nnls(H, b)[0] # update latent user factors
            
            W = np.array([w])
            W = np.concatenate((W, np.sqrt(regParam)*np.eye(rank)))
            
            for m in items_to_update:
                b = factors_ratings[factors_ratings.movieId==m].rating.values
                b = np.concatenate((b, np.zeros(rank)))
                h = nnls(W, b)[0]
                h = [float(f) for f in h]
                idx = factors_ratings[factors_ratings.movieId==m].index[0]
                factors_ratings.features[idx] = h
        
        # Update internal latent factors
        new_latentFactor_schema = StructType([StructField('new_id',
                                                        IntegerType(),
                                                        True),
                                            StructField('new_features', 
                                                        ArrayType(FloatType(),True),
                                                        True)])
        new_userFactors = [(user,[float(f) for f in w])]
        new_userFactors = spark.createDataFrame(new_userFactors, 
                                                schema=new_latentFactor_schema)
        self._update_userFactors(new_userFactors)
        
        self.updated_users = list(set(self.updated_users + [user]))
        new_itemFactors = factors_ratings[['movieId', 'features']]
        new_itemFactors = [tuple(row) for _,row in new_itemFactors.iterrows()]
        new_itemFactors = [(int(mid), list(features)) for mid,features in new_itemFactors]
        new_itemFactors = spark.createDataFrame(new_itemFactors, 
                                                schema=new_latentFactor_schema)
        self._update_itemFactors(new_itemFactors)
        self.updated_movies = list(set(self.updated_movies + items_to_update))
        
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
        M = self.update_threshold
        
        test = test.withColumn('movie_rating', sqlfun.struct('userId', 
                                                             'movieId', 
                                                             'rating'))
        test = test.select('userId', 'movie_rating')
        test = test.groupby('userId').agg(sqlfun.collect_list('movie_rating')\
                                                .alias('movie_rating_list'))
        test = test.withColumn('movie_rating_list', 
                               sqlfun.shuffle(col('movie_rating_list')))
        test = test.withColumn('profile_size', sqlfun.size('movie_rating_list'))
        test = test.withColumn('train_size', sqlfun.when(sqlfun.ceil(col('profile_size')/2)<M,
                                                         sqlfun.ceil(col('profile_size')/2))\
                                                    .otherwise(M))
        test = test.withColumn('validation_size', 
                               col('profile_size')-col('train_size'))
        # Need functions to extract movie/rating tuples from test.
        @sqlfun.udf(returnType=StructType([StructField('userId', IntegerType()),
                                           StructField('movieId', IntegerType()),
                                           StructField('rating', FloatType())]))
        def extract_one(a, j):
            # remember to pass j with sqlfun.lit(j)
            return a[j]
        @sqlfun.udf(returnType=ArrayType(StructType([StructField('userId', 
                                                                 IntegerType()),
                                           StructField('movieId', IntegerType()),
                                           StructField('rating', FloatType())])))
        def extract_slice(a, start, stop):
            return a[start:stop]
        
        latentFactor_schema = StructType([StructField('id', IntegerType()),
                                          StructField('features', 
                                                        ArrayType(FloatType()))])
        userFactors = spark.createDataFrame([], schema=latentFactor_schema)
        
        ratings_schema = StructType([StructField('userId', IntegerType()),
                                     StructField('movieId', IntegerType()),
                                     StructField('rating', FloatType())])
        ratings = spark.createDataFrame([], schema=ratings_schema)
        
        rmse_list = []
        
        for j in range(M):
            new_ratings = test.filter(j<col('train_size'))
            if new_ratings.count()==0:
                break
            validation_sets = new_ratings.select('userId',
                                                 extract_slice('movie_rating_list',
                                                               'train_size',
                                                               sqlfun.lit(None))\
                                                  .alias('validation_set'),
                                                 'validation_size')
            new_ratings = new_ratings.select(extract_one('movie_rating_list',
                                                         sqlfun.lit(j))\
                                             .alias('train_set'))
            new_ratings = [tuple(r[0]) for r in new_ratings.toLocalIterator()]
            ratings, userFactors, itemFactors =\
                                        self._eval_addRatings(ratings,
                                                              new_ratings,
                                                              userFactors,
                                                              itemFactors)
            total_val_size = 0
            sse_list = []
            for val_set in validation_sets.toLocalIterator():
                user = val_set[0]
                val_items = [r[1] for r in val_set[1]]
                val_ratings = [r[2] for r in val_set[1]]
                val_size = val_set[2]
                total_val_size+=val_size
                
                preds = self._eval_predict(user, val_items,
                                               userFactors, itemFactors)
                sse = np.sum((np.array(preds)-np.array(val_ratings))**2)
                sse_list.append(sse)
            sse_list = np.array(sse_list)
            rmse_list.append(np.sqrt(sse_list.sum()/total_val_size))
            print('Iteration {} done. RMSE is {}'.format(j, rmse_list[-1]))
        
        elbow_index = self._eval_get_elbow_index(rmse_list, alpha)
        rmse_adj = self._eval_get_rmse_adj(rmse_list, elbow_index)
        
        return rmse_list, elbow_index, rmse_adj
        
    def eval_withoutRatings(self, test, itemFactors, alpha=0.9):
        """
        Evaluates on new users who do not supply ratings with their movies.
        The test set is given a 'rating' column consisting of the 
        ownership_conversion attribute of the collectionRecommender object.
        This augmented test set is then sent to evaluate_withRatings.
        """
        test = test.withColumn('rating', sqlfun.lit(self.ownership_conversion))
        return self.eval_withRatings(test, itemFactors, alpha)
    def _eval_addRatings(self, ratings, new_ratings, userFactors, itemFactors):
        """
        Helper function for evaluation. Adds new_ratings to ratings and calls
        online update to update userFactors and itemFactors.
        
        :returns: (updated_ratings, updated_userFactors, updated_itemFactors)
        """
        new_ratings_schema = StructType([StructField('new_userId', IntegerType(),True),
                                         StructField('new_movieId', IntegerType(), True),
                                         StructField('new_rating', FloatType(), True)])
        new_ratings_df = spark.createDataFrame(new_ratings, 
                                               schema=new_ratings_schema)
        ratings = self._eval_update_ratings(ratings, new_ratings_df)
        userFactors, itemFactors = self._eval_online_update(ratings, 
                                                            new_ratings, 
                                                            userFactors, 
                                                            itemFactors)
        return ratings, userFactors, itemFactors
    
    def _eval_update_ratings(self, ratings, new_ratings):
        """
        Helper function for evaluation. Appends new ratings to old ratings
        DataFrame and returns the updated ratings DataFrame
        
        :returns: updated_ratings
        """
        join_condition = [ratings.userId==new_ratings.new_userId,
                          ratings.movieId==new_ratings.new_movieId]
        ratings = ratings.join(new_ratings,
                                 join_condition,
                                 'outer')
        
        ratings = ratings\
                   .withColumn('new_rating',
                               sqlfun.when(ratings\
                                               .new_rating\
                                               .isNotNull(),
                                           ratings\
                                               .new_rating)\
                                      .otherwise(ratings\
                                                     .rating))
        ratings = ratings\
                   .select(sqlfun.coalesce(ratings.userId,
                                           ratings.new_userId)\
                                 .alias('userId'),
                           sqlfun.coalesce(ratings.movieId,
                                           ratings.new_movieId)\
                               .alias('movieId'),
                           ratings.new_rating\
                               .alias('rating'))
        return ratings
    
    def _eval_online_update(self, ratings, new_ratings, userFactors, itemFactors):
        """
        Helper function for evaluation.
        Updates latent factors of users in new_ratings. If partial_als is
        required due to new movies, will also initialize and update the latent
        factors for these movies. Movies already in the ratings DataFrame will
        not have their latent factors updated.
        
        :returns: updated_userFactors, updated_itemFactors
        """
        users_to_update = list(set([nr[0] for nr in new_ratings]))
        
        for user in users_to_update:
            profile_size = ratings.filter(ratings.userId==user).count()
            userFactors, itemFactors = self._eval_updateUser(user,
                                                             profile_size,
                                                             ratings, 
                                                             userFactors, 
                                                             itemFactors)
        return userFactors, itemFactors
    
    def _eval_updateUser(self, user, profile_size, ratings, userFactors,
                         itemFactors):
        """
        Helper function for evaluation. Updates the latent factors for user.
        If user has rated movies which have been rated by nobody else, will
        instead call partial_als to update user's latent factors as well as
        item factors for these new movies.
        
        :returns: updated_userFactors, updated_itemFactors
        """
        rank = self.als.getRank()
        regParam = self.als.getRegParam()
        # maxIter = self.als.getMaxIter()
        
        user_ratings = ratings.filter(ratings.userId==user)
        user_ratings = user_ratings.join(itemFactors,
                                         user_ratings.movieId==itemFactors.id,
                                         how='inner')
        if profile_size > user_ratings.count():
            return self._eval_partial_als(user, ratings, 
                                          userFactors, itemFactors)
            
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
        userFactors = self._eval_update_userFactors(userFactors, new_userFactors)
        return userFactors, itemFactors
    
    def _eval_partial_als(self, user, ratings, userFactors, itemFactors):
        """
        Helper function for evaluation. Performs ALS to update latent
        factors for user as well as latent factors for movies in user's profile
        which have been rated by nobody else.
        
        :returns: updated_userFactors, updated_itemFactors
        """
        rank = self.als.getRank()
        maxIter = self.als.getMaxIter()
        regParam = self.als.getRegParam()
        
        user_profile = ratings.filter(ratings.userId==user)
        user_profile = user_profile.join(itemFactors,
                                         user_profile.movieId==itemFactors.id,
                                         'left')
        items_to_update = user_profile.filter(col('features').isNull())\
                                      .select('movieId')\
                                      .collect()
        items_to_update = [r[0] for r in items_to_update]
        
        # initialize missing item features
        user_profile = user_profile.withColumn('features',
                                               sqlfun.when(col('features').isNull(),
                                                           sqlfun.array_repeat(\
                                                               sqlfun.sqrt(col('rating'))\
                                                               +0.1*sqlfun.randn(),
                                                               rank))\
                                                     .otherwise(col('features')))
        
        factors_ratings = user_profile.select('movieId', 'features', 'rating')\
                                      .toPandas()
        
        # begin ALS
        for j in range(maxIter):
            H = np.array([features for features in factors_ratings.features.values])
            H = np.concatenate((H, np.sqrt(regParam)*np.eye(rank)))
            
            b = factors_ratings.rating.values
            b = np.concatenate((b, np.zeros(rank)))
            
            w = nnls(H, b)[0] # update latent user factors
            
            W = np.array([w])
            W = np.concatenate((W, np.sqrt(regParam)*np.eye(rank)))
            
            for m in items_to_update:
                b = factors_ratings[factors_ratings.movieId==m].rating.values
                b = np.concatenate((b, np.zeros(rank)))
                h = nnls(W, b)[0]
                h = [float(f) for f in h]
                idx = factors_ratings[factors_ratings.movieId==m].index[0]
                factors_ratings.features[idx] = h
        
        # Update internal latent factors
        new_latentFactor_schema = StructType([StructField('new_id',
                                                        IntegerType(),
                                                        True),
                                            StructField('new_features', 
                                                        ArrayType(FloatType(),True),
                                                        True)])
        new_userFactors = [(user,[float(f) for f in w])]
        new_userFactors = spark.createDataFrame(new_userFactors, 
                                                schema=new_latentFactor_schema)
        userFactors = self._eval_update_userFactors(userFactors, new_userFactors)
        
        new_itemFactors = factors_ratings[['movieId', 'features']]
        new_itemFactors = [tuple(row) for _,row in new_itemFactors.iterrows()]
        new_itemFactors = [(int(mid), list(features)) for mid,features in new_itemFactors]
        new_itemFactors = spark.createDataFrame(new_itemFactors, 
                                                schema=new_latentFactor_schema)
        itemFactors = self._eval_update_itemFactors(itemFactors, new_itemFactors)
        
        return userFactors, itemFactors
    
    def _eval_update_userFactors(self, userFactors, new_factors):
        """
        Helper function for evaluation. Updates old user latent factors
        with new entries in new_userFactors.
        
        :returns: updated_userFactors
        """
        join_condition = userFactors.id==new_factors.new_id
        
        userFactors = userFactors.join(new_factors,
                                         on=join_condition,
                                         how='outer')
        userFactors = userFactors\
                       .withColumn('new_features',
                                   sqlfun.when(userFactors\
                                                   .new_features\
                                                   .isNotNull(),
                                               userFactors\
                                                   .new_features)\
                                          .otherwise(userFactors\
                                                         .features))
        userFactors = userFactors\
                       .select(sqlfun.coalesce(userFactors.id,
                                               userFactors.new_id)\
                                     .alias('id'),
                               userFactors.new_features.alias('features'))
        
        return userFactors
    
    def _eval_update_itemFactors(self, itemFactors, new_factors):
        """
        Helper function for evaluation. Updates old item latent factors with
        new entries in new_itemFactors.
        
        :returns: updated_itemFactors
        """
        join_condition = itemFactors.id==new_factors.new_id
        
        itemFactors = itemFactors.join(new_factors,
                                         on=join_condition,
                                         how='outer')
        itemFactors = itemFactors\
                       .withColumn('new_features',
                                   sqlfun.when(itemFactors\
                                                   .new_features\
                                                   .isNotNull(),
                                               itemFactors.\
                                                   new_features)\
                                          .otherwise(itemFactors\
                                                         .features))
        itemFactors = itemFactors\
                       .select(sqlfun.coalesce(itemFactors.id,
                                               itemFactors.new_id)\
                                     .alias('id'),
                               itemFactors.new_features.alias('features'))
        
        return itemFactors
    
    def _eval_predict(self, user, val_items, userFactors, itemFactors):
        """
        Helper function for evaluation. Makes predictions for user on items
        in val_items. Uses supplied latent factors.
        
        :returns: list of float predictions
        """
        w = userFactors.filter(col('id')==user).collect()[0][1]
        w = np.array(w)
        H = itemFactors.filter(col('id').isin(val_items)).select('id', 
                                                                 'features')\
                                                         .collect()
        preds = []
        for item in val_items:
            h = list(filter(lambda x: x[0]==item, H))[0][1]
            h = np.array(h)
            preds.append(w.dot(h))
        
        return preds
    
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
    
#    def crossValidate(self):
#        pass
#    def save(self):
#        pass
#    def load(self):
#        pass
#    def updateParams(self):
#        pass
    
        

