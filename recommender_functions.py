#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:38:43 2019

Author: Jesse Hamer

Auxilliary functions for use with collectionRecommender.
"""

import numpy as np
import pandas as pd

from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql.types import StructType,StructField,ArrayType,\
                            IntegerType,FloatType
import pyspark.sql.functions as sqlfun
from pyspark.sql.functions import col

from scipy.optimize import nnls

from itertools import product

import time

# Assumes active Spark session called spark.

def update_ratings(ratings, new_ratings):
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

def update_predictions(predictions, new_predictions):
    join_condition = [predictions.userId==new_predictions.userId_new,
                      predictions.movieId==new_predictions.movieId_new]
    predictions = predictions.join(new_predictions,
                                   join_condition,
                                   'outer')
    
    predictions = predictions.withColumn('predictedRating_new',
                                         sqlfun.when(predictions\
                                                         .predictedRating_new\
                                                         .isNotNull(),
                                                     predictions\
                                                         .predictedRating_new)\
                                               .otherwise(predictions\
                                                          .predictedRating))
    predictions = predictions.select(sqlfun.coalesce(predictions.userId,
                                                   predictions.userId_new)\
                                                   .alias('userId'),
                                   sqlfun.coalesce(predictions.movieId,
                                                   predictions.movieId_new)\
                                                   .alias('movieId'),
                                   predictions.predictedRating_new\
                                              .alias('predictedRating'))
                                   
    return predictions

def update_latentFactors(old_factors, new_factors):
    join_condition = old_factors.id==new_factors.new_id
    
    old_factors = old_factors.join(new_factors,
                                   on=join_condition,
                                   how='outer')
    old_factors = old_factors.withColumn('new_features',
                                         sqlfun.when(old_factors\
                                                     .new_features\
                                                     .isNotNull(),
                                                     old_factors\
                                                     .new_features)\
                                               .otherwise(old_factors.features))
    old_factors = old_factors.select(sqlfun.coalesce(old_factors.id,
                                                     old_factors.new_id)\
                                           .alias('id'),
                                     old_factors.new_features.alias('features'))
                                         
    return old_factors

# Spark SQL UDFs for creating new columns.
@sqlfun.udf(returnType=FloatType())
def dot_product(x):
    total=0
    for pair in x:
        total+=pair[0]*pair[1]
    return total

@sqlfun.udf(returnType=ArrayType(StructType([StructField('userId', 
                                                         IntegerType()),
                                   StructField('movieId', IntegerType()),
                                   StructField('rating', FloatType())])))
def extract_slice(a, start, stop):
    return a[start:stop]

@sqlfun.udf(returnType=StructType([StructField('first_features', 
                                               ArrayType(FloatType())),
                                   StructField('second_features', 
                                               ArrayType(StructType([StructField('id', 
                                                                                 IntegerType()),
                                                                     StructField('features', 
                                                                                 ArrayType(FloatType()))])))]))
# Used in online updating
def new_lf(latentFactor_list, rating_list, rank, maxIter, regParam):
    factors_to_update = []
    latentFactor_list = [list(r) for r in latentFactor_list]
    
    for i in range(len(latentFactor_list)):
        if latentFactor_list[i][1] is None:
            factors_to_update.append(i)
            latentFactor_list[i][1] = np.sqrt(rating_list[i])*np.ones(rank)
        else:
            latentFactor_list[i][1] = np.array(latentFactor_list[i][1])
            
    
    b = np.array(rating_list)
    A = np.array([lf[1] for lf in latentFactor_list])
    
    b = np.concatenate((b, np.zeros(rank)))
    A = np.concatenate((A, np.sqrt(regParam)*np.eye(rank)))
    
    w = nnls(A, b)[0]
    
    
    if factors_to_update:
        for j in range(maxIter):
            W = np.array([w])
            W = np.concatenate((W, np.sqrt(regParam)*np.eye(rank)))
            
            for f in factors_to_update:
                b_partial = np.concatenate((b[f:f+1], np.zeros(rank)))
                h = nnls(W, b_partial)[0]
                A[f] = h
            
            w = nnls(A, b)[0]
    
    first_features = [float(x) for x in w]
    
    updated_ids = np.array([t[0] for t in latentFactor_list])[factors_to_update]
    updated_ids = [int(x) for x in updated_ids]
    
    updated_factors = A[factors_to_update]
    updated_factors = [[float(x) for x in r] for r in updated_factors]
    
    second_features = list(zip(updated_ids, updated_factors))
    
    if not second_features:
        second_features = None
            
    return [first_features, second_features]

# used in online updating
@sqlfun.udf(returnType=ArrayType(FloatType()))
def get_avg(features_list):
    M = np.array(features_list)
    avgs = list(M.mean(axis=0))
    return [float(x) for x in avgs]

@sqlfun.udf(returnType=FloatType())
def get_sqrt(num):
    return float(np.sqrt(num))

@sqlfun.udf(returnType=IntegerType())
def get_obscurity(profile, train_items, M):
    profile_size = len(profile)
    nonobscure_items = [mid for _,mid,_ in profile if mid in train_items]
    obscurity = profile_size - len(nonobscure_items)
    if profile_size >=40:
        if obscurity>=profile_size-5*M/4:
            return None
        else:
            return 1
    else:
        if obscurity >= int(3*profile_size/8):
            return None
        else:
            return 1
        
@sqlfun.udf(returnType=ArrayType(StructType([StructField('userId', 
                                                         IntegerType()),
                                   StructField('movieId', IntegerType()),
                                   StructField('rating', FloatType())])))
def get_val_set(validation_set, train_items):
    val_set = [t for t in validation_set if t[1] in train_items]
    return val_set

def product_dict(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()
    for p in product(*values):
        yield dict(zip(keys, p))

                
    
    
    
    
    
    
    
    