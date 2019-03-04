#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Film Buff *** scratch work

Author: Jesse Hamer

Date Created: 2/27/19
"""

import pandas as pd
import numpy as np

#%%
# RUN IF SPARK IS WANTED #
from pyspark import SparkContext
from pyspark.sql import SparkSession
spark_home = '/usr/local/spark'
sc = SparkContext('local', 'film_buff', spark_home)
spark = SparkSession(sc)

#%%
# Set directory names #
data_dir= 'datasets/ml-latest/'
sql_dir = 'sq_databases/'

# Set file names; note that .csv needs to be appended #
genome_scores_fn = 'genome-scores'
genome_tags_fn = 'genome-tags'
links_fn = 'links'
# movies_fn = 'movies' --> Does not have IMDB data
movies_fn = 'movies_updated'
ratings_fn = 'ratings'
tags_fn = 'tags'

#%%

# Read pandas dataframes #
genome_scores_pd = pd.read_csv(data_dir + genome_scores_fn + '.csv')
genome_tags_pd = pd.read_csv(data_dir + genome_tags_fn + '.csv')
links_pd = pd.read_csv(data_dir + links_fn + '.csv',
                       dtype={'imdbId':str},
                       usecols=['movieId', 'imdbId'])
links_pd.imdbId = 'tt' + links_pd.imdbId
movies_pd = pd.read_csv(data_dir + movies_fn + '.csv')
ratings_pd = pd.read_csv(data_dir + ratings_fn + '.csv')
tags_pd = pd.read_csv(data_dir + tags_fn + '.csv')

#%%

# Use OMDB API to extract additional movie info and append it to the movies
# dataset #

from omdb import OMDBClient
from requests.exceptions import HTTPError
import time

OMDB_API_KEY='YOUR_OMDB_APIKEY_HERE'
client = OMDBClient(apikey=OMDB_API_KEY)

missing_info = []

i = 0

max_i = links_pd.shape[0]

# Use an indicator so that retry amount is limited.
retry = 0

while i < max_i:
    movieId = links_pd.movieId[i]
    imdbId = links_pd.imdbId[i]
    
    try:
        response = client.imdbid(imdbId)
        movies_pd.loc[movies_pd.movieId==movieId, 'title'] = response['title']
        movies_pd.loc[movies_pd.movieId==movieId, 'year'] = response['year']
        movies_pd.loc[movies_pd.movieId==movieId, 'rated'] = response['rated']
        movies_pd.loc[movies_pd.movieId==movieId, 'runtime'] = response['runtime']
        movies_pd.loc[movies_pd.movieId==movieId, 'director'] = response['director']
        movies_pd.loc[movies_pd.movieId==movieId, 'writer'] = response['writer']
        movies_pd.loc[movies_pd.movieId==movieId, 'actors'] = response['actors']
        movies_pd.loc[movies_pd.movieId==movieId, 'plot'] = response['plot']
        movies_pd.loc[movies_pd.movieId==movieId, 'language'] = response['language']
        movies_pd.loc[movies_pd.movieId==movieId, 'country'] = response['country']
        movies_pd.loc[movies_pd.movieId==movieId, 'awards'] = response['awards']
        movies_pd.loc[movies_pd.movieId==movieId, 'imdb_rating'] = response['imdb_rating']
        movies_pd.loc[movies_pd.movieId==movieId, 'imdb_votes'] = response['imdb_votes']
        movies_pd.loc[movies_pd.movieId==movieId, 'awards'] = response['awards']
        
        # Iterate i and reset retry counter to 0 if successful
        i+=1
        retry = 0
    except KeyError as key:
        print('KeyError for film {}: {} not found.'.format(imdbId, key))
        missing_info.append((imdbId, key))
        i+=1
        retry = 0
    except HTTPError:
        if retry <=3:
            print('HTTPError occured... Waiting 5 seconds and retrying.')
            time.sleep(5)
            retry+=1
        else:
            print('Too many retries. Exiting.')
            break
# Note: tt1628842, tt5300252, tt6086096, tt2081881 - title not found!

# Clean up the movies dataset to be put into the database.
            
movies_pd.imdb_votes = movies_pd.imdb_votes\
                        .str.replace(',', '')\
                        .str.replace('N/A', 'nan')\
                        .astype(float)

# Four of the movies could not properly be loaded using OMDB for some reason.
# We will manually fix these movies.

# This one was possibly meant to be Independence Day: Resurgence, but not sure
movies_pd.iloc[35909, :] = { 'movieId': 143357,
                             'title': 'Independence Day 3',
                             'genres': '(no genres listed)',
                             'year': '2017',
                             'rated': 'N/A',
                             'runtime': 'N/A',
                             'director': 'N/A',
                             'writer': 'N/A',
                             'actors': 'N/A',
                             'plot': 'N/A',
                             'language': 'N/A',
                             'country': 'N/A',
                             'awards': 'N/A',
                             'imdb_rating': np.nan,
                             'imdb_votes': np.nan
                           }

movies_pd.iloc[52226, :] = { 'movieId': 180275,
                             'title': 'Liberation: Battle For Berlin',
                             'genres': movies_pd.iloc[52226, 2],
                             'year': '1971',
                             'rated': 'N/A',
                             'runtime': '79 min',
                             'director': 'Yuriy Ozerov',
                             'writer': 'Yuri Bondarev, Oskar Kurganov',
                             'actors': 'Nikolay Olyalin, Larisa Golubkina'+\
                                        'Barbara Brylska',
                             'plot': "Stalin orders to hasten the Vistula-Oder"+\
                                     "offensive in order to relieve the Allies."+\
                                     "Karl Wolff is sent to negotiate with the"+\
                                     "Americans. Zhukov rejects Stavka's order"+\
                                     "to take Berlin, the Soviets and the Poles"+\
                                     "storm the Tiergarten.",
                             'language': 'Russian, German',
                             'country': 'Soviet Union, East Germany, Poland, Italy',
                             'awards': 'N/A',
                             'imdb_rating': 7.2,
                             'imdb_votes': 125
                           }

movies_pd.iloc[53768, :] = { 'movieId': 183709,
                             'title': 'Harvest',
                             'genres': movies_pd.iloc[52226, 2],
                             'year': '2017',
                             'rated': 'N/A',
                             'runtime': '12 min',
                             'director': 'Kevin Byrnes',
                             'writer': 'Kevin Byrnes (story), Andrew Scott-Ramsay',
                             'actors': 'Patrick Mulvey',
                             'plot': "As we follow Jenni through a week, we"+\
                                     "join a growing audience interested in"+\
                                     "understanding who she is and the patterns"+\
                                     "that define her.",
                             'language': 'N/A',
                             'country': 'USA',
                             'awards': 'N/A',
                             'imdb_rating': 6.5,
                             'imdb_votes': 10
                           }

movies_pd.iloc[57399, :] = { 'movieId': 192089,
                             'title': 'National Theatre Live: One Man, Two Guvnors',
                             'genres': movies_pd.iloc[52226, 2],
                             'year': '2011',
                             'rated': '12A',
                             'runtime': '180 min',
                             'director': 'Robin Lough',
                             'writer': 'Richard Bean, Carlo Goldoni (play)',
                             'actors': 'David Benson, Oliver Chris, Polly Conway',
                             'plot': "Fired from his skiffle band, Francis"+\
                                     "Henshall becomes minder to Roscoe Crabbe,"+\
                                     "a small time East End hood, now in"+\
                                     "Brighton to collect £6,000 from his"+\
                                     "fiancée's dad. But Roscoe is really his"+\
                                     "sister Rachel posing as her own dead"+\
                                     "brother, who's been killed by her"+\
                                     "boyfriend Stanley Stubbers. Holed up at"+\
                                     "The Cricketers' Arms, the permanently"+\
                                     "ravenous Francis spots the chance of an"+\
                                     "extra meal ticket and takes a second job"+\
                                     "with one Stanley Stubbers, who is hiding"+\
                                     "from the police and waiting to be"+\
                                     "re-united with Rachel. To prevent"+\
                                     "discovery, Francis must keep his two"+\
                                     "guvnors apart. Simple.",
                             'language': 'English',
                             'country': 'UK',
                             'awards': 'N/A',
                             'imdb_rating': 8.3,
                             'imdb_votes': 84
                           }

def get_uniques(attr_idx, data, split_char=None, strip=False):
    unique_attrs=dict()
    is_list = False
    for i in range(data.shape[0]):
        attrs = data.iloc[i, attr_idx]
        try:
            if split_char:
                attrs_list = attrs.split(split_char)
                is_list=True
                if strip:
                    attrs_list = [a.strip() for a in attrs_list]
        except AttributeError:
                print('Attribute error: \n{}'.format(data.iloc[i, :]))
        if is_list:
            for attr in attrs_list:
                if attr in unique_attrs.keys():
                    unique_attrs[attr]+=1
                else:
                    unique_attrs[attr]=1
        else:
            if attrs in unique_attrs.keys():
                unique_attrs[attrs]+=1
            else:
                unique_attrs[attrs]=1
    return unique_attrs

unique_genres = get_uniques(2, movies_pd, split_char='|')
unique_rated = get_uniques(4, movies_pd)
unique_languages = get_uniques(10, movies_pd, split_char=',', strip=True)
unique_countries = get_uniques(11, movies_pd, split_char=',', strip=True)

def alphabetize(attr_idx, data, split_char, strip=False):
    for i in range(data.shape[0]):
        attrs = data.iloc[i, attr_idx]
        attrs_list = attrs.split(split_char)
        if strip:
            attrs_list = [a.strip() for a in attrs_list]
        attrs_list = sorted(attrs_list)
        attrs = ''
        for attr in attrs_list:
            attrs+=attr+'|'
        attrs = attrs.rstrip('|')
        data.iloc[i, attr_idx] = attrs

# Alphabetize the lists of genres, directors, writers, languages, and countries
alphabetize(2, movies_pd, '|')
alphabetize(6, movies_pd, ',', strip=True)
alphabetize(7, movies_pd, ',', strip=True)
alphabetize(10, movies_pd, ',', strip=True)
alphabetize(11, movies_pd, ',', strip=True)

# Some investigation revealed UNRATED and NOT RATED mean the same thing.
# Also, M and GP are archaic versions of PG. For consistency with IMDB, we
# will keep them, but the 7 movies with dual M/PG rating will be converted
# to M for simplicity, since all were released when M was the official rating.
# For later analysis, it may be wise to replace M and GP with PG.
movies_pd.rated = movies_pd.rated.str.upper()
movies_pd.rated = movies_pd.rated.str.replace('UNRATED', 'NOT RATED')
movies_pd.rated = movies_pd.rated.str.replace('NR', 'NOT RATED')
movies_pd.rated = movies_pd.rated.str.replace('M/PG', 'M')

movies_pd.imdb_rating = movies_pd.imdb_rating\
                                 .str.replace('N/A', 'nan')\
                                 .astype(float)

movies_pd.to_csv(data_dir + 'movies_updated.csv', index=False)
    
client.session.close()
        

#%%
# Set up SQL database #

from sqlalchemy import create_engine, Table, Column, Integer, String, Float,\
                         MetaData
postgresql_uri = 'postgresql://postgres:es33590811@localhost/movie_lens'
engine = create_engine(postgresql_uri)

metadata = MetaData()

movies_table = Table('movies', metadata,
                     Column('movieId', Integer, primary_key=True),
                     Column('title', String),
                     Column('genres', String),
                     Column('year', String),
                     Column('rated', String),
                     Column('runtime', String),
                     Column('director', String),
                     Column('writer', String),
                     Column('actors', String),
                     Column('plot', String),
                     Column('language', String),
                     Column('country', String),
                     Column('awards', String),
                     Column('imdb_rating', Float),
                     Column('imdb_votes', Integer),
                     Column('awards', String))

links_table = Table('links', metadata,
                    Column('movieId', Integer, primary_key=True),
                    Column('imdbId', String))

genome_tags_table = Table('genome_tags', metadata,
                          Column('tagId', Integer, primary_key=True),
                          Column('tag', String))

genome_scores_table = Table('genome_scores', metadata,
                            Column('movieId', Integer, primary_key=True),
                            Column('tagId', Integer, primary_key=True),
                            Column('relevance', Float))

tags_table = Table('tags', metadata,
                   Column('userId', Integer),
                   Column('movieId', Integer),
                   Column('tag', String),
                   Column('timestamp', Integer))

ratings_table = Table('ratings', metadata,
                      Column('userId', Integer, primary_key=True),
                      Column('movieId', Integer, primary_key=True),
                      Column('rating', Float),
                      Column('timestamp', Integer))


metadata.create_all(engine)

# Populate tables with pandas dataframes

movies_pd.to_sql('movies', engine, if_exists='append', index=False)
links_pd.to_sql('links', engine, if_exists='append', index=False)
genome_tags_pd.to_sql('genome_tags', engine, if_exists='append', index=False)
genome_scores_pd.to_sql('genome_scores', engine, if_exists='append', index=False)
tags_pd.to_sql('tags', engine, if_exists='append', index=False)
ratings_pd.to_sql('ratings', engine, if_exists='append', index=False)

#%%

# Train recommendation engine using Spark's MLLib library

from pyspark.ml.recommendation import ALS

ratings_pd.drop(columns=['timestamp'], inplace=True)

# First perform some training time analysis. Note that num_iterations is set
# to 10 for the training algorithm.

train_time_data = pd.DataFrame(columns=['num_users', 'total_obs', 
                                        'num_features', 'train_time_sec', 
                                        'model_type'])

all_users = ratings_pd.userId.unique()

num_users_list = [100, 500, 1000, 5000, 10000]

dataset_sizes = []

num_features = [10, 50, 100]

# Perform analysis 5 times
for i in range(5):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('~~~~~~~~~~~ITERATION {}~~~~~~~~~~~~~'.format(i))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    for num_users in num_users_list:
        users = np.random.choice(all_users, num_users, replace=False)
        ratings_subset_pd = ratings_pd[np.isin(ratings_pd.userId, users)]
        dataset_sizes.append(ratings_subset_pd.shape[0])
        print('Num Users: {}, Total Ratings: {}'.format(num_users, 
                                                          dataset_sizes[-1]))
        ratings_subset_df = spark.createDataFrame(ratings_subset_pd)
        ratings_subset_df = ratings_subset_df.withColumn('imp_rating', 
                                                         (ratings_subset_df.\
                                                          rating>=4.0).cast('int'))
        for nf in num_features:
            als = ALS(rank=nf, userCol='userId', itemCol='movieId', 
                      ratingCol='rating', seed=9, nonnegative=True)
            print('Fitting {}-feature explicit model...'.format(nf))
            start_time = time.time()
            als.fit(ratings_subset_df)
            elapsed_time = time.time()-start_time
            new_data = {'num_users': num_users, 'total_obs':dataset_sizes[-1], 
                        'num_features': nf, 'train_time_sec': elapsed_time,
                        'model_type':'explicit'}
            train_time_data = \
                train_time_data.append(new_data, ignore_index=True)
            print('Finished fitting model.\n Total elapsed time: {}'.format(
                    np.round(elapsed_time, 2)))
            #######
            
            als = ALS(rank=nf, userCol='userId', itemCol='movieId', nonnegative=True,
                      ratingCol='imp_rating', implicitPrefs=True, seed=9)
            print('Fitting {}-feature implicit model...'.format(nf))
            start_time = time.time()
            als.fit(ratings_subset_df)
            elapsed_time = time.time()-start_time
            new_data = {'num_users': num_users, 'total_obs':dataset_sizes[-1], 
                        'num_features': nf, 'train_time_sec': elapsed_time,
                        'model_type':'implicit'}
            train_time_data = \
                train_time_data.append(new_data, ignore_index=True)
            print('Finished fitting model.\n Total elapsed time: {}'.format(
                    np.round(elapsed_time, 2)))
            print('############################\n')
        print('**************************\n')
        
train_time_data.to_csv('datasets/train_time_data.csv',
                                index=False)




















