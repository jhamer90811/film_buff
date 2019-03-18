#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 15:38:12 2019

Author: Jesse Hamer

Collection of functions for user collection visualization using Bokeh.
"""

import numpy as np
import pandas as pd

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Panel, Tabs, TextInput, Dropdown
from bokeh.io import curdoc
from bokeh.layouts import column, widgetbox, row
from bokeh.palettes import Spectral6

# These data reading lines are for demo purposes only. The actual app
# should query a database to pull the appropriate user data.
data_info = pd.read_csv('datasets/collection_info_127765.csv')
data_genome = pd.read_csv('datasets/collection_genomes_127765.csv')

to_drop = data_info.director.isna() | data_info.actors.isna()
data_info = data_info[~to_drop]
data_genome = data_genome[~to_drop]

colors = Spectral6*5
        
# The first panel will aggregate tag genomes across the entire collection
# and plot the top K tags by relevance as a bar chart. Tooltips will yield
# the 5 films wich have the highest relevance for the corresponding tag.

tag_means = data_genome.drop(columns='movieId')\
                       .mean(axis=0)\
                       .sort_values(ascending=False)

def get_top_5_by_relevance(tags):
    top_5_lists = []
    for tag in tags:
        top_5 = data_genome[['movieId', tag]].sort_values(tag,
                                                           ascending=False)\
                                              .head()\
                                              [['movieId']]
        top_5 = top_5.join(data_info[['title']]).title
        top_5 = '*|*'.join(top_5)
        top_5_lists.append(top_5)
    return top_5_lists

p1_data_dict = {'tags':list(tag_means[:10].index),
                'relevance': list(tag_means[:10]),
                'top_5': get_top_5_by_relevance(tag_means[:10].index),
                'color':colors[:10]}

p1_data_source = ColumnDataSource(p1_data_dict)

p1_TOOLTIPS = [('Tag', '@tags'),
               ('Average Relevance', '@relevance{00.00%}'),
               ('Most relevant films', '@top_5')]

p1 = figure(x_range=list(tag_means[:10].index), x_axis_label = 'Tags', 
            y_axis_label='Average Relevance', tooltips=p1_TOOLTIPS)

p1_bar = p1.vbar(x='tags', width=0.9, top='relevance', source=p1_data_source,
                 color='color')

p1.xaxis.major_label_orientation = 'vertical'

p1.width=1000
p1.height=1000

def update_p1(attr, old, new):
    new = int(new)
    if new<1:
        new=1
        p1_input.value=str(new)
    if new>30:
        new=30
        p1_input.value=str(new)
        
    new_tags = list(tag_means[:new].index)
    new_relevances = list(tag_means[:new])
    new_top_5 = get_top_5_by_relevance(new_tags)
    new_colors = colors[:new]
    
    new_p1_data = {'tags': new_tags,
                   'relevance': new_relevances,
                   'top_5': new_top_5,
                   'color': new_colors}
    
    p1.x_range.factors = new_tags
    p1_data_source.data = new_p1_data

p1_input = TextInput(title='Number of Top Tags', value=str(10), 
                      placeholder='# between 1 and 30')
p1_input.on_change('value', update_p1)

p1_widgets = widgetbox([p1_input])

p1_panel = Panel(child=row([p1_widgets, p1]), 
                 title='Most Relevant Tags Across Collection')

# Pane 2 will offer a dropdown menu to choose from Directors or
# Actors. Given the chosen group, it will display the most-represented
# individuals in the user's collection, and tooltips will display their
# top highest IMDB-rated films

director_counts = {}
actor_counts = {}
for _, r in data_info.iterrows():
    director_list = r.director.split('|')
    director_list = [d.strip() for d in director_list]
    for d in director_list:
        if d in director_counts.keys():
            director_counts[d]+=1
        else:
            director_counts[d]=1
    actor_list = r.actors.split(',')
    actor_list = [a.strip() for a in actor_list]
    for a in actor_list:
        if a in actor_counts.keys():
            actor_counts[a]+=1
        else:
            actor_counts[a]=1

director_counts = list(director_counts.items())
actor_counts = list(actor_counts.items())

director_counts = sorted(director_counts, key = lambda t: (-t[1], t[0]))
directors = [d[0] for d in director_counts]
directors_num_movies = [d[1] for d in director_counts]

actor_counts = sorted(actor_counts, key= lambda t: (-t[1], t[0]))
actors = [a[0] for a in actor_counts]
actors_num_movies = [a[1] for a in actor_counts]

names_dict = {'director': directors, 'actors': actors}
nums_dict = {'director': directors_num_movies, 'actors': actors_num_movies}

def get_top_5_by_imdb(names, group):
    top_5_lists = []
    for name in names:
        mask = data_info[group].apply(lambda x: name in x)
        movies = data_info[mask][['title', 'imdb_rating']].sort_values('imdb_rating',
                                                                      ascending=False)\
                                                          .head()
        movies = '(' + movies.title + ',' + movies.imdb_rating.astype(str) + ')'
        top_5_lists.append('*|*'.join(movies))
    return top_5_lists

p2_data_dict = {'names': names_dict['director'][:5],
                'nums': nums_dict['director'][:5],
                'top_5': get_top_5_by_imdb(names_dict['director'][:5], 
                                           'director'),
                'color': colors[:5]}

p2_data_source = ColumnDataSource(data=p2_data_dict)

p2_TOOLTIPS = [('Director', '@names'),
               ('Number of Films', '@nums'),
               ('Top 5 IMDB-Rated Films', '@top_5')]

p2 = figure(x_range = names_dict['director'][:5], x_axis_label = 'Directors',
            y_axis_label = 'Number of Films in Collection',
            tooltips = p2_TOOLTIPS)

p2.vbar(x='names', top='nums', color='color', source=p2_data_source, width=0.9)

p2.xaxis.major_label_orientation = 'vertical'
p2.width=1000
p2.height=1000

def update_p2_input(attr, old, new):
    new = int(new)
    if new < 1:
        new = 1
        p2_input.value = str(new)
    if new>30:
        new=30
        p2_input.value = str(new)
    
    current_group = p2_dropdown.value
    
    new_names = names_dict[current_group][:new]
    new_nums = nums_dict[current_group][:new]
    new_top_5 = get_top_5_by_imdb(new_names, current_group)
    new_colors = colors[:new]
    
    new_data = {'names': new_names,
                'nums': new_nums,
                'top_5': new_top_5,
                'color': new_colors}
    
    p2.x_range.factors = new_names
    p2_data_source.data = new_data
    
p2_input = TextInput(title = 'Number of directors to display',
                     value = str(5),
                     placeholder = '# between 1 and 30')
                     
p2_input.on_change('value', update_p2_input)
    
def update_p2_dropdown(attr, old, new):
    
    current_num_movies = int(p2_input.value)
    
    new_names = names_dict[new][:current_num_movies]
    new_nums = nums_dict[new][:current_num_movies]
    new_top_5 = get_top_5_by_imdb(new_names, new)
    new_colors = colors[:current_num_movies]
    
    new_data = {'names': new_names,
                'nums': new_nums,
                'top_5': new_top_5,
                'color': new_colors}
    
    new_names_label = 'Director' if new=='director' else 'Actor'
    
    new_p2_TOOLTIPS = [(new_names_label, '@names'),
                       ('Number of Films', '@nums'),
                       ('Top 5 IMDB-Rated Films', '@top_5')]
    
    p2.xaxis.axis_label = new_names_label
    p2.tools[-1].tooltips = new_p2_TOOLTIPS
    p2.x_range.factors = new_names
    p2_data_source.data = new_data
    
p2_dropdown = Dropdown(label='Choose who to plot', 
                       menu = [('Directors', 'director'), 
                               ('Actors', 'actors')],
                       value = 'director')

p2_dropdown.on_change('value', update_p2_dropdown)

p2_widgets = widgetbox([p2_input, p2_dropdown])
        
p2_panel = Panel(child=row([p2_widgets, p2]),
                 title='Most Films by Director or Actor')

tabs = Tabs(tabs=[p1_panel, p2_panel])

curdoc().add_root(tabs)
