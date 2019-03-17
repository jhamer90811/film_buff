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
from bokeh.models.widgets import Panel, Tabs, Slider
from bokeh.io import curdoc
from bokeh.layouts import column, widgetbox

# These data reading lines are for demo purposes only. The actual app
# should query a database to pull the appropriate user data.
data_info = pd.read_csv('datasets/collection_info_127765.csv')
data_genome = pd.read_csv('datasets/collection_genomes_127765.csv')

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
        
        
# First figure will aggregate tag genomes across the entire collection
# and plot the top K tags by relevance as a bar chart. Tooltips will yield
# the 5 films wich have the highest relevance for the corresponding tag.

p1_data_dict = {'tags':list(tag_means[:10].index),
                'relevance': list(tag_means[:10]),
                'top_5': get_top_5_by_relevance(tag_means[:10].index)}

p1_data_source = ColumnDataSource(p1_data_dict)

p1_TOOLTIPS = [('Tag', '@tags'),
               ('Average Relevance', '@relevance{00.00%}'),
               ('Most relevant films', '@top_5')]

p1 = figure(x_range=list(tag_means[:10].index), x_axis_label = 'Tags', 
            y_axis_label='Average Relevance', tooltips=p1_TOOLTIPS)

p1_bar = p1.vbar(x='tags', width=0.9, top='relevance', source=p1_data_source)

def update_p1(attr, old, new):
    new_tags = list(tag_means[:new].index)
    new_relevances = list(tag_means[:new])
    new_top_5 = get_top_5_by_relevance(new_tags)
    
    new_p1_data = {'tags': new_tags,
                   'relevance': new_relevances,
                   'top_5': new_top_5}
    
    p1.x_range.factors = new_tags
    p1_data_source.data = new_p1_data

p1_slider = Slider(start=1, end=100, value=10, 
                   step=1, title='Number of Top Tags')
p1_slider.on_change('value', update_p1)

p1_widgets = widgetbox([p1_slider])

p1_panel = Panel(child=column([p1_widgets, p1]), 
                 title='Most Relevant Tags Across Collection')











tabs = Tabs(tabs=[p1_panel])

curdoc().add_root(tabs)
