# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:52:07 2017

@author: Dustin
"""

import numpy as np
import pandas as pd
import generation_functions as gf
from timeit import default_timer
import requests, json
from types import SimpleNamespace
from PIL import Image

import matplotlib.pyplot as plt

train_images = []
train_labels = []
row_per_card = 15

set_name = "2xm"

set_url = "https://api.scryfall.com" + "/sets/" + set_name
response = requests.get(set_url)
current_set = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))

for j in range(current_set.card_count):
    t0 = default_timer()
    card_number = j + 1
    gf.CallImage(set_name, card_number)
    im = gf.PullImage(set_name, card_number)
    counter = 1
    for i in range(row_per_card):
        im1 = gf.DirtyImage(im)
        # im2 = gf.d_reshape(im1)
        img = Image.fromarray(im1, 'RGB')
        img.save(gf.DirtyCardFileName(set_name, card_number) + "_" + str(counter) + '.png')
        train_images.append( im1 )
        train_labels.append( gf.CardName(set_name, card_number))
        counter += 1
    t1 = default_timer(); t_percard = t1 - t0
    print( gf.DirtyCardFileName(set_name, card_number) + ' is complete! ' +"(" + str(j) + "/" + str(current_set.card_count) + ") " + str(t_percard) + 'seconds')
    