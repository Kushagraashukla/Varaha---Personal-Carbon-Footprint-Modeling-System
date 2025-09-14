# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

# Model load
with open(r"C:/Users/OMEN/OneDrive/Documents/machine learning/ML projects/carbonfootprintpredictor/trained_model.sav", "rb") as f:
    loaded_model = pickle.load(f)

# New input row (same feature order as training)
new_data = np.array([[
    1,275,16,4,34,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,1,0,0,
    False,False,False,False,False,True,False,False,True,True,
    False,False,False,True,False,False,False,False,
    True,False,False,False,False,False,False,True,False,False,True
]])
yy = loaded_model.predict(new_data)

print("The average carbon footprint of the person is", yy[0])

