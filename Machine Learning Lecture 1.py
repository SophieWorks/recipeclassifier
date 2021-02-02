# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 00:09:51 2021

@author: sophi
"""

import numpy as np
import pandas as pd

from sklearn import svm

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

recipes = pd.read_excel('cupvsmuf.xlsx')
print(recipes.head())


#PLOT!!!

sns.lmplot('Flour','Sugar', data=recipes, hue='Type',
           palette='Set1',fit_reg=True, 
           scatter_kws={"s":70})


#FORMAT!!

type_label=np.where(recipes['Type']=='Muffin', 0, 1)
recipe_features = recipes.columns.values[1:].tolist()
print('Recipe features=', recipe_features)

ingredients = recipes[['Flour','Sugar']].values
print(ingredients)

#fit model!! svc - support vector calssification

model=svm.SVC(kernel='linear')
model.fit(ingredients,type_label)

#calculating hyperplane...math stuff!

w=model.coef_[0]
a = -w[0]/w[1]
xx=np.linspace(30,60)
yy=a*xx-(model.intercept_[0])/w[1]


#plotting hyperplace

b = model.support_vectors_[0]
yy_down=a*xx+(b[1]-a*b[0])
b=model.support_vectors_[-1]
yy_up=a*xx+(b[1]-a*b[0])

sns.lmplot('Flour','Sugar', data=recipes, hue='Type',
           palette='Set1',fit_reg=False, 
           scatter_kws={"s":70})
plt.plot(xx,yy,linewidth=2,color='black')
plt.plot(xx,yy_down,'k--')
plt.plot(xx,yy_up,'k--')
