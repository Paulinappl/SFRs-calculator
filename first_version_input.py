# Take UGRZ
# SFR OPTIC and UV 
# See what parameters of ilustris work
# Use 11 halos (Galaxy clusters) 
# SUB-HALOS = GALAXIES

# USE COLOR G - R AND MASS
#  CORRELATION MATRIXES
# 


# Luis and Paulina are going to work today on writing the code to take user input,
# and Jiani and I are going to work on getting the TNG data and processing it to a 
# form the model can use. If you want to work on any of that, let us know! Tomorrow 
# we can actually work on the ML model itself and on predicting the SFRs from the user photometry


# First, we need to import the main modules to our script. 

import csv
import numpy as np
import pandas as pd
import matplotli.pyplot as plt

#This command reads the dataframe
data = pd.read_csv("A2670Finalcat.csv")


# Here will go the part of the code for retrive the TNG data


# Here, we start coding our ML model.
# Firts, we need to chose the ML model to use to reach our goal. 
# We want to 'predict' the SFR in clusters galaxies using their optical 
# photometry and possibly another properties (e.g., mass, size or HI 
# content, etc.), then a good machine learning method (and basic) is the 
# regression lineal modeling. 

#The primary goal of linear regression is to find the "best-fit" line (or 
# hyperplane in higher dimensions) that minimizes the difference between 
# the predicted values and the actual observed values.

# This best-fit line is defined by a linear equation of the form:

# Y = b0 ​+ b1​X1 ​+ b2​X2 ​+...+ bn​Xn​

# In this equation:

 #   Y represents the dependent variable we want to predict.
 #   X1,X2,...,Xn are the independent variables or features.
 #   b0 is the intercept (the value of Y when all X values are zero).
 #   b1,b2,...,bn are the coefficients that determine the relationship 
 #   between each independent variable and the dependent variable.

# Linear regression assumes that there is a linear relationship between 
# the predictors and the target variable. The goal of the model is to 
# estimate the coefficients (b0,b1,...,bn) that minimize the sum of the 
# squared differences between the predicted values and the actual values 
# in the training data. This process is often referred to as 
# "fitting the model."

# We start importing the basic packages (note you can put out some 
# of them)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy import stats

# Now, we import the 'dataset', for this code, we are using TNG Data. 

df=pd.read_csv('TNG_data')

# A extremly important step is to determine the ouliers in the data set.
# There are several methods. We follow a basic search for multicollinearity
# between variables. Thus a helpful tool is a correlation matrix. 

def tidy_corr_matrix(corr_mat):
    '''
    This funtion turn the pandas corr_matrix into a tidy format
    '''
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1','variable_2','r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)
    
    return(corr_mat)

corr_matrix = datos.select_dtypes(include=['float64', 'int']).corr(method='pearson')
tidy_corr_matrix(corr_matrix).head(10) #Here you can change 10 for the number of 
# independent variable in the dataset

# Plotting the correlation matrix 

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

sns.heatmap(
    corr_matrix,
    annot     = True,
    cbar      = False,
    annot_kws = {"size": 8},
    vmin      = -1,
    vmax      = 1,
    center    = 0,
    cmap      = sns.diverging_palette(20, 220, n=200),
    square    = True,
    ax        = ax
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation = 45,
    horizontalalignment = 'right',
)

ax.tick_params(labelsize = 10)

# Visualization data distribution 

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 5)) # You replace 2 by 
# number of varibles
axes = axes.flat
columnas_numeric = datos.select_dtypes(include=['float64', 'int']).columns

for i, colum in enumerate(columnas_numeric):
    sns.histplot(
        data    = datos,
        x       = colum,
        stat    = "count",
        kde     = True,
        color   = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
        line_kws= {'linewidth': 2},
        alpha   = 0.3,
        ax      = axes[i]
    )
    axes[i].set_title(colum, fontsize = 10, fontweight = "bold")
    axes[i].tick_params(labelsize = 8)
    axes[i].set_xlabel("")


    
fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('Distribución variables numéricas', fontsize = 10, fontweight = "bold");










