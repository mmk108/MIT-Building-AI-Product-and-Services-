#!/usr/bin/env python
# coding: utf-8

# # Define training and testing sets in practice
# 
# **Author: Jessica Cervi**
# 
# 
# ## Activity Overview
# 
# This activity is designed to consolidate your knowledge about the differences in the training and testing sets and to teach you  how to define those in `Python` using the `sklearn` library.
# 
# In this activity, we'll use one of the toy datasets made available on `sklearn`. We choose to use the `wine` dataset.
# 
# 
# This assignment is designed to help you apply the machine learning algorithms you've learned using the packages in `Python`. `Python` concepts, instructions, and starter code are embedded within this Jupyter Notebook to help guide you as you progress through the activity. Remember to run the code of each code cell prior to submitting the assignment. Upon completing the activity, we encourage you to compare your work against the solution file to perform a self-assessment.

# ## Define Training and Testing Sets in Practice
# 
# We have seen that it is standard practice to split the data available into a training and testing sets to avoid overfitting, choose the best model to use, and to minimized errors.

# ### Importing the Dataset and Exploratory Data Analysis (EDA)
# 
# We begin by using the libraries `sklearn`, `NumPy`, and `pandas` to import and read the datasets.  Let's have a closer look at each of them:
# 
# - `NumPy` is a library for the `Python` programming language that adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. The code within the `NumPy` library is divided into submodules to faciltate the usage. For example, in the code cell below, we import the module `random` used to generate and work with random numbers.
# 
# - `pandas` is a  software library written for the `Python` programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating dataframes and numerical tables series.
# 
# - `load_wine` is one of the toy datasets readily available from the `sklearn` library. `Scikit-learn` (also known as `sklearn`) is a free software machine learning library for the `Python` programming language. It features various classification, regression, and clustering algorithms.
# 
#     In the code cell below, we start by importing the necessary libraries and modules. Next, we load the dataset from `sklearn` and assign that to the variable `wine`. Next, we use a combination of `NumPy`  and `pandas` functions to create the dataframe `df`.
# 
# **Note:** This is not the standard way of importing data during a regular project. The code in the cell below is only appropriate to use when importing toy datasets from `sklearn`. 

# In[1]:


#import the libraries
import pandas as pd 
import numpy as np
from sklearn.datasets import load_wine

# save load_wine() sklearn dataset to wine
# if you'd like to check dataset type use: type(load_wine())
# if you'd like to view list of attributes use: dir(load_wine())
wine = load_wine()

# np.c_ is the numpy concatenate function
# which is used to concat wine['data'] and wine['target'] arrays 
# for pandas column argument: concat wine['feature_names'] list
# and string list (in this case one string); you can make this anything you'd like..  
# the original dataset would probably call this ['Species']

df = pd.DataFrame(data= np.c_[wine['data'], wine['target']], #using Dataframe to create a dataframe
                     columns= wine['feature_names'] + ['target'])


# Before performing any algorithm on the dataframe, it's always good practice to perform exploratory data analysis.
# 
# We begin by visualizing the first several rows of the DataFrame `df` using the function `.head()`. By default, `.head()` displays the first five rows of a DataFrame; this can be changed by passing the desired number of rows to the function `.head()` as an integer.
# 
# Complete the code below to visualize the first ten rows of `df`.

# In[2]:


df.head( )


# Next, we retrieve some more information about our DataFrame by using the properties `.shape` and `columns`.
# 
# Here's a brief description of what each of the above functions does:
# 
# - `.shape`: Returns a tuple representing the dimensionality of the DataFrame.
# - `.columns`: Returns the column labels of the DataFrame.
# - `.describe`(): Computes and shows summary statistics related to the DataFrame.
# 
# 
# Run the cells below to get information aboout the DataFrame.

# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.describe()


# ### Separating the Dataset Into a Training and Testing Dataset
# 
# 
# Before you separate our original data into training and testing sets in `Python` using `sklearn`, in the text cell below provide a short description about the differences between the two sets.

# **DOUBLE CLICK ON THIS CELL TO TYPE YOUR ANSWER**
# 
# **YOUR ANSWER HERE:** 

# As you have seen in Video 3 for this week, it is important to split the data into  *training* and *testing* sets.
# 
# To split the data into  training and testing datasets, we can use the function [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) from `sklearn`. This function does a random split of data arrays or matrices into train and test subsets and returns a list containing a train-test split of inputs.
# 
# As we observe, in our case, the function `train_test_split` takes four arguments:
# 
# - `X`: Input dataframe
# - `y`: Output dataframe
# - `test_size`: Should be between 0.0 and 1.0 and should represent the proportion of the dataset to include in the test split
# - `random_state`: Controls the shuffling applied to the data before applying the split. Ensures the reproducibility of the results across multiple function calls
# 
# In the code cell below, fill-in the ellipsis to set the argument `test_size` equal to `0.3` and `random_state` equal to `123`.

# In[6]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, df, test_size=.3, random_state=123)



# You can see the size of the resulting train and test subsets using `.shape`:

# In[7]:


X_train.shape


# In[8]:


X_test.shape


# We will learn how to separate the data into inputs and outputs  and how to implement algorithms in the next segments of this week of the course.
# 
# Stay tuned!
