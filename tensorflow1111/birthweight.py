import numpy as np
import pandas as pd
#import statsmodels.api as sm
import matplotlib.pyplot as plt
#from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


# load dataset

#data = np.loadtxt('./new_baby .csv', dtype=np.float32, delimiter=',')
#print(data)
dta = pd.read_csv('./new_baby .csv', dtype=np.float32, delimiter=',')
print(dta)