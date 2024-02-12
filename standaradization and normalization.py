# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:23:04 2023

@author: Admin
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
d=pd.read_csv("C:/Datasets/mtcars_dup.csv.xls")
d.describe()
a=d.describe()

#Initalize the scalar
scalar=StandardScaler()
df=scalar.fit_transform(d)
dataset=pd.DataFrame(df)
res=dataset.describe()
#Here if you will chech res,in variable envrionment then

##########################################################3
#Normalization
ethnic=pd.read_csv("C:/Data Science/Datasets/ethnic diversity.csv")
#Now read columns
ethnic.columns
#There are some col which not useful,we need to drop them
ethnic.drop(['Employee_Name','EmpID','Zip'],axis=1,inplace=True)
#Now read min value and maxx values of salaries and aage
a1=ethnic.describe()
#Check all data frame in varaible explore
#You fine min salary is 0 and maz is 108304
#Same way check for age,there is huge difference
#first we will have to cinvert non numeric data to label encoding
ethnic=pd.get_dummies(ethnic,drop_first=True)
#Normalization function written where ethnic argument is passed
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm_func(ethnic)
b=df_norm.describe()
#If you will oberseved the b frame
#It has the dimenasion 8,81
#Earlier in a they were 8,11 it is beacuse all non numeric
#data has been converted to numeric using label encoding