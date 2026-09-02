# %% [markdown]
# #### Exercise 2
# Ahmet Meriç Kızıltaş
# 

# %% [markdown]
# In this exercise we are going to use pandas and seaborn to visualize data related to a kidney disease dataset. We begin by importing the libraries

# %%
!mamba install pandas
!mamba install seaborn
!mamba install xlrd

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp

# %% [markdown]
# a) Read the dataset using pandas’ read_excel function. Use pandas.melt to transform the data
# from “wide” to “long” format, using class (indicating ckd for chronic kidney disease or notckd
# for its absence) as the identifier variable. (2P)

# %%
kidney_data = pd.read_excel("chronic_kidney_disease_numerical.xls", index_col = 0)
kidney_data.head()
kidney_data = pd.melt(kidney_data, id_vars = "class")

# %%
kidney_data.head()

# %% [markdown]
# b) For each numerical attribute, such as age or blood pressure, create two boxplots side-by-side. One
# should show the attribute’s distribution among patients suffering from chronic kidney disease, the
# other one from patients who do not suffer from the disease. Hint: Due to the different numerical
# ranges, you will have to disable sharing of y axes between plots of different attributes. (4P)

# %%
#fig, ax = plt.subplots(4, len(kidney_data["variable"].unique())//4 + 1)
sns.set_palette("PRGn")

for i in range(len(kidney_data["variable"].unique())):
    plt.figure(figsize = (3, 3))
    sns.boxplot(data = kidney_data[kidney_data["variable"] == kidney_data["variable"].unique()[i]], hue = "class", y = "value")
    plt.title(kidney_data["variable"].unique()[i])
    plt.show()



# %% [markdown]
# c) Based on viewing the plots, name an attribute that appears to be highly indicative of chronic
# kidney disease, and one that seems to be mostly unrelated to it. (1P)

# %% [markdown]
# Observing the plots, we see that the attributes *albumin* and *sugar* are highly sensitive to kidney disease (in fact, every single person that doesn't suffer from it observes a 0 in both categories). Meanwhile, *potassium* levels seem to not be affected by this disease.


