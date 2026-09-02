# %% [markdown]
# #### Exercise 1
# Ahmet Meriç Kızıltaş

# %% [markdown]
# Plotly is a plotting library with bindings for different programming languages. It enables the creation
# of interactive plots based on HTML, CSS and javascript. An introduction to the parallel coordinates
# plot with plotly can be found here: https://plotly.com/python/parallel-coordinates-plot/
# In this exercises, we will work with a Mice Protein Expression Dataset, which contains expression levels
# of 77 proteins, measured in the cerebral cortex of 8 classes of mice. The classes result from two genotypes
# (Ts65Dn, which serves as a mouse model of human down syndrome, vs. normal controls), two treatments
# (injection of the drug memantine vs. a saline solution as a control), and two experimental conditions
# related to context fear conditioning (context-shock, which should lead to learning, vs. shock-context, in
# which no learning takes place). Counting all repeated measurements, there are 1080 samples overall,
# some with missing data. You can find more information on the data in the corresponding publication.

# %%
#We begin by installing plotly
!mamba install plotly
!mamba install pandas
!mamba install xlrd
!mamba install nbformat

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd
import plotly.express as px

# %%
!mamba list nbformat

# %% [markdown]
# **a)** *Use pandas to read the file Data_Cortex_Nuclear.xls provided via eCampus. Extract subgroups
# t-CS-s and c-CS-s. How many mice were measured for each class? (1P)*
# 

# %%
data = pd.read_excel("Data_Cortex_Nuclear.xls", index_col = 0)
data.head()

# %%
data = pd.concat([data[(data["class"] == 'c-CS-s')],data[(data["class"] == 't-CS-s')]])
data["class"].value_counts()

# %% [markdown]
# There are 135 measurements of mouses of the class *c-CS-s* and 105 of the class *t-CS-s*.

# %% [markdown]
# **b)** *Create a parallel coordinates plot with plotly from the following 5 proteins: (pPKCG N, pP70S6 N,
# pS6 N, pGSK3B N, ARC N). Assign different colors to the two selected classes. Annotate every
# axis with the correct protein name. (9P)*

# %%
data.insert(0, "class_id", [(0 if i == 'c-CS-s' else 1) for i in data["class"]])

# %%
fig = px.parallel_coordinates(data, color="class_id",
                              dimensions=['pPKCG_N', 'pP70S6_N','pS6_N', 'pGSK3B_N','ARC_N'],
                              color_continuous_scale=px.colors.diverging.Tealrose)
fig.show()

# %% [markdown]
# **c)** *Explore the data by interacting with the parallel coordinates plot. Do you find anything suspicious
# about the data set? (1P)*

# %% [markdown]
# The data set looks fairly normal to me. We see the data points evenly distributed around the range of the variables, and even though the subgroups have different distributions, the correlation between variables seems to be the same in all cases(positive, negative, positive, positive). We have some apparent outliers: an individual on group 1 with low values on all proteins, another individual on group 1 that has a particularly high value on protein pS6_N, and 2 individuals on group 0 with high values on proteins pS6_N, pGSK3B_N and ARC_N


