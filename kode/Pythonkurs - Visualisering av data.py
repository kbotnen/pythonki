#!/usr/bin/env python
# coding: utf-8

# # Datasett - randomisert

# In[ ]:


# Load the numpy module with alias np for easy reference later.
import numpy as np # pip install numpy or conda install -c conda-forge numpy

# Load the plotly express module with alias px for easy reference later.
import plotly.express as px # pip install plotly or conda install -c conda-forge plotly


# In[ ]:


x_data = np.random.randint(0, 10, size=10) # Use NumPy to generate random data for the x-axis.
y_data = np.random.randint(0, 10, size=10) # Use NumPy to generate random data for the y-axis.
color_data = x_data                        # A random array-like list of colors.

print(type(x_data)) # Inspect the datatype, its 'array-like'.
print(x_data)       # Inspect the data.

print(type(y_data)) # Inspect the datatype, its 'array-like'.
print(y_data)       # Inspect the data.


# In[ ]:


# Plot a barplot with plotly express.
fig = px.bar(
    x=x_data, # Values from this column or array_like are used to position marks along the x axis.
    y=y_data, # Values from this column or array_like are used to position marks along the y axis.
    color=color_data, # It will work perfectly fine without the colors too.
) 

fig.show()


# Dokumentasjon av plotly: [https://plotly.com/python-api-reference/index.html](https://plotly.com/python-api-reference/index.html)
# 
# Dokumentasjon av pandas.dataframe: [https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)

# # Datasett - Gapminder
# 
# Gapminder identifies systematic misconceptions about important global trends and proportions and uses reliable data to develop easy to understand teaching materials to rid people of their misconceptions.
# 
# Gapminder is an independent Swedish foundation with no political, religious, or economic affiliations.
# 
# Les mer om Gapminder: [https://www.gapminder.org/data/](https://www.gapminder.org/data/)
# 
# Flere datasett: [https://open-numbers.github.io/](https://open-numbers.github.io/)

# In[ ]:


# Use the built-in sample dataset gapminder.
df = px.data.gapminder() # Read the dataset into a variable called df, of type Pandas.DataFrame.
df.head(15)              # Use the Pandas.DataFrame built-in method to list the 15 first rows of our DataFrame.


# ### Datautsnitt: year == 2007, type: scatter

# In[ ]:


# Use Plotly Express's scatter method to create a scatter plot.
# Provide the data frame and specify columns for the axies.

fig = px.scatter(
    df.query("year == 2007"), # use Pandas.DataFrame.Query() method to retrieve only 2007 data.
    x="gdpPercap",            # gdpPercap is the column name for GDP per Capita.
    y="lifeExp"               # lifeExp is the column name for Life Expectancy.
)

fig.show()                    # Render the plot.


# In[ ]:


# Use Plotly Express's scatter method to create a scatter plot.
# Provide the data frame and specify columns for the axies.
# Added parameters to make the plot more interesting.

fig = px.scatter(
    df.query("year == 2007"), # use Pandas.DataFrame.Query() method to retrieve only 2007 data.
    x="gdpPercap",            # gdpPercap is the column name for GDP per Capita.
    y="lifeExp",              # lifeExp is the column name for life expectancy.
    size="pop",               # size of markers are based on the population value.
    color="lifeExp",          # color of the markers are assigned based on lifeExp value.
    hover_name="country"      # if we hover the mouse over the markers, show the country name.
)

fig.show()                    # Render the plot.


# ### Datautsnitt: continent == Europe, type: line

# In[ ]:


europe_data = px.data.gapminder().query("continent=='Europe'")
print(europe_data)


# In[ ]:


# Use Plotly Express's scatter method to create a line plot.
# Provide the data frame and specify columns for the axies.
fig = px.line(
    europe_data,    # use a variable that contains a Pandas.DataFrame.
    x="year",       # year column.
    y="lifeExp",    # life expectancy column.
    color="country" # color of the markers are assigned based on country value.
)

fig.show()          # Render the plot.


# In[ ]:


# Use Plotly Express's scatter method to create a line plot.
# Provide the data frame and specify columns for the axies.
fig = px.line(
    europe_data,    # use a variable that contains a Pandas.DataFrame.
    x="year",       # year column.
    y="pop",        # population column.
    color="country" # color of the markers are assigned based on country value.
)

fig.show()          # Render the plot.


# ### Datautsnitt: country == norway, type: bar

# In[ ]:


norway_data = px.data.gapminder().query("country=='Norway'")
print(norway_data)


# In[ ]:


# Use Plotly Express's scatter method to create a bar plot.
# Provide the data frame and specify columns for the axies.
fig = px.bar(
    norway_data,                 # use a variable that contains a Pandas.DataFrame.
    x="year",                    # year column.
    y="pop",                     # population column.
    color="lifeExp",             # color of the markers are assigned based on life expectancy value.
    color_continuous_scale="ice" # build a continuous color scale.
)

fig.show()                       # Render the plot.


# ### Datautsnitt: country == europe, year == 2007, type: pie

# In[ ]:


# Use Plotly Express's scatter method to create a pie chart.
# Provide the data frame and specify needed options.
fig = px.pie(
    df.query("year==2007").query("continent=='Europe'"), # use Pandas.DataFrame.Query() method to retrieve 2007 && europe data.
    values="pop",                                        # used to set values associated to sectors.
    names="country",                                     # used as labels for sectors.
    title="Europe's Population Distribution"             # the figure title.
)

fig.show()                                               # Render the plot.


# ### Datautsnitt: gapminder, type: animated bubble / scatter

# In[ ]:


# Hans Rosling style of visualisation. TED 2006, data-bubble software.
fig = px.scatter(
    px.data.gapminder(),       # use the whole dataset / same as df.
    x="gdpPercap",             # gdpPercap is the column name for GDP per Capita.
    y="lifeExp",               # lifeExp is the column name for life expectancy.
    animation_frame="year",    # this column are used to assign marks to animation frames.
    animation_group="country", # matching rows will be treated as if they describe the same object in each frame.
    size="pop",                # values from this column are used to assign mark sizes.
    color="country",           # values from this column are used to assign color to marks.
    hover_name="country",      # values from this column appear in bold in the hover tooltip.
    log_x = True,              # the x-axis is log-scaled in cartesian coordinates.
    size_max=45,               # maximum mark size when using size.
    range_x=[100,100000],      # overrides auto-scaling on the x-axis in cartesian coordinates.
    range_y=[25,90]            # overrides auto-scaling on the y-axis in cartesian coordinates.
)

fig.show()


# # Lagre plot

# In[ ]:


# pip install -U kaleido
# conda install -c conda-forge python-kaleido

import os

# Lets create a folder to put our exported plots in.
if not os.path.exists("image_export"):
    os.mkdir("image_export")
    
fig.write_image("image_export/plot_1.png") # Export as static image.


# In[ ]:


import plotly.offline as ploff # Import more functionality from plotly package.

ploff.offline.plot(fig, filename='image_export/plot_1.html') # Export as dynamic webpage.


# # Datasett - Kaggle nettsamfunn
# 
# Nettsiden: [https://www.kaggle.com/](https://www.kaggle.com/) inneholder et stort utvalg av datasett innenfor områder som finans, klima, utdanning med mer. I tillegg til datasett vil du også finne online-kurs, andre sine jupyter notebooks, forum med mer.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt # pip install matplotlib
import opendatasets as od # % pip install opendatasets

od.download('https://www.kaggle.com/ajaypalsinghlo/world-happiness-report-2021') # Prereq: Prepare your API-user/pw on kaggle.com


# In[ ]:


df = pd.read_csv('world-happiness-report-2021/world-happiness-report-2021.csv')
df.head(5)


# # Oppsummering
# 
# Da har vi plottet data v.h.a plotly. Vi har også brukt forskjellige datasett, fra de innebygde testsettene i plotly, til datasett som vi har hentet fra nettsamfunnet kaggle.com.

# In[ ]:




