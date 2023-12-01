#!/usr/bin/env python
# coding: utf-8

# # Les inn en Excelfil v.h.a Python
# 
# Vi starter med å vise hvordan vi kan lese inn en tilfeldig Excelfil.

# In[ ]:


import pandas as pd
# Remember: 'conda install openpyxl' before importing xlsx

# Read all sheets into a dictionary, sheet_name=None results in all sheets being read.
all_sheets = pd.read_excel("../data/excel/KPI_tabell.xlsx", sheet_name=None)


# In[ ]:


print(type(all_sheets)) # Verify what datastructure we have.
print(all_sheets.keys()) # Let's get the name of the sheets.


# In[ ]:


for item in all_sheets.values(): # Let us inspect the results.
    print(type(item))
    print(item)
    print("---")


# Vi ser at vi nå har lest inn en excelfil som har to ark i seg. Hvert av de to arkene er represent som en Pandas DataFrame, og er omkapslet av en Dictionary. La oss droppe den overordnede strukturen og få direkte tilgang til hvert av de to arkene.

# In[ ]:


df_current = all_sheets.get("Recent", "Not found")
df_history = all_sheets.get("History", "Not found")


# In[ ]:


df_current


# In[ ]:


df_history


# In[ ]:


df_current.rename(columns={"Unnamed: 0":"Årstall"}, inplace=True)
df_history.rename(columns={"Unnamed: 0":"Årstall"}, inplace=True)


# In[ ]:


df_current


# In[ ]:


df_history


# Siden dette er ferdigtygget materiale så finner vi et annet datasett som vi kan jobbe med.

# # Enkel statistikk og pivot

# In[ ]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)

import opendatasets as od # % pip install opendatasets

od.download("https://www.kaggle.com/datasets/eringill/2023-american-vehicle-prices/metadata") # Prereq: Prepare your API-user/pw on kaggle.com


# In[ ]:


bilpriser = pd.read_csv('2023-american-vehicle-prices/prices_clean.csv')
bilpriser


# In[ ]:


bilpriser.groupby("make").price.min()


# In[ ]:


bilpriser.groupby("make").price.max()


# In[ ]:


bilpriser.groupby("make").price.mean()


# In[ ]:


bilpriser.groupby("make").describe()


# En av funksjonene som brukes ofte i Excel er pivottabeller. Selv om vi på en måte har gjort det samme i eksemplene over ved hjelp av groupby() så har Pandas en egen pivot_table() metode vi kan se på og.

# In[ ]:


# Aggregate on "Symbol". The default aggregation function is 'mean'.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot_table.html
pivot = pd.pivot_table(data=bilpriser, index='make', values=["price"])
pivot


# In[ ]:


# Aggreagate the sum instead of the default mean.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot_table.html
pivot = pd.pivot_table(data=bilpriser, index='make', values=["price"], aggfunc='sum')
pivot


# In[ ]:


# Use multiple aggregate functions.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot_table.html
pivot = pd.pivot_table(data=bilpriser, index='make', values=["price"], aggfunc=['mean', 'sum'])
pivot


# In[ ]:


# Different aggregate functions for each column.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot_table.html
pivot = pd.pivot_table(data=bilpriser, index='make', values=["price", "year"], aggfunc={"price": 'mean', "year": 'count'})
pivot


# In[ ]:


# Adding totals to our pivot.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot_table.html
pivot = pd.pivot_table(data=bilpriser, index='make', values=["price", "year"], aggfunc={"price": 'mean', "year": 'count'}, margins=True)
pivot


# In[ ]:


# Sort our result by chaining the sort_values().
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot_table.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html
pivot = pd.pivot_table(data=bilpriser, index='make', values=["price", "year"], aggfunc={"price": 'mean', "year": 'count'}, margins=True).sort_values(by="year")
pivot


# In[ ]:


pivot.to_clipboard(excel=True) # We can copy our dataframe to the clipboard!


# # Skrive til en Excelfil v.h.a Python
# 
# Vi har lest inn en fil, sett litt på innholdet og nå gjenstår det kun å skrive innhold tilbake til filen.

# In[ ]:


bilpriser.to_excel("Pythonkurs - Python og excel.xlsx", sheet_name='Bilpriser')


# In[ ]:


# Write our pivots to Excel.
original_pivot = pivot.copy(deep=True)
pivot.to_excel("Pythonkurs - Del 3 - Python og excel.xlsx", sheet_name='Bilpriser pivot')


# In[ ]:


# Write all data to csv, which does not have the same restrictions as xlsx.
bilpriser.to_csv("Pythonkurs - Python og excel.csv")


# In[ ]:


# Add a new column with new data.
pivot["Mine prisforventninger 2024"] = ((5 * pivot["price"])/100) + pivot["price"]
pivot


# In[ ]:


# Write our updated DataFrame to Excel, we want it as a new Sheet.
pivot.to_excel("Pythonkurs Python og excel.xlsx", sheet_name='Bilpriser pivot forventninger')


# Vi ser at vi ikke har fått et nytt faneark, men overskrevet det eksisterende. For å få lagt til et nytt faneark må vi trikse litt.

# In[ ]:


with pd.ExcelWriter("Pythonkurs - Python og excel.xlsx") as writer:
    original_pivot.to_excel(writer, sheet_name='Bilpriser pivot')
    pivot.to_excel(writer, sheet_name='Bilpriser pivot forventninger')


# Da har vi sett på hvordan vi kan lese, manipulere og skrive Excelfiler v.h.a Python.
