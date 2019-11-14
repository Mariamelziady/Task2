#!/usr/bin/env python
# coding: utf-8

# ## Integrating Datasets and Engineering Features
# 
# This notebook contains examples and code from various sources:
# 
# <a href="github.com/alicezheng/feature-engineering-book">Feature Engineering for Machine Learning Book Code Repository </a>
# <br><a href="github.com/jakevdp/PythonDataScienceHandbook"> Python Data Science Handbook Code Repository </a>
# 
# The goal of this notebook is to illustrate how integration of multiple datasets can be done and how features can be engineered.
# 
# Detailed explanations for important code snippets are provided by Mervat Abuelkheir as part of the CSEN1095 Data Engineering Course.
# 
# Pay attention to the <span style="color:red"> <b> paragraphs in bold red</b></span>; they ask you to do something and provide input!
# 

# For convenience, we will start by redefining the `display()` functionality from the previous section:
# 

# In[2]:


import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)


# ## Part 1: Merging and Joining Datasets
# 
# One essential feature offered by Pandas is its high-performance, in-memory join and merge operations.
# If you have ever worked with databases, you should be familiar with this type of data interaction.
# The main interface for this is the ``pd.merge`` function, and we'll see few examples of how this can work in practice.
# 
# ### Relational Algebra
# 
# The behavior implemented in ``pd.merge()`` is a subset of what is known as *relational algebra*, which is a formal set of rules for manipulating relational data, and forms the conceptual foundation of operations available in most databases.
# The strength of the relational algebra approach is that it proposes several primitive operations, which become the building blocks of more complicated operations on any dataset.
# With this lexicon of fundamental operations implemented efficiently in a database or other program, a wide range of fairly complicated composite operations can be performed.
# 
# Pandas implements several of these fundamental building-blocks in the ``pd.merge()`` function and the related ``join()`` method of ``Series`` and ``Dataframe``s.
# As we will see, these let you efficiently link data from different sources.

# ### Categories of Joins
# 
# The ``pd.merge()`` function implements a number of types of joins: the *one-to-one*, *many-to-one*, and *many-to-many* joins.
# All three types of joins are accessed via an identical call to the ``pd.merge()`` interface; the type of join performed depends on the form of the input data.
# Here we will show simple examples of the three types of merges, and discuss detailed options further below.

# #### One-to-one joins
# 
# Perhaps the simplest type of merge expresion is the one-to-one join, which is in many ways very similar to the column-wise concatenation.
# 
# As a concrete example, consider the following two ``DataFrames`` which contain information on several employees in a company:

# In[3]:


df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
display('df1', 'df2')


# To combine this information into a single ``DataFrame``, we can use the ``pd.merge()`` function. 
# 
# The ``pd.merge()`` function recognizes that each ``DataFrame`` has an "employee" column, and automatically joins using this column as a key.
# The result of the merge is a new ``DataFrame`` that combines the information from the two inputs.
# Notice that the order of entries in each column is not necessarily maintained: in this case, the order of the "employee" column differs between ``df1`` and ``df2``, and the ``pd.merge()`` function correctly accounts for this.
# 
# Additionally, keep in mind that the merge in general discards the index, except in the special case of merges by index (see the ``left_index`` and ``right_index`` keywords, discussed momentarily).

# In[4]:


df3 = pd.merge(df1, df2)
df3


# #### Many-to-one joins
# 
# Many-to-one joins are joins in which one of the two key columns contains duplicate entries.
# For the many-to-one case, the resulting ``DataFrame`` will preserve those duplicate entries as appropriate. 
# Consider the following example of a many-to-one join:

# In[105]:


df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
display('df3', 'df4', 'pd.merge(df3, df4)')


# The resulting ``DataFrame`` will have an aditional column with the "supervisor" information, where the information is repeated in one or more locations as required by the inputs.

# #### Many-to-many joins
# 
# Many-to-many joins are a bit confusing conceptually, but are nevertheless well defined.
# If the key column in both the left and right array contains duplicates, then the result is a many-to-many merge.
# This will be perhaps most clear with a concrete example.
# Consider the following, where we have a ``DataFrame`` showing one or more skills associated with a particular group.
# By performing a many-to-many join, we can recover the skills associated with any individual person:

# In[106]:


df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
display('df1', 'df5', "pd.merge(df1, df5)")


#  In practice, datasets are rarely as clean as the one we worked with above. In the following section we'll consider some of the options provided by `pd.merge()` that enable you to tune how the join operations work.

# ### Specification of the Merge Key
# 
# We've already seen the default behavior of ``pd.merge()``: it looks for one or more matching column names between the two inputs, and uses this as the key.
# However, often the column names will not match so nicely, and ``pd.merge()`` provides a variety of options for handling this.

# #### The ``on`` keyword
# 
# Most simply, you can explicitly specify the name of the key column using the ``on`` keyword, which takes a column name or a list of column names. This option works only if both the left and right ``DataFrame``s have the specified column name.

# In[107]:


display('df1', 'df2', "pd.merge(df1, df2, on='employee')")


# #### The ``left_on`` and ``right_on`` keywords
# 
# At times you may wish to merge two datasets with different column names; for example, we may have a dataset in which the employee name is labeled as "name" rather than "employee".
# In this case, we can use the ``left_on`` and ``right_on`` keywords to specify the two column names:

# In[108]:


df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
display('df1', 'df3', 'pd.merge(df1, df3, left_on="employee", right_on="name")')


# The result has a redundant column that we can drop if desired–for example, by using the ``drop()`` method of ``DataFrame``s:

# In[109]:


pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1)


# #### The ``left_index`` and ``right_index`` keywords
# 
# Sometimes, rather than merging on a column, you would instead like to merge on an index.
# For example, your data might look like this:

# In[110]:


df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
display('df1a', 'df2a')


# You can use the index as the key for merging by specifying the ``left_index`` and/or ``right_index`` flags in ``pd.merge()``:

# In[111]:


display('df1a', 'df2a',
        "pd.merge(df1a, df2a, left_index=True, right_index=True)")


# For convenience, ``DataFrame``s implement the ``join()`` method, which performs a merge that defaults to joining on indices:

# In[112]:


display('df1a', 'df2a', 'df1a.join(df2a)')


# ### Specifying Set Arithmetic for Joins
# 
# In all the preceding examples we have glossed over one important consideration in performing a join: the type of set arithmetic used in the join.
# This comes up when a value appears in one key column but not the other. Consider this example:

# In[113]:


df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']},
                   columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']},
                   columns=['name', 'drink'])
display('df6', 'df7', 'pd.merge(df6, df7)')


# Here we have merged two datasets that have only a single "name" entry in common: Mary.
# By default, the result contains the *intersection* of the two sets of inputs; this is what is known as an *inner join*.
# We can specify this explicitly using the ``how`` keyword, which defaults to ``"inner"``:

# In[114]:


pd.merge(df6, df7, how='inner')


# Other options for the ``how`` keyword are ``'outer'``, ``'left'``, and ``'right'``.
# An *outer join* returns a join over the union of the input columns, and fills in all missing values with NAs:

# In[115]:


display('df6', 'df7', "pd.merge(df6, df7, how='outer')")


# The *left join* and *right join* return joins over the left entries and right entries, respectively.
# For example:

# In[116]:


display('df6', 'df7', "pd.merge(df6, df7, how='left')")


# The output rows now correspond to the entries in the left input. Using
# ``how='right'`` works in a similar manner.

# ### Overlapping Column Names: The ``suffixes`` Keyword
# 
# Finally, you may end up in a case where your two input ``DataFrame``s have conflicting column names.
# Consider this example:

# In[117]:


df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})
display('df8', 'df9', 'pd.merge(df8, df9, on="name", suffixes=["_L", "_R"])')


# ## Example: The US States Data
# 
# Merge and join operations come up most often when combining data from different sources.
# Here we will consider an example of some data about US states and their populations.
# The data files can be found in the data folder.
# 
# Let's take a look at the three datasets, using the Pandas ``read_csv()`` function:

# In[118]:


pop = pd.read_csv('data/state-population.csv')
areas = pd.read_csv('data/state-areas.csv')
abbrevs = pd.read_csv('data/state-abbrevs.csv')

display('pop.head()', 'areas.head()', 'abbrevs.head()')


# Given this information, say we want to compute a relatively straightforward result: rank US states and territories by their 2010 population density.
# We clearly have the data here to find this result, but we'll have to combine the datasets to find the result.
# 
# We'll start with a many-to-one merge that will give us the full state name within the population ``DataFrame``.
# We want to merge based on the ``state/region``  column of ``pop``, and the ``abbreviation`` column of ``abbrevs``.
# We'll use ``how='outer'`` to make sure no data is thrown away due to mismatched labels.

# In[ ]:


merged = pd.merge(pop, abbrevs, how='outer',
                  left_on='state/region', right_on='abbreviation')
merged = merged.drop('abbreviation', 1) # drop duplicate info
merged.head()


# Let's double-check whether there were any mismatches here, which we can do by looking for rows with nulls:

# In[ ]:


merged.isnull().any()


# Some of the ``population`` info is null; let's figure out which these are!

# In[ ]:


merged[merged['population'].isnull()].head()


# It appears that all the null population values are from Puerto Rico prior to the year 2000; this is likely due to this data not being available from the original source.
# 
# More importantly, we see also that some of the new ``state`` entries are also null, which means that there was no corresponding entry in the ``abbrevs`` key!
# Let's figure out which regions lack this match:

# In[ ]:


merged.loc[merged['state'].isnull(), 'state/region'].unique()


# We can quickly infer the issue: our population data includes entries for Puerto Rico (PR) and the United States as a whole (USA), while these entries do not appear in the state abbreviation key.
# We can fix these quickly by filling in appropriate entries:

# In[ ]:


merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
merged.isnull().any()


# No more nulls in the ``state`` column: we're all set!
# 
# Now we can merge the result with the area data using a similar procedure.
# Examining our results, we will want to join on the ``state`` column in both:

# In[ ]:


final = pd.merge(merged, areas, on='state', how='left')
final.head()


# Again, let's check for nulls to see if there were any mismatches:

# In[ ]:


final.isnull().any()


# There are nulls in the ``area`` column; we can take a look to see which regions were ignored here:

# In[ ]:


#final['state'][final['area (sq. mi)'].isnull()].unique()
final.loc[final['area (sq. mi)'].isnull(), 'state'].unique()


# We see that our ``areas`` ``DataFrame`` does not contain the area of the United States as a whole.
# We could insert the appropriate value (using the sum of all state areas, for instance), but in this case we'll just drop the null values because the population density of the entire United States is not relevant to our current discussion:

# In[ ]:


final.dropna(inplace=True)
final.head()


# Let's check if there are any duplicate records in the dataset before we proceed to query it.

# In[ ]:


final.duplicated()


# Thankfully there are no duplicates. Now we have all the data we need. To answer the question of interest, let's first select the portion of the data corresponding with the year 2000, and the total population.
# We'll use the ``query()`` function to do this quickly (this requires the ``numexpr`` package to be installed.

# In[ ]:


get_ipython().system('pip install numexpr')


# In[ ]:


data2010 = final.query("year == 2010 & ages == 'total'")
data2010.head()


# Now let's compute the population density and display it in order.
# We'll start by re-indexing our data on the state, and then compute the result:

# In[ ]:


data2010.set_index('state', inplace=True)
density = data2010['population'] / data2010['area (sq. mi)']


# In[ ]:


density.sort_values(ascending=False, inplace=True)
density.head()


# In[ ]:


density.tail()


# ## <span style="color:red">Assignment Mini Challenge 1</span>
# 
# In the data/microbiome subdirectory, there are 9 spreadsheets of microbiome data that was acquired from RNA sequencing procedures, with a 10th file describing the content of each spreadsheet. 
# 
# <span style="color:red"><b>Write code that imports each of the data spreadsheets and combines them into a single `DataFrame`, adding identifying information from the metadata spreadsheet as columns in the combined `DataFrame`.</b></span>

# In[5]:


meta =pd.read_excel('C:/Users/omara/Desktop/GUC/Year 5/Semester 9/Data Engineerig CSEN1095/New folder/CSEN1095-Data-Engineering-master/data/microbiome/metadata.xls')

meta


# In[6]:


import os
path = os.getcwd()
data_path ='C:/Users/omara/Desktop/GUC/Year 5/Semester 9/Data Engineerig CSEN1095/New folder/CSEN1095-Data-Engineering-master/data/microbiome/'

files = os.listdir(data_path)
files_xls = [f for f in files if f[-3:] == 'xls']

mid1 = pd.DataFrame()
mid2 = pd.DataFrame()
mid3 = pd.DataFrame()
mid4 = pd.DataFrame()
mid5 = pd.DataFrame()
mid6 = pd.DataFrame()
mid7 = pd.DataFrame()
mid8 = pd.DataFrame()
mid9 = pd.DataFrame()
for f in files_xls:
    
        if(f == 'MID1.xls'):
           
           df1 = pd.read_excel(data_path+f)
           mid1["Name"] = df1.iloc[:,0]
           mid1["EXTRACTION_CONTROL"] = df1.iloc[:,1]
            
        elif(f=='MID2.xls'):
            
           df1 = pd.read_excel(data_path+f)
           mid2["Name"] = df1.iloc[:,0]
           mid2["NEC_1_TISSUE"] = df1.iloc[:,1]
        
        elif(f=='MID3.xls'):
           df1 = pd.read_excel(data_path+f)
           mid3["Name"] = df1.iloc[:,0]
           mid3["CONTROL_1_TISSUE"] = df1.iloc[:,1]
                  
        elif(f=='MID4.xls'):
           df1 = pd.read_excel(data_path+f)
           mid4["Name"] = df1.iloc[:,0]
           mid4["NEC_2_TISSUE"] = df1.iloc[:,1]              
                  
        elif(f=='MID5.xls'):
           df1 = pd.read_excel(data_path+f)
           mid5["Name"] = df1.iloc[:,0]
           mid5["CONTROL_2_TISSUE"] = df1.iloc[:,1]            
                  
        elif(f=='MID6.xls'):
           df1 = pd.read_excel(data_path+f)
           mid6["Name"] = df1.iloc[:,0]
           mid6["NEC_1_STOOL"] = df1.iloc[:,1]
                  
        elif(f=='MID7.xls'):
           df1 = pd.read_excel(data_path+f)
           mid7["Name"] = df1.iloc[:,0]
           mid7["CONTROL_1_STOOL"] = df1.iloc[:,1]         
                  
        elif(f=='MID8.xls'):
           df1 = pd.read_excel(data_path+f)
           mid8["Name"] = df1.iloc[:,0]
           mid8["NEC_2_STOOL"] = df1.iloc[:,1]                  
                  
        elif(f=='MID9.xls'):
           df1 = pd.read_excel(data_path+f)
           mid9["Name"] = df1.iloc[:,0]
           mid9["CONTROL_2_STOOL"] = df1.iloc[:,1]
                  
#         path = 'C:/Users/omara/Desktop/GUC/Year 5/Semester 9/Data Engineerig CSEN1095/New folder/CSEN1095-Data-Engineering-master/data/microbiome/'+f
#         data = pd.read_excel(path)
#         df = df.append(data)
# df

final = pd.merge(mid1,mid2)
final = pd.merge(final,mid3)
final = pd.merge(final,mid4)
final = pd.merge(final,mid5)
final = pd.merge(final,mid6)
final = pd.merge(final,mid7)
final = pd.merge(final,mid8)
final = pd.merge(final,mid9)

final


# ### Checking for categorical variables independence using the Chi-Square test
# 
# To illustrate how to use the chi-square test to determine correlation between categorical attributes, let's use another dataset that has more potential for correlated attributes: the Census (Adult Income) Dataset. 

# ### The Adult Income Dataset
# 
# The <a href="https://www.kaggle.com/wenruliu/adult-income-dataset">Adult Income Dataset</a> includes data about an individual’s annual income. Intuitively, income is influenced by the individual’s education level, age, gender, occupation, and etc. The dataset contains 14 columns detailing attributes related to the demographics and other features that describe a person. The target attribute, Income, is divide into two classes: <=50K and >50K. A description of the attributes follows:
# 
# <b>age</b>: continuous.
# <br><b>workclass</b>: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# <br><b>fnlwgt</b>: continuous.
# <br><b>education</b>: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# <br><b>education-num</b>: continuous.
# <br><b>marital-status</b>: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# <br><b>occupation</b>: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# <br><b>relationship</b>: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# <br><b>race</b>: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# <br><b>gender</b>: Female, Male.
# <br><b>capital-gain</b>: continuous.
# <br><b>capital-loss</b>: continuous.
# <br><b>hours-per-week</b>: continuous.
# <br><b>native-country</b>: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# <br><b>income</b>: >50K, <=50K

# Let's import some important modules and then import the data.

# In[7]:


import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

income_df = pd.read_csv("C:/Users/omara/Desktop/GUC/Year 5/Semester 9/Data Engineerig CSEN1095/New folder/CSEN1095-Data-Engineering-master/data/income_data.csv")

# List attribute names
print (list(income_df))

# Display the shape of the dataset (#rows, #columns)
print (income_df.shape)


# You may want to inspect some records from the dataset to get a feel of the attributes and their values.

# In[192]:


#income_df.head() # first 5 rows
income_df.sample(10) # random 10 rows


# Let's explore if there is a correlation with gender and workclass. First, we need to build the contingency matrix for the two attributes:

# In[8]:


contengency_table = pd.crosstab(income_df["workclass"],income_df["gender"], margins= True)
contengency_table


# Let's state the hypotheses:
# 
# <b>Null hypothesis:</b> There is no statistically significant relationship between gender and workclass.
# 
# <b>Alternative hypothesis:</b> There is a statistically significant relationship between the gender and workclass.
# 
# Each cell in the table represents the frequency count for the intersection of both values. The intersection of "male" and "federal-gov" represents the number of men who work in jobs related to the federal government.
# 
# Now let's calculate the chi-square test using SciPy. The `chi2_contengency()` method is applied to a two dimensional array representing the actual attribute values, and it automatically computes the contingency matrix and outputs three numbers: the chi2 value (difference between observed and expected counts), the p-value, and the degrees of freedom. What you want to check is the second value outputted, which is the p-value. You want it to be smaller than 0.05 so that the null hypothesis can be rejected.   

# In[9]:


# Take the previously produced contingency matrix and apply the chi2 test method to it
#st.chi2_contingency(_)
st.chi2_contingency(contengency_table)


# <span style="color:red"><b>Do we reject the null hypothesis based on the previous results?</b></span>
# 
# <span style="color:red"><b>In the previous output, and based on your understanding of the chi2 test from the lecture, are the degrees of freedom produced correct? If no, explain why and fix the problem and recompute the chi2 test.</b></span>
# 

# ## <span style="color:red">Assignment Mini Challenge 2</span>
# 
# <span style="color:red"><b>Write code that will compute the correlation between the `race` and `education` attributes. Is there a correlation or are the attributes independent?</b></span>

# In[21]:


contengency_table = pd.crosstab(income_df["race"],income_df["education"])
contengency_table

st.chi2_contingency(contengency_table)


# ## Part 2 - Engineering Features
# 
# 
# 
# 
# 

# Feature engineering is not a deterministic task, but some prelimenary things can still be done in a straigtforward way. In the following cells we will perform encoding, discretization, and aggregation tasks as part of feature engineering.
# 
# We are still using the Adult Income Dataset. Let's investigate descriptions of individual attributes.

# In[11]:


def summerize_data(df):
    for column in df.columns:
        print (column)
        if df.dtypes[column] == np.object: # Categorical data
            print (df[column].value_counts())
        else:
            print (df[column].describe()) 
            
        print ('\n')
    
summerize_data(income_df)


# You can use the `OneHotCategoricalEncoder()` method to perform one-hot encoding, but you need to install the feature-engine package using:
# 
#     pip install feature-engine
# 
# at the command prompt. Let's try this method.

# In[ ]:


get_ipython().system('pip install feature-engine')


# In[13]:


# Try one hot encoding

# Copy the original data
encoded_income_df = income_df.copy()

# Select the numeric columns
numeric_subset = income_df.select_dtypes('number')
# Select the categorical columns
categorical_subset = income_df.select_dtypes('object')
# Import feature-engine
from feature_engine.categorical_encoders import OneHotCategoricalEncoder

# Define encoder method
encoder = OneHotCategoricalEncoder()

# Apply encoder to categorical subset, except income
onehot_categorical_subset = encoder.fit_transform(categorical_subset[categorical_subset.columns.drop("income")])

# Beauty of feature-engine encoder method is it already returns a dataframe, so we need not worry about conversions
# Concatenate and reconstruct new dataset
onehot_encoded_data = pd.concat([numeric_subset, onehot_categorical_subset, income_df["income"]], axis = 1)

# display sample 10 records
onehot_encoded_data.sample(10)


# #### Discretization 
# 
# Now let's discretize some of the numerical attributes. We will work with the <b>age</b> and the <b>hours-per-week</b> as examples.

# In[14]:


# Group the "age" column
age_group = [] # define array structure
for age in encoded_income_df["age"]:
    if age < 25:
        age_group.append("<25")
    elif 25 <= age <= 34:
        age_group.append("25-34")
    elif 34 < age <= 44:
        age_group.append("35-44")
    elif 44 < age <= 54:
        age_group.append("45-54")
    elif 54 < age <= 65:
        age_group.append("55-64")
    else:
        age_group.append("65 and over")
        
# Copy dataframe to keep original 
new_income_df = encoded_income_df.copy()
new_income_df["age_group"] = age_group
del new_income_df["age"]

# Same thing for "hours-per-week"
work_hours_per_week = []
for hours in encoded_income_df["hours-per-week"]:
    if hours < 16:
        work_hours_per_week.append("<16")
    elif 16 <= hours <= 32:
        work_hours_per_week.append("16-32")
    elif 32 < hours <= 48:
        work_hours_per_week.append("32-48")
    elif 48 < hours <= 60:
        work_hours_per_week.append("48-60")
    else:
        work_hours_per_week.append("60 and over")
        
new_income_df["work_hours_per_week"] = work_hours_per_week
del new_income_df["hours-per-week"]

new_income_df.head(10)


# There is an easier way in python than using if else statements. Pandas `cut` function can be used to group continuous or countable data in to bins.

# In[15]:


# Take another copy of the original dataset
new_income_df2 = encoded_income_df.copy()

# Cut the age attribute into intervals
# age_group2 = pd.cut(new_income_df2.age, [20,40,60,80])

# You can label the intervals for more meaningful representation
age_group2 = pd.cut(new_income_df2.age, [20,40,60,80],labels=['young','middle-aged','senior'])

new_income_df2["age_group"] = age_group2
del new_income_df2["age"]

new_income_df2.head(10)


# ## <span style="color:red">Assignment Mini Challenge 3</span>
# 
# It is possible to use `cut` with quantiles instead of intervals (depth-based binning, # of ovservations in bin instead of interval of values). Read relevant documentation of the method on the Web.
# 
# In the Adult Income dataset, there is an `age` attribute, which is numerical.
# 
# <span style="color:red"><b>Write code that will do the following:</b></span>
#     
# <span style="color:red"><b>1- Apply quantile cutting to the `age` attribute and construct new dataset.</b></span>
# 
# <span style="color:red"><b>2- Apply one-hot encoding again to the discretized attribute.</b></span>

# In[16]:


dfeq = encoded_income_df.copy()
dfcut = pd.qcut(dfeq.age,q=6 )

dfeq["age_group"] = dfcut
del dfeq["age"]



# for col in dfeq.columns: 
#     print(col) 

numeric_subset = dfeq.select_dtypes('number')


df1 = dfeq.select_dtypes('object')
df2 = dfeq.select_dtypes('category').astype(object)

df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)

categorical_subset = pd.concat([df1,df2] , axis=1)
# print(categorical_subset)


# Import feature-engine
from feature_engine.categorical_encoders import OneHotCategoricalEncoder

# Define encoder method
encoder = OneHotCategoricalEncoder()

# Apply encoder to categorical subset, except income
onehot_categorical_subset = encoder.fit_transform(categorical_subset[categorical_subset.columns.drop("income")])
# print(onehot_categorical_subset)


onehot_encoded_data = pd.concat([numeric_subset, onehot_categorical_subset, income_df["income"]], axis = 1)
# for col in onehot_encoded_data.columns: 
#     print(col) 

# display sample 10 records

onehot_encoded_data



# ## <span style="color:red">Assignment Mini Challenge 4</span>
# 
# In the Adult Income dataset, there is an `hours_per_week` attribute, which is numerical.
# 
# <span style="color:red"><b>Write code that will do the following:</b></span>
# 
# <span style="color:red"><b>1- Transform this attribute into a categorical attribute by dividing the range of values into bins with the following lables: "0-9", "10-19", "20-29", "30-39", "40-49", "50+". Rename the new attribute as `working_hours_categories`. </b></span>
# 
# <span style="color:red"><b>2- Perform the chi2 test to find if the discretized attribute `working_hours_categories` and `gender` are correlated.</b></span>
# 

# In[22]:


dfe = encoded_income_df.copy()

work_hours_per_week = []
for hours in dfe["hours-per-week"]:
    if hours <= 9:
        work_hours_per_week.append("0-9")
    elif 10 <= hours <= 19:
        work_hours_per_week.append("10-19")
    elif 20 <= hours <= 29:
        work_hours_per_week.append("20-29")
    elif 30 <= hours <= 39:
        work_hours_per_week.append("30-39")
    elif 40 <= hours <= 49:
        work_hours_per_week.append("40-49")
    else:
        work_hours_per_week.append("50+")
    
  
dfe["working_hours_categories"] = work_hours_per_week
del dfe["hours-per-week"]

contengency_table = pd.crosstab(dfe["working_hours_categories"],dfe["gender"])
st.chi2_contingency(contengency_table)



# 

# #### Aggregation
# 
# Now let's try to aggregate numerical values according to a categorical attribute. If there was a time attribute then aggregation could have been performed on different time intervals. For the dataset we have it is sufficient to apply aggregation over the workclass attribute for now. We will use the `groupby` function. Then, it is possible to compute aggregate values (e.g. mean) per workclass group for the numerical attributes.

# In[18]:


# We will work with the original dataset income_df
# Group workclass attribute by its categorical values
grouped_income = income_df.groupby(["workclass"])

# Compute mean per group using agg function
grouped_income.agg(np.mean).head()


# The `agg` function intelligently ignores categorical attributes.

# ## <span style="color:red">Assignment Mini Challenge 5</span>
# 
# <span style="color:red"><b>Why did we not apply the aggregation function on the encoded dataset?</b></span>
# 
# ``GroupBy`` has its own associated ``aggregate()`` method which allows for more flexibility and aggregate different columns using different aggregation functions in one shot. Look up how this method works, 
# 
# <span style="color:red"><b>Write code that uses the ``aggregate()`` to aggregate the `age` column by mean, the `fnlwgt` column by min, the `education-num` by max, the `capital-gain` by max, the `capital-loss` by max, and the `hours-per-week` by mean. </b></span>

# In[19]:


dfagg = income_df.copy()

grouped_multiple= dfagg.groupby("workclass").agg({
                  'age':['mean'],'fnlwgt':['min'],'capital-gain':['max']
                   ,'capital-loss':['max'],'hours-per-week':['mean'],'educational-num':['max'],})

grouped_multiple


# ## <span style="color:red">Assignment Bonus Challenge</span>
# 
# <span style="color:red"><b>For the Adult Income Dataset:</b></span>
# 
# <span style="color:red"><b>1- Try to come up with at least one indicator feature based on threshold, multiple attributes, or multiple categorical values in an attribute</b></span>
# 
# <span style="color:red"><b>2- Try to come up with at least one meaningful interaction feature.</b></span>

# In[ ]:




