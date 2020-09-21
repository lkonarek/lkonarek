---
layout: post
title: Analysis of Suicide Rates since 1985 
---

Thursday September 10th marked World Suicide Prevention Day, a day where awareness is brought to one of the most prevalent and preventable causes of death: suicide. Our society has a strange relationship with suicide. We all know that it is a problem, however we rarely talk about it or acknowledge it as if that may give people ideas. I believe by investigating more of what causes suicides and why people ultimately feel the need to take their own life we will prevent these tragedies far more effectively. Although this analysis will not even begin to answer why people commit suicide I aim to identify: 
- If suicides are going up year over year
- The genders most likely to commit suicide 
- The age ranges that typically commit suicide at higher rates 
- If richer countries commit suicide at lower rates

### Introduction
Before we answer any of the questions above, we must first import the necessary packages and our dataset:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns; sns.set()
```


```python
df = pd.read_csv('data/master.csv')
```


```python
# Let's first get a brief overview of all the information that the table provides:
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>sex</th>
      <th>age</th>
      <th>suicides_no</th>
      <th>population</th>
      <th>suicides/100k pop</th>
      <th>country-year</th>
      <th>HDI for year</th>
      <th>gdp_for_year ($)</th>
      <th>gdp_per_capita ($)</th>
      <th>generation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Albania</td>
      <td>1987</td>
      <td>male</td>
      <td>15-24 years</td>
      <td>21</td>
      <td>312900</td>
      <td>6.71</td>
      <td>Albania1987</td>
      <td>NaN</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>Generation X</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Albania</td>
      <td>1987</td>
      <td>male</td>
      <td>35-54 years</td>
      <td>16</td>
      <td>308000</td>
      <td>5.19</td>
      <td>Albania1987</td>
      <td>NaN</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>Silent</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Albania</td>
      <td>1987</td>
      <td>female</td>
      <td>15-24 years</td>
      <td>14</td>
      <td>289700</td>
      <td>4.83</td>
      <td>Albania1987</td>
      <td>NaN</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>Generation X</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Albania</td>
      <td>1987</td>
      <td>male</td>
      <td>75+ years</td>
      <td>1</td>
      <td>21800</td>
      <td>4.59</td>
      <td>Albania1987</td>
      <td>NaN</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>G.I. Generation</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Albania</td>
      <td>1987</td>
      <td>male</td>
      <td>25-34 years</td>
      <td>9</td>
      <td>274300</td>
      <td>3.28</td>
      <td>Albania1987</td>
      <td>NaN</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>Boomers</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 27820 entries, 0 to 27819
    Data columns (total 12 columns):
    country               27820 non-null object
    year                  27820 non-null int64
    sex                   27820 non-null object
    age                   27820 non-null object
    suicides_no           27820 non-null int64
    population            27820 non-null int64
    suicides/100k pop     27820 non-null float64
    country-year          27820 non-null object
    HDI for year          8364 non-null float64
     gdp_for_year ($)     27820 non-null object
    gdp_per_capita ($)    27820 non-null int64
    generation            27820 non-null object
    dtypes: float64(2), int64(4), object(6)
    memory usage: 2.5+ MB


Therefore the main data that the dataset includes is:
- The Country name 
- The year 
- The sex/gender 
- Different age profiles 
- The population, 
- And lastly, some of the numerical and financial figures of each country. 

From looking at these table names we can see for one that the column: 'country-year' is redundant as that information is already included in the two seperate columns of country and year. 


```python
# These are the different age groups that are included. As we can see there are different age discrepancies and 
# it is likely that some countries do not include suicides for anything younger than 15, judging from the fact that 
# 5-14 Years has 32 less values:
print(df['age'].value_counts())

# This is is also inline with the generation column (or at least it should be...): 
print()
print(df['generation'].value_counts())
```

    35-54 years    4642
    15-24 years    4642
    55-74 years    4642
    25-34 years    4642
    75+ years      4642
    5-14 years     4610
    Name: age, dtype: int64
    
    Generation X       6408
    Silent             6364
    Millenials         5844
    Boomers            4990
    G.I. Generation    2744
    Generation Z       1470
    Name: generation, dtype: int64



```python
# Country data for the following:
df['country'].value_counts()
```




    Austria                   382
    Mauritius                 382
    Iceland                   382
    Netherlands               382
    Republic of Korea         372
                             ... 
    Bosnia and Herzegovina     24
    Macau                      12
    Cabo Verde                 12
    Dominica                   12
    Mongolia                   10
    Name: country, Length: 101, dtype: int64


Interestingly, there are no null values present at all other then HDI per year (Human Development Index), therefore this dataset is largely already cleaned:

```python

percent_missing = df.isnull().sum() * 100 / len(df)
null_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
null_df[null_df['percent_missing'].gt(0)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column_name</th>
      <th>percent_missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>HDI for year</td>
      <td>HDI for year</td>
      <td>69.935298</td>
    </tr>
  </tbody>
</table>
</div>



However, judging from the fact that this column (HDI for year) is ~70% null, it doesnt provide a lot of value as it currently is. We can choose to get rid of the feature entirely, which would make some sense given that it is almost entirely null, however since it is a continuous value, we can use the mean to fill in. 

Be advised that this normally would be a pretty bad way to fill in the values when approximately 70% is missing/null, however since there is only one feature with null values and it will be interesting to look at the relationship of how 'HDI for year' affects the results.  

Now before we make other changes let's first make sure all the columns are recording the data correctly.

```python

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 27820 entries, 0 to 27819
    Data columns (total 12 columns):
    country               27820 non-null object
    year                  27820 non-null int64
    sex                   27820 non-null object
    age                   27820 non-null object
    suicides_no           27820 non-null int64
    population            27820 non-null int64
    suicides/100k pop     27820 non-null float64
    country-year          27820 non-null object
    HDI for year          8364 non-null float64
     gdp_for_year ($)     27820 non-null object
    gdp_per_capita ($)    27820 non-null int64
    generation            27820 non-null object
    dtypes: float64(2), int64(4), object(6)
    memory usage: 2.5+ MB


Rename the columns with '($)' in them for better interpretation:

```python

df.rename(columns={' gdp_for_year ($) ': 'gdp_for_year', 
                  'gdp_per_capita ($)': 'gdp_per_capita'}, inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>sex</th>
      <th>age</th>
      <th>suicides_no</th>
      <th>population</th>
      <th>suicides/100k pop</th>
      <th>country-year</th>
      <th>HDI for year</th>
      <th>gdp_for_year</th>
      <th>gdp_per_capita</th>
      <th>generation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Albania</td>
      <td>1987</td>
      <td>male</td>
      <td>15-24 years</td>
      <td>21</td>
      <td>312900</td>
      <td>6.71</td>
      <td>Albania1987</td>
      <td>NaN</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>Generation X</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Albania</td>
      <td>1987</td>
      <td>male</td>
      <td>35-54 years</td>
      <td>16</td>
      <td>308000</td>
      <td>5.19</td>
      <td>Albania1987</td>
      <td>NaN</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>Silent</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Albania</td>
      <td>1987</td>
      <td>female</td>
      <td>15-24 years</td>
      <td>14</td>
      <td>289700</td>
      <td>4.83</td>
      <td>Albania1987</td>
      <td>NaN</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>Generation X</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Albania</td>
      <td>1987</td>
      <td>male</td>
      <td>75+ years</td>
      <td>1</td>
      <td>21800</td>
      <td>4.59</td>
      <td>Albania1987</td>
      <td>NaN</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>G.I. Generation</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Albania</td>
      <td>1987</td>
      <td>male</td>
      <td>25-34 years</td>
      <td>9</td>
      <td>274300</td>
      <td>3.28</td>
      <td>Albania1987</td>
      <td>NaN</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>Boomers</td>
    </tr>
  </tbody>
</table>
</div>

### Analysis

Question 1: If suicides are going up year over year:

```python
num_s = df['suicides_no'].groupby(df.year).sum()
print(num_s)
```

    year
    1985    116063
    1986    120670
    1987    126842
    1988    121026
    1989    160244
    1990    193361
    1991    198020
    1992    211473
    1993    221565
    1994    232063
    1995    243544
    1996    246725
    1997    240745
    1998    249591
    1999    256119
    2000    255832
    2001    250652
    2002    256095
    2003    256079
    2004    240861
    2005    234375
    2006    233361
    2007    233408
    2008    235447
    2009    243487
    2010    238702
    2011    236484
    2012    230160
    2013    223199
    2014    222984
    2015    203640
    2016     15603
    Name: suicides_no, dtype: int64

Suicides over time (visualized)

```python

num_s = df['suicides_no'].groupby(df.year).sum()
num_s.plot()
plt.xlabel('year')
plt.ylabel('suicides_no')
```




    Text(0, 0.5, 'suicides_no')




![second pic]({{ site.baseurl }}/images/image14.png)

```python
f,ax = plt.subplots(1,1, figsize=(17,6))
ax = sns.barplot(x = df['year'], y = 'suicides_no', data=df, palette='Spectral')
```


![second pic]({{ site.baseurl }}/images/image19.png)


```python
## Suicides number by year (high to low)
year_suicides = df.groupby('year')[['suicides_no']].sum().reset_index()
year_suicides.sort_values(by='suicides_no', ascending=False).style.background_gradient(cmap='Greens', 
                                                                                       subset=['suicides_no'])
```




<style  type="text/css" >
    #T_304ede1e_f43f_11ea_a187_186590dc81a7row0_col1 {
            background-color:  #00441b;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row1_col1 {
            background-color:  #00441b;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row2_col1 {
            background-color:  #00441b;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row3_col1 {
            background-color:  #00441b;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row4_col1 {
            background-color:  #004a1e;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row5_col1 {
            background-color:  #004c1e;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row6_col1 {
            background-color:  #005020;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row7_col1 {
            background-color:  #005522;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row8_col1 {
            background-color:  #005522;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row9_col1 {
            background-color:  #005924;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row10_col1 {
            background-color:  #005924;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row11_col1 {
            background-color:  #005b25;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row12_col1 {
            background-color:  #005e26;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row13_col1 {
            background-color:  #006027;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row14_col1 {
            background-color:  #006227;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row15_col1 {
            background-color:  #006328;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row16_col1 {
            background-color:  #006328;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row17_col1 {
            background-color:  #006428;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row18_col1 {
            background-color:  #006729;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row19_col1 {
            background-color:  #03702e;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row20_col1 {
            background-color:  #03702e;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row21_col1 {
            background-color:  #05712f;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row22_col1 {
            background-color:  #117b38;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row23_col1 {
            background-color:  #19833e;
            color:  #f1f1f1;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row24_col1 {
            background-color:  #208843;
            color:  #000000;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row25_col1 {
            background-color:  #258d47;
            color:  #000000;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row26_col1 {
            background-color:  #4bb062;
            color:  #000000;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row27_col1 {
            background-color:  #81ca81;
            color:  #000000;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row28_col1 {
            background-color:  #8ace88;
            color:  #000000;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row29_col1 {
            background-color:  #8bcf89;
            color:  #000000;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row30_col1 {
            background-color:  #92d28f;
            color:  #000000;
        }    #T_304ede1e_f43f_11ea_a187_186590dc81a7row31_col1 {
            background-color:  #f7fcf5;
            color:  #000000;
        }</style><table id="T_304ede1e_f43f_11ea_a187_186590dc81a7" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >year</th>        <th class="col_heading level0 col1" >suicides_no</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row0" class="row_heading level0 row0" >14</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row0_col0" class="data row0 col0" >1999</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row0_col1" class="data row0 col1" >256119</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row1" class="row_heading level0 row1" >17</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row1_col0" class="data row1 col0" >2002</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row1_col1" class="data row1 col1" >256095</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row2" class="row_heading level0 row2" >18</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row2_col0" class="data row2 col0" >2003</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row2_col1" class="data row2 col1" >256079</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row3" class="row_heading level0 row3" >15</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row3_col0" class="data row3 col0" >2000</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row3_col1" class="data row3 col1" >255832</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row4" class="row_heading level0 row4" >16</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row4_col0" class="data row4 col0" >2001</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row4_col1" class="data row4 col1" >250652</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row5" class="row_heading level0 row5" >13</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row5_col0" class="data row5 col0" >1998</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row5_col1" class="data row5 col1" >249591</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row6" class="row_heading level0 row6" >11</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row6_col0" class="data row6 col0" >1996</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row6_col1" class="data row6 col1" >246725</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row7" class="row_heading level0 row7" >10</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row7_col0" class="data row7 col0" >1995</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row7_col1" class="data row7 col1" >243544</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row8" class="row_heading level0 row8" >24</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row8_col0" class="data row8 col0" >2009</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row8_col1" class="data row8 col1" >243487</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row9" class="row_heading level0 row9" >19</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row9_col0" class="data row9 col0" >2004</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row9_col1" class="data row9 col1" >240861</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row10" class="row_heading level0 row10" >12</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row10_col0" class="data row10 col0" >1997</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row10_col1" class="data row10 col1" >240745</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row11" class="row_heading level0 row11" >25</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row11_col0" class="data row11 col0" >2010</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row11_col1" class="data row11 col1" >238702</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row12" class="row_heading level0 row12" >26</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row12_col0" class="data row12 col0" >2011</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row12_col1" class="data row12 col1" >236484</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row13" class="row_heading level0 row13" >23</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row13_col0" class="data row13 col0" >2008</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row13_col1" class="data row13 col1" >235447</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row14" class="row_heading level0 row14" >20</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row14_col0" class="data row14 col0" >2005</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row14_col1" class="data row14 col1" >234375</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row15" class="row_heading level0 row15" >22</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row15_col0" class="data row15 col0" >2007</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row15_col1" class="data row15 col1" >233408</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row16" class="row_heading level0 row16" >21</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row16_col0" class="data row16 col0" >2006</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row16_col1" class="data row16 col1" >233361</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row17" class="row_heading level0 row17" >9</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row17_col0" class="data row17 col0" >1994</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row17_col1" class="data row17 col1" >232063</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row18" class="row_heading level0 row18" >27</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row18_col0" class="data row18 col0" >2012</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row18_col1" class="data row18 col1" >230160</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row19" class="row_heading level0 row19" >28</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row19_col0" class="data row19 col0" >2013</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row19_col1" class="data row19 col1" >223199</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row20" class="row_heading level0 row20" >29</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row20_col0" class="data row20 col0" >2014</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row20_col1" class="data row20 col1" >222984</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row21" class="row_heading level0 row21" >8</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row21_col0" class="data row21 col0" >1993</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row21_col1" class="data row21 col1" >221565</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row22" class="row_heading level0 row22" >7</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row22_col0" class="data row22 col0" >1992</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row22_col1" class="data row22 col1" >211473</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row23" class="row_heading level0 row23" >30</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row23_col0" class="data row23 col0" >2015</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row23_col1" class="data row23 col1" >203640</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row24" class="row_heading level0 row24" >6</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row24_col0" class="data row24 col0" >1991</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row24_col1" class="data row24 col1" >198020</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row25" class="row_heading level0 row25" >5</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row25_col0" class="data row25 col0" >1990</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row25_col1" class="data row25 col1" >193361</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row26" class="row_heading level0 row26" >4</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row26_col0" class="data row26 col0" >1989</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row26_col1" class="data row26 col1" >160244</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row27" class="row_heading level0 row27" >2</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row27_col0" class="data row27 col0" >1987</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row27_col1" class="data row27 col1" >126842</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row28" class="row_heading level0 row28" >3</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row28_col0" class="data row28 col0" >1988</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row28_col1" class="data row28 col1" >121026</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row29" class="row_heading level0 row29" >1</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row29_col0" class="data row29 col0" >1986</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row29_col1" class="data row29 col1" >120670</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row30" class="row_heading level0 row30" >0</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row30_col0" class="data row30 col0" >1985</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row30_col1" class="data row30 col1" >116063</td>
            </tr>
            <tr>
                        <th id="T_304ede1e_f43f_11ea_a187_186590dc81a7level0_row31" class="row_heading level0 row31" >31</th>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row31_col0" class="data row31 col0" >2016</td>
                        <td id="T_304ede1e_f43f_11ea_a187_186590dc81a7row31_col1" class="data row31 col1" >15603</td>
            </tr>
    </tbody></table>


As we can see, total suicides went up in the late 80s and early 90s nominally, however in real terms, total suicides have generally been goiind down given that total suicides have stayed relatively stagnant from 1995-2010, despite worldwide population going up significantly over this time. 

Question 2 & 3: The genders most likely to commit suicide and the Age Cohort

```python
df2 = df.copy()
df2['year'] = pd.to_datetime(df['year'], format = '%Y')
data = df2.groupby(['year', 'sex']).agg('mean').reset_index()
sns.lineplot(x = 'year', y = 'suicides/100k pop', hue = 'sex', data = df2)
plt.xlim('1985', '2015')
plt.title('Evolution of the mean suicides number per 100k population (1985 - 2015)');
```


![second pic]({{ site.baseurl }}/images/image15.png)


The above diagram shows a stark comparison between suicide rates consistently between men and women over roughly the past 30 years. Let's now look at how adding in the age groups changes the results: 


```python
# This code groups by year, sex, and age group and then takes the aggregates the mean of each: 
df2 = df2.groupby(['year', 'sex', 'age']).agg('mean').reset_index()

# Then instantiate the SNS plot:
sns.relplot(x = 'year', y='suicides/100k pop', 
            hue = 'sex', col='age', data = df2, col_wrap = 3, facet_kws=dict(sharey=False), kind='line')

plt.xlim("1985", "2015")
plt.subplots_adjust(top=0.9)
plt.suptitle("Evolution of suicide by sex and age category (1985 - 2015)", size=18)
```




![second pic]({{ site.baseurl }}/images/image16.png)



```python
# Box/barplot: 
f,ax = plt.subplots(1,1, figsize=(13,6))
ax = sns.barplot(x=df['generation'], y = 'suicides_no', hue='sex', data=df, palette='dark')
```


![second pic]({{ site.baseurl }}/images/image17.png)



```python
## Same but with age and sex: 
f,ax = plt.subplots(1,1,figsize=(13,6))
ax = sns.barplot(x=df['age'], y = 'suicides_no', hue='sex', data=df, palette = 'coolwarm')
```


![second pic]({{ site.baseurl }}/images/image18.png)



```python
## Suicides number by age group
age_grp = df.groupby('age')[['suicides_no']].sum().reset_index()
age_grp.sort_values(by='suicides_no', ascending=False).style.background_gradient(cmap='Blues', subset=['suicides_no'])
```




<style  type="text/css" >
    #T_3055736e_f43f_11ea_a187_186590dc81a7row0_col1 {
            background-color:  #08306b;
            color:  #f1f1f1;
        }    #T_3055736e_f43f_11ea_a187_186590dc81a7row1_col1 {
            background-color:  #3686c0;
            color:  #000000;
        }    #T_3055736e_f43f_11ea_a187_186590dc81a7row2_col1 {
            background-color:  #81badb;
            color:  #000000;
        }    #T_3055736e_f43f_11ea_a187_186590dc81a7row3_col1 {
            background-color:  #b2d2e8;
            color:  #000000;
        }    #T_3055736e_f43f_11ea_a187_186590dc81a7row4_col1 {
            background-color:  #c6dbef;
            color:  #000000;
        }    #T_3055736e_f43f_11ea_a187_186590dc81a7row5_col1 {
            background-color:  #f7fbff;
            color:  #000000;
        }</style><table id="T_3055736e_f43f_11ea_a187_186590dc81a7" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >age</th>        <th class="col_heading level0 col1" >suicides_no</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_3055736e_f43f_11ea_a187_186590dc81a7level0_row0" class="row_heading level0 row0" >2</th>
                        <td id="T_3055736e_f43f_11ea_a187_186590dc81a7row0_col0" class="data row0 col0" >35-54 years</td>
                        <td id="T_3055736e_f43f_11ea_a187_186590dc81a7row0_col1" class="data row0 col1" >2452141</td>
            </tr>
            <tr>
                        <th id="T_3055736e_f43f_11ea_a187_186590dc81a7level0_row1" class="row_heading level0 row1" >4</th>
                        <td id="T_3055736e_f43f_11ea_a187_186590dc81a7row1_col0" class="data row1 col0" >55-74 years</td>
                        <td id="T_3055736e_f43f_11ea_a187_186590dc81a7row1_col1" class="data row1 col1" >1658443</td>
            </tr>
            <tr>
                        <th id="T_3055736e_f43f_11ea_a187_186590dc81a7level0_row2" class="row_heading level0 row2" >1</th>
                        <td id="T_3055736e_f43f_11ea_a187_186590dc81a7row2_col0" class="data row2 col0" >25-34 years</td>
                        <td id="T_3055736e_f43f_11ea_a187_186590dc81a7row2_col1" class="data row2 col1" >1123912</td>
            </tr>
            <tr>
                        <th id="T_3055736e_f43f_11ea_a187_186590dc81a7level0_row3" class="row_heading level0 row3" >0</th>
                        <td id="T_3055736e_f43f_11ea_a187_186590dc81a7row3_col0" class="data row3 col0" >15-24 years</td>
                        <td id="T_3055736e_f43f_11ea_a187_186590dc81a7row3_col1" class="data row3 col1" >808542</td>
            </tr>
            <tr>
                        <th id="T_3055736e_f43f_11ea_a187_186590dc81a7level0_row4" class="row_heading level0 row4" >5</th>
                        <td id="T_3055736e_f43f_11ea_a187_186590dc81a7row4_col0" class="data row4 col0" >75+ years</td>
                        <td id="T_3055736e_f43f_11ea_a187_186590dc81a7row4_col1" class="data row4 col1" >653118</td>
            </tr>
            <tr>
                        <th id="T_3055736e_f43f_11ea_a187_186590dc81a7level0_row5" class="row_heading level0 row5" >3</th>
                        <td id="T_3055736e_f43f_11ea_a187_186590dc81a7row5_col0" class="data row5 col0" >5-14 years</td>
                        <td id="T_3055736e_f43f_11ea_a187_186590dc81a7row5_col1" class="data row5 col1" >52264</td>
            </tr>
    </tbody></table>




```python
## Suicides number per 100k population
per100k = df.groupby(['country', 'year'])[['suicides/100k pop']].sum().reset_index()
per100k.sort_values(by='suicides/100k pop', ascending=False).head(20).style.background_gradient(cmap='Reds', 
                                                                                                subset=['suicides/100k pop'])
```




<style  type="text/css" >
    #T_3060dace_f43f_11ea_a187_186590dc81a7row0_col2 {
            background-color:  #67000d;
            color:  #f1f1f1;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row1_col2 {
            background-color:  #f14331;
            color:  #f1f1f1;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row2_col2 {
            background-color:  #fc8464;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row3_col2 {
            background-color:  #fc8e6e;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row4_col2 {
            background-color:  #fc9272;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row5_col2 {
            background-color:  #fc9576;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row6_col2 {
            background-color:  #fc997a;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row7_col2 {
            background-color:  #fc9d7f;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row8_col2 {
            background-color:  #fca082;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row9_col2 {
            background-color:  #fcad90;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row10_col2 {
            background-color:  #fcb499;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row11_col2 {
            background-color:  #fcbda4;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row12_col2 {
            background-color:  #fdd3c1;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row13_col2 {
            background-color:  #fed8c7;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row14_col2 {
            background-color:  #fedfd0;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row15_col2 {
            background-color:  #fee5d9;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row16_col2 {
            background-color:  #feeae1;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row17_col2 {
            background-color:  #fff2ec;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row18_col2 {
            background-color:  #fff4ee;
            color:  #000000;
        }    #T_3060dace_f43f_11ea_a187_186590dc81a7row19_col2 {
            background-color:  #fff5f0;
            color:  #000000;
        }</style><table id="T_3060dace_f43f_11ea_a187_186590dc81a7" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >country</th>        <th class="col_heading level0 col1" >year</th>        <th class="col_heading level0 col2" >suicides/100k pop</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row0" class="row_heading level0 row0" >1255</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row0_col0" class="data row0 col0" >Lithuania</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row0_col1" class="data row0 col1" >1995</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row0_col2" class="data row0 col2" >639.3</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row1" class="row_heading level0 row1" >1256</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row1_col0" class="data row1 col0" >Lithuania</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row1_col1" class="data row1 col1" >1996</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row1_col2" class="data row1 col2" >595.61</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row2" class="row_heading level0 row2" >948</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row2_col0" class="data row2 col0" >Hungary</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row2_col1" class="data row2 col1" >1991</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row2_col2" class="data row2 col2" >575</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row3" class="row_heading level0 row3" >1260</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row3_col0" class="data row3 col0" >Lithuania</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row3_col1" class="data row3 col1" >2000</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row3_col2" class="data row3 col2" >571.8</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row4" class="row_heading level0 row4" >949</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row4_col0" class="data row4 col0" >Hungary</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row4_col1" class="data row4 col1" >1992</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row4_col2" class="data row4 col2" >570.26</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row5" class="row_heading level0 row5" >1261</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row5_col0" class="data row5 col0" >Lithuania</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row5_col1" class="data row5 col1" >2001</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row5_col2" class="data row5 col2" >568.98</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row6" class="row_heading level0 row6" >1752</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row6_col0" class="data row6 col0" >Russian Federation</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row6_col1" class="data row6 col1" >1994</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row6_col2" class="data row6 col2" >567.64</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row7" class="row_heading level0 row7" >1258</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row7_col0" class="data row7 col0" >Lithuania</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row7_col1" class="data row7 col1" >1998</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row7_col2" class="data row7 col2" >566.36</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row8" class="row_heading level0 row8" >1257</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row8_col0" class="data row8 col0" >Lithuania</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row8_col1" class="data row8 col1" >1997</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row8_col2" class="data row8 col2" >565.44</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row9" class="row_heading level0 row9" >1259</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row9_col0" class="data row9 col0" >Lithuania</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row9_col1" class="data row9 col1" >1999</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row9_col2" class="data row9 col2" >561.53</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row10" class="row_heading level0 row10" >1994</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row10_col0" class="data row10 col0" >Sri Lanka</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row10_col1" class="data row10 col1" >1985</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row10_col2" class="data row10 col2" >558.72</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row11" class="row_heading level0 row11" >1262</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row11_col0" class="data row11 col0" >Lithuania</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row11_col1" class="data row11 col1" >2002</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row11_col2" class="data row11 col2" >555.62</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row12" class="row_heading level0 row12" >1753</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row12_col0" class="data row12 col0" >Russian Federation</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row12_col1" class="data row12 col1" >1995</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row12_col2" class="data row12 col2" >547.38</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row13" class="row_heading level0 row13" >1234</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row13_col0" class="data row13 col0" >Latvia</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row13_col1" class="data row13 col1" >1995</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row13_col2" class="data row13 col2" >545.62</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row14" class="row_heading level0 row14" >697</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row14_col0" class="data row14 col0" >Estonia</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row14_col1" class="data row14 col1" >1995</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row14_col2" class="data row14 col2" >543.19</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row15" class="row_heading level0 row15" >950</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row15_col0" class="data row15 col0" >Hungary</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row15_col1" class="data row15 col1" >1993</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row15_col2" class="data row15 col2" >539.28</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row16" class="row_heading level0 row16" >951</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row16_col0" class="data row16 col0" >Hungary</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row16_col1" class="data row16 col1" >1994</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row16_col2" class="data row16 col2" >535.81</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row17" class="row_heading level0 row17" >1263</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row17_col0" class="data row17 col0" >Lithuania</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row17_col1" class="data row17 col1" >2003</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row17_col2" class="data row17 col2" >530.52</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row18" class="row_heading level0 row18" >1995</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row18_col0" class="data row18 col0" >Sri Lanka</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row18_col1" class="data row18 col1" >1986</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row18_col2" class="data row18 col2" >529.8</td>
            </tr>
            <tr>
                        <th id="T_3060dace_f43f_11ea_a187_186590dc81a7level0_row19" class="row_heading level0 row19" >698</th>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row19_col0" class="data row19 col0" >Estonia</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row19_col1" class="data row19 col1" >1996</td>
                        <td id="T_3060dace_f43f_11ea_a187_186590dc81a7row19_col2" class="data row19 col2" >528.72</td>
            </tr>
    </tbody></table>




```python
# Let's fill in our Null values at this point. Since it was just null for one single column (HDI for year) we don't 
# need to specify which column:

df.fillna(df.mean(), inplace=True)
```


```python
# Let's also drop the country-year column as we mentioned earlier that it is redundant:
df.drop("country-year", axis=1, inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>sex</th>
      <th>age</th>
      <th>suicides_no</th>
      <th>population</th>
      <th>suicides/100k pop</th>
      <th>HDI for year</th>
      <th>gdp_for_year</th>
      <th>gdp_per_capita</th>
      <th>generation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Albania</td>
      <td>1987</td>
      <td>male</td>
      <td>15-24 years</td>
      <td>21</td>
      <td>312900</td>
      <td>6.71</td>
      <td>0.776601</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>Generation X</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Albania</td>
      <td>1987</td>
      <td>male</td>
      <td>35-54 years</td>
      <td>16</td>
      <td>308000</td>
      <td>5.19</td>
      <td>0.776601</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>Silent</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Albania</td>
      <td>1987</td>
      <td>female</td>
      <td>15-24 years</td>
      <td>14</td>
      <td>289700</td>
      <td>4.83</td>
      <td>0.776601</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>Generation X</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Albania</td>
      <td>1987</td>
      <td>male</td>
      <td>75+ years</td>
      <td>1</td>
      <td>21800</td>
      <td>4.59</td>
      <td>0.776601</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>G.I. Generation</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Albania</td>
      <td>1987</td>
      <td>male</td>
      <td>25-34 years</td>
      <td>9</td>
      <td>274300</td>
      <td>3.28</td>
      <td>0.776601</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>Boomers</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's look at the data types again to see if any changes need to be made:
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 27820 entries, 0 to 27819
    Data columns (total 11 columns):
    country              27820 non-null object
    year                 27820 non-null int64
    sex                  27820 non-null object
    age                  27820 non-null object
    suicides_no          27820 non-null int64
    population           27820 non-null int64
    suicides/100k pop    27820 non-null float64
    HDI for year         27820 non-null float64
    gdp_for_year         27820 non-null object
    gdp_per_capita       27820 non-null int64
    generation           27820 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 2.3+ MB


Ultimately, it is better to change the objects into category types for classification purposes and to change "gdp_for_year" to an int:


```python
df[['country', 'age', 'sex', 'generation']]= df[['country', 'age', 'sex', 'generation']].astype('category')

df['gdp_for_year'] = df['gdp_for_year'].str.replace(',', "").astype('int')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 27820 entries, 0 to 27819
    Data columns (total 11 columns):
    country              27820 non-null category
    year                 27820 non-null int64
    sex                  27820 non-null category
    age                  27820 non-null category
    suicides_no          27820 non-null int64
    population           27820 non-null int64
    suicides/100k pop    27820 non-null float64
    HDI for year         27820 non-null float64
    gdp_for_year         27820 non-null int64
    gdp_per_capita       27820 non-null int64
    generation           27820 non-null category
    dtypes: category(4), float64(2), int64(5)
    memory usage: 1.6 MB


Question 4: If richer countries commit suicide at lower rates:

```python
# Let's investigate the correlation of factors using a heatmap:
f,ax = plt.subplots(1,1,figsize=(10,10))
ax = sns.heatmap(df.corr(),annot=True, cmap='coolwarm')
```


![second pic]({{ site.baseurl }}/images/image20.png)


Nothing really jumping off the page here to be honest. 

```python
f, ax = plt.subplots(1,1, figsize=(8,6))
ax = sns.scatterplot(x="gdp_for_year", y='suicides_no', data=df, color='red')
```


![second pic]({{ site.baseurl }}/images/image21.png)



```python
# This above relationship certainly doesn't seem linear, and that makes sense from the correlation heatmap we 
# created earlier, therefore GDP doesnt seem to have a significant impact on suicide rate. 

# Since we took the time to fill in the nulls of HDI let's look at that relationship with our defacto y-variable 
# 'suicides_no'

f, ax = plt.subplots(1,1, figsize=(10,8))
ax = sns.scatterplot(x="HDI for year", y="suicides_no", data=df, color='green')
```


![second pic]({{ site.baseurl }}/images/image22.png)



```python
##Suicides by gender:
df.groupby(['sex'])['suicides_no'].sum()
```




    sex
    female    1559510
    male      5188910
    Name: suicides_no, dtype: int64




```python
print(5188910/1559510)
```

    3.3272694628440984


As we can see men commit suicide across the board about 3.3x as often as women. This is even more prounounced for some countries like Russia: 


```python
f, ax = plt.subplots(1,1, figsize=(10,10))
ax = sns.boxplot(x='age', y='suicides_no', hue='sex',
                 data=df[df['country']=='Russian Federation'],
                 palette='Set1')
```


![second pic]({{ site.baseurl }}/images/image23.png)



```python
## Using cat.codes method to convert category into numerical labels
columns = df.select_dtypes(['category']).columns
df[columns] = df[columns].apply(lambda fx: fx.cat.codes)
df.dtypes
```




    country                 int8
    year                   int64
    sex                     int8
    age                     int8
    suicides_no            int64
    population             int64
    suicides/100k pop    float64
    HDI for year         float64
    gdp_for_year           int64
    gdp_per_capita         int64
    generation              int8
    dtype: object


Let's do some unsupervised ML to determine if we can establish two statistically significant clusters of countries: those that commit suicide at higher rates, vs those that do not: 

```python
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
x = df.drop('suicides_no', axis=True)
y = df['suicides_no']
kmeans = KMeans(n_clusters=2)
kmeans.fit(x)
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=0, tol=0.0001, verbose=0)
y_kmeans = kmeans.predict(x)
x, y_kmeans = make_blobs(n_samples=600, centers=2, cluster_std=0.60, random_state=0)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:,0], x[:,1], c=y_kmeans, cmap='cool')
```




    <mpl_toolkits.mplot3d.art3d.Path3DCollection at 0x1a1a0c6c50>




![second pic]({{ site.baseurl }}/images/image24.png)

### Conclusion and Future Considerations

Ultimately through our analysis we identified a number of interesting insights. Firstly, as we discussed in Question 1, although nominal suicides stayed relatively stagnant from ~1995-2015, the proportion of deaths by suicide dropped over this time, due to increasing worldwide population over the time timeframe. Interestingly we also identified that although suicide is very commonly brought up as a very important issue affecting children and young people (as it should be), that suicides and suicidal thoughts clearly affect people of virtually every age range, and are highest in total suicides in the age range of 35-54 years old for both men and women. (More analysis would need to be completed to determine definitively if this is the most common age range of suicide deaths or if this is just the most populated demographic from the dataset, however it is still interesting). In addition more individuals in the dataset committed suicide between the ages of 55-74 then 25-34, showing again how common suicide is for people in an older demographic.  

Through the analysis, we also found that men commit suicide 3.3x more often then women across the whole data set, but that is even more pronounced by some of the largest countries in the dataset (Brazil/Russia, for example) 

We also successfully identified that GDP is not a good predictor of suicides. Although the countries with the highest number of suicides have the lowest GDP,  suicide rates don't go up or down as you get to the richer countries on the list. In addition many of the countries that have GDP_for_year below 0.50 have lower suicide rates then countries with much higher GDP. Therefore we can say there is limited evidence of a statistically significant relationship between GDP and suicide rates. A similar relationship emerged with HDI and suicides, and if anything, as HDI increases suicide numbers also increase (although this must be taken with a significant grain of salt given that we had to fill in the null values for HDI). 

Lastly, we managed to identify that there were two statistically significant clusters of countries (those that commit suicide at higher rates and those that don't) amongst the data set. More analysis would have to be done to identify what ultimately seperates these groups from eachother, however it is clear that it is not necessarily HDI or GDP. Therefore more information would need to be provided regarding demographics, cultural, and sociological factors (race, ethnicity, income inequality, religion, etc.) as well as geographical. 

Future Considerations:
- Although this is a topic that has gained a lot of interest in recent years, there is still a lot to determine what sort of geopolitical factors play a role, and by how much. In the future it would be interesting to incorporate alcohol and drug use statistics, the level of religion/secularism in each country, the general attitudes and consumption statistics in each country, etc.
- I believe that through increased research in understanding what makes suicide seemingly an attractive option for people will allow us to better prevent these deaths, as opposed to acting like suicide doesn't exist. We do nothing to fight suicide by burying our heads in the sand and hoping the problem goes away. Grassroot organizations that aim to assist people with mental illness problems are vital, but we also need to better understand what perpetuates and exacerbates those problems. 

Throughout this analysis, I completed a great deal more Supervised and Unsupervised ML which largely resulted in results with minimal conclusions, feel free to email me if you're interested in the other analysis I completed. In addition please feel free to read the following links for more information: 
- https://pubmed.ncbi.nlm.nih.gov/30933702/
- https://pubmed.ncbi.nlm.nih.gov/31504808/

Thank you! 