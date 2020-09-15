---
layout: post
title: Suicide Rates since 1985 
---

Thursday September 10th marked World Suicide Prevention Day, a day where awareness is brought to one of the most prevalent and preventable causes of death: suicide. Our society has a strange relationship with suicide. We all know that it is a problem, however we rarely talk about it or acknowledge it as if that may give people ideas. I believe by investigating more of what causes suicides and why people ultimately feel the need to take their own life we will prevent these tragedies far more effectively. Although this analysis will not even begin to answer why people commit suicide I aim to identify: 
- If suicides are going up year over year
- The genders most likely to commit suicide 
- The age ranges that typically commit suicide at higher rates 
- If richer countries commit suicide at lower rates

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
- The population. 
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




    Text(0.5, 0.98, 'Evolution of suicide by sex and age category (1985 - 2015)')




![second pic]({{ site.baseurl }}/images/image15.png)



```python
## Number of suicides in 1985
year_1985 = df[(df['year'] == 1985)]
year_1985 = year_1985.groupby('country')[['suicides_no']].sum().reset_index()

## Sorting values in ascending order
year_1985 = year_1985.sort_values(by='suicides_no', ascending=False)

## Styling output dataframe
year_1985.style.background_gradient(cmap='Purples', subset=['suicides_no'])
```




<style  type="text/css" >
    #T_2d11a772_f43f_11ea_a187_186590dc81a7row0_col1 {
            background-color:  #3f007d;
            color:  #f1f1f1;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row1_col1 {
            background-color:  #63439c;
            color:  #f1f1f1;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row2_col1 {
            background-color:  #b0afd4;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row3_col1 {
            background-color:  #e4e3f0;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row4_col1 {
            background-color:  #e7e6f1;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row5_col1 {
            background-color:  #e9e8f2;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row6_col1 {
            background-color:  #ecebf4;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row7_col1 {
            background-color:  #efedf5;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row8_col1 {
            background-color:  #f1eff6;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row9_col1 {
            background-color:  #f2f0f7;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row10_col1 {
            background-color:  #f3f2f8;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row11_col1 {
            background-color:  #f4f3f8;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row12_col1 {
            background-color:  #f5f3f8;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row13_col1 {
            background-color:  #f5f4f9;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row14_col1 {
            background-color:  #f5f4f9;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row15_col1 {
            background-color:  #f6f5f9;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row16_col1 {
            background-color:  #f7f5fa;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row17_col1 {
            background-color:  #f7f6fa;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row18_col1 {
            background-color:  #f9f7fb;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row19_col1 {
            background-color:  #f9f7fb;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row20_col1 {
            background-color:  #faf9fc;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row21_col1 {
            background-color:  #fbfafc;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row22_col1 {
            background-color:  #fbfafc;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row23_col1 {
            background-color:  #fbfafc;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row24_col1 {
            background-color:  #fbfafc;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row25_col1 {
            background-color:  #fbfafc;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row26_col1 {
            background-color:  #fbfafc;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row27_col1 {
            background-color:  #fbfafc;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row28_col1 {
            background-color:  #fbfafc;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row29_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row30_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row31_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row32_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row33_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row34_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row35_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row36_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row37_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row38_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row39_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row40_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row41_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row42_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row43_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row44_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row45_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row46_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_2d11a772_f43f_11ea_a187_186590dc81a7row47_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }</style><table id="T_2d11a772_f43f_11ea_a187_186590dc81a7" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >country</th>        <th class="col_heading level0 col1" >suicides_no</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row0" class="row_heading level0 row0" >46</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row0_col0" class="data row0 col0" >United States</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row0_col1" class="data row0 col1" >29446</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row1" class="row_heading level0 row1" >24</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row1_col0" class="data row1 col0" >Japan</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row1_col1" class="data row1 col1" >23257</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row2" class="row_heading level0 row2" >16</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row2_col0" class="data row2 col0" >France</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row2_col1" class="data row2 col1" >12501</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row3" class="row_heading level0 row3" >41</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row3_col0" class="data row3 col0" >Sri Lanka</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row3_col1" class="data row3 col1" >5668</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row4" class="row_heading level0 row4" >45</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row4_col0" class="data row4 col0" >United Kingdom</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row4_col1" class="data row4 col1" >5105</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row5" class="row_heading level0 row5" >22</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row5_col0" class="data row5 col0" >Italy</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row5_col1" class="data row5 col1" >4759</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row6" class="row_heading level0 row6" >8</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row6_col0" class="data row6 col0" >Brazil</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row6_col1" class="data row6 col1" >4228</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row7" class="row_heading level0 row7" >36</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row7_col0" class="data row7 col0" >Republic of Korea</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row7_col1" class="data row7 col1" >3689</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row8" class="row_heading level0 row8" >10</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row8_col0" class="data row8 col0" >Canada</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row8_col1" class="data row8 col1" >3258</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row9" class="row_heading level0 row9" >43</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row9_col0" class="data row9 col0" >Thailand</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row9_col1" class="data row9 col1" >2982</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row10" class="row_heading level0 row10" >40</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row10_col0" class="data row10 col0" >Spain</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row10_col1" class="data row10 col1" >2514</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row11" class="row_heading level0 row11" >7</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row11_col0" class="data row11 col0" >Belgium</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row11_col1" class="data row11 col1" >2281</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row12" class="row_heading level0 row12" >3</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row12_col0" class="data row12 col0" >Austria</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row12_col1" class="data row12 col1" >2091</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row13" class="row_heading level0 row13" >1</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row13_col0" class="data row13 col0" >Argentina</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row13_col1" class="data row13 col1" >1988</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row14" class="row_heading level0 row14" >2</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row14_col0" class="data row14 col0" >Australia</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row14_col1" class="data row14 col1" >1861</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row15" class="row_heading level0 row15" >30</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row15_col0" class="data row15 col0" >Netherlands</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row15_col1" class="data row15 col1" >1638</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row16" class="row_heading level0 row16" >29</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row16_col0" class="data row16 col0" >Mexico</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row16_col1" class="data row16 col1" >1544</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row17" class="row_heading level0 row17" >9</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row17_col0" class="data row17 col0" >Bulgaria</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row17_col1" class="data row17 col1" >1456</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row18" class="row_heading level0 row18" >12</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row18_col0" class="data row18 col0" >Colombia</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row18_col1" class="data row18 col1" >1001</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row19" class="row_heading level0 row19" >34</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row19_col0" class="data row19 col0" >Portugal</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row19_col1" class="data row19 col1" >983</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row20" class="row_heading level0 row20" >11</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row20_col0" class="data row20 col0" >Chile</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row20_col1" class="data row20 col1" >683</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row21" class="row_heading level0 row21" >17</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row21_col0" class="data row21 col0" >Greece</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row21_col1" class="data row21 col1" >405</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row22" class="row_heading level0 row22" >15</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row22_col0" class="data row22 col0" >Ecuador</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row22_col1" class="data row22 col1" >393</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row23" class="row_heading level0 row23" >31</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row23_col0" class="data row23 col0" >New Zealand</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row23_col1" class="data row23 col1" >338</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row24" class="row_heading level0 row24" >39</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row24_col0" class="data row24 col0" >Singapore</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row24_col1" class="data row24 col1" >324</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row25" class="row_heading level0 row25" >47</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row25_col0" class="data row25 col0" >Uruguay</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row25_col1" class="data row25 col1" >287</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row26" class="row_heading level0 row26" >20</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row26_col0" class="data row26 col0" >Ireland</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row26_col1" class="data row26 col1" >276</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row27" class="row_heading level0 row27" >35</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row27_col0" class="data row27 col0" >Puerto Rico</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row27_col1" class="data row27 col1" >269</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row28" class="row_heading level0 row28" >21</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row28_col0" class="data row28 col0" >Israel</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row28_col1" class="data row28 col1" >234</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row29" class="row_heading level0 row29" >13</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row29_col0" class="data row29 col0" >Costa Rica</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row29_col1" class="data row29 col1" >128</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row30" class="row_heading level0 row30" >28</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row30_col0" class="data row30 col0" >Mauritius</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row30_col1" class="data row30 col1" >104</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row31" class="row_heading level0 row31" >42</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row31_col0" class="data row31 col0" >Suriname</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row31_col1" class="data row31 col1" >80</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row32" class="row_heading level0 row32" >33</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row32_col0" class="data row32 col0" >Paraguay</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row32_col1" class="data row32 col1" >63</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row33" class="row_heading level0 row33" >32</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row33_col0" class="data row33 col0" >Panama</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row33_col1" class="data row33 col1" >56</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row34" class="row_heading level0 row34" >26</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row34_col0" class="data row34 col0" >Luxembourg</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row34_col1" class="data row34 col1" >55</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row35" class="row_heading level0 row35" >19</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row35_col0" class="data row35 col0" >Iceland</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row35_col1" class="data row35 col1" >32</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row36" class="row_heading level0 row36" >44</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row36_col0" class="data row36 col0" >Trinidad and Tobago</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row36_col1" class="data row36 col1" >29</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row37" class="row_heading level0 row37" >25</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row37_col0" class="data row37 col0" >Kuwait</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row37_col1" class="data row37 col1" >17</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row38" class="row_heading level0 row38" >5</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row38_col0" class="data row38 col0" >Bahrain</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row38_col1" class="data row38 col1" >11</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row39" class="row_heading level0 row39" >23</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row39_col0" class="data row39 col0" >Jamaica</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row39_col1" class="data row39 col1" >8</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row40" class="row_heading level0 row40" >38</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row40_col0" class="data row40 col0" >Seychelles</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row40_col1" class="data row40 col1" >8</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row41" class="row_heading level0 row41" >6</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row41_col0" class="data row41 col0" >Barbados</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row41_col1" class="data row41 col1" >7</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row42" class="row_heading level0 row42" >37</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row42_col0" class="data row42 col0" >Saint Vincent and Grenadines</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row42_col1" class="data row42 col1" >2</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row43" class="row_heading level0 row43" >27</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row43_col0" class="data row43 col0" >Malta</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row43_col1" class="data row43 col1" >2</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row44" class="row_heading level0 row44" >18</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row44_col0" class="data row44 col0" >Grenada</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row44_col1" class="data row44 col1" >1</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row45" class="row_heading level0 row45" >4</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row45_col0" class="data row45 col0" >Bahamas</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row45_col1" class="data row45 col1" >1</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row46" class="row_heading level0 row46" >14</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row46_col0" class="data row46 col0" >Dominica</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row46_col1" class="data row46 col1" >0</td>
            </tr>
            <tr>
                        <th id="T_2d11a772_f43f_11ea_a187_186590dc81a7level0_row47" class="row_heading level0 row47" >0</th>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row47_col0" class="data row47 col0" >Antigua and Barbuda</td>
                        <td id="T_2d11a772_f43f_11ea_a187_186590dc81a7row47_col1" class="data row47 col1" >0</td>
            </tr>
    </tbody></table>




```python
df2015 = df[(df['year'] == 2015)]
df2015['country'].value_counts()
```




    Netherlands           12
    Norway                12
    Denmark               12
    Russian Federation    12
    Kazakhstan            12
                          ..
    Lithuania             12
    Qatar                 12
    Panama                12
    Poland                12
    Switzerland           12
    Name: country, Length: 62, dtype: int64




```python
df2015 = df2015.groupby('country')[['suicides_no']].sum().reset_index()
df2015 = df2015.sort_values(by='suicides_no', ascending=False)
```


```python
df2015
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
      <th>suicides_no</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>60</td>
      <td>United States</td>
      <td>44189</td>
    </tr>
    <tr>
      <td>45</td>
      <td>Russian Federation</td>
      <td>25432</td>
    </tr>
    <tr>
      <td>27</td>
      <td>Japan</td>
      <td>23092</td>
    </tr>
    <tr>
      <td>43</td>
      <td>Republic of Korea</td>
      <td>13510</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Brazil</td>
      <td>11163</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Belize</td>
      <td>26</td>
    </tr>
    <tr>
      <td>48</td>
      <td>Seychelles</td>
      <td>7</td>
    </tr>
    <tr>
      <td>46</td>
      <td>Saint Vincent and Grenadines</td>
      <td>3</td>
    </tr>
    <tr>
      <td>0</td>
      <td>Antigua and Barbuda</td>
      <td>1</td>
    </tr>
    <tr>
      <td>21</td>
      <td>Grenada</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>62 rows × 2 columns</p>
</div>




```python
## Number of suicides in 2015 (the last full year of statistics): 
year_2015 = df[(df['year'] == 2015)]
year_2015 = year_2015.groupby('country')[['suicides_no']].sum().reset_index()

## Sorting values in ascending order
year_2015 = year_2015.sort_values(by='suicides_no', ascending=False)

## Styling output dataframe
year_2015.style.background_gradient(cmap='Reds', subset=['suicides_no'])
```




<style  type="text/css" >
    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row0_col1 {
            background-color:  #67000d;
            color:  #f1f1f1;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row1_col1 {
            background-color:  #f44d38;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row2_col1 {
            background-color:  #f96245;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row3_col1 {
            background-color:  #fca98c;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row4_col1 {
            background-color:  #fcbba1;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row5_col1 {
            background-color:  #fcc2aa;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row6_col1 {
            background-color:  #fdd3c1;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row7_col1 {
            background-color:  #fedbcc;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row8_col1 {
            background-color:  #fee1d3;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row9_col1 {
            background-color:  #fee3d6;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row10_col1 {
            background-color:  #fee5d9;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row11_col1 {
            background-color:  #fee6da;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row12_col1 {
            background-color:  #fee8dd;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row13_col1 {
            background-color:  #feeae0;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row14_col1 {
            background-color:  #feeae0;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row15_col1 {
            background-color:  #feeae1;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row16_col1 {
            background-color:  #ffece4;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row17_col1 {
            background-color:  #ffede5;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row18_col1 {
            background-color:  #ffeee7;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row19_col1 {
            background-color:  #ffeee7;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row20_col1 {
            background-color:  #ffeee7;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row21_col1 {
            background-color:  #ffeee7;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row22_col1 {
            background-color:  #fff0e8;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row23_col1 {
            background-color:  #fff0e8;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row24_col1 {
            background-color:  #fff0e8;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row25_col1 {
            background-color:  #fff0e9;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row26_col1 {
            background-color:  #fff1ea;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row27_col1 {
            background-color:  #fff1ea;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row28_col1 {
            background-color:  #fff1ea;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row29_col1 {
            background-color:  #fff1ea;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row30_col1 {
            background-color:  #fff2eb;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row31_col1 {
            background-color:  #fff2ec;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row32_col1 {
            background-color:  #fff2ec;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row33_col1 {
            background-color:  #fff3ed;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row34_col1 {
            background-color:  #fff3ed;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row35_col1 {
            background-color:  #fff3ed;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row36_col1 {
            background-color:  #fff3ed;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row37_col1 {
            background-color:  #fff4ee;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row38_col1 {
            background-color:  #fff4ee;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row39_col1 {
            background-color:  #fff4ee;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row40_col1 {
            background-color:  #fff4ee;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row41_col1 {
            background-color:  #fff4ee;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row42_col1 {
            background-color:  #fff4ef;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row43_col1 {
            background-color:  #fff4ef;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row44_col1 {
            background-color:  #fff4ef;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row45_col1 {
            background-color:  #fff4ef;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row46_col1 {
            background-color:  #fff4ef;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row47_col1 {
            background-color:  #fff4ef;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row48_col1 {
            background-color:  #fff5f0;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row49_col1 {
            background-color:  #fff5f0;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row50_col1 {
            background-color:  #fff5f0;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row51_col1 {
            background-color:  #fff5f0;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row52_col1 {
            background-color:  #fff5f0;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row53_col1 {
            background-color:  #fff5f0;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row54_col1 {
            background-color:  #fff5f0;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row55_col1 {
            background-color:  #fff5f0;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row56_col1 {
            background-color:  #fff5f0;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row57_col1 {
            background-color:  #fff5f0;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row58_col1 {
            background-color:  #fff5f0;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row59_col1 {
            background-color:  #fff5f0;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row60_col1 {
            background-color:  #fff5f0;
            color:  #000000;
        }    #T_2d2d5260_f43f_11ea_a187_186590dc81a7row61_col1 {
            background-color:  #fff5f0;
            color:  #000000;
        }</style><table id="T_2d2d5260_f43f_11ea_a187_186590dc81a7" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >country</th>        <th class="col_heading level0 col1" >suicides_no</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row0" class="row_heading level0 row0" >60</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row0_col0" class="data row0 col0" >United States</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row0_col1" class="data row0 col1" >44189</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row1" class="row_heading level0 row1" >45</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row1_col0" class="data row1 col0" >Russian Federation</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row1_col1" class="data row1 col1" >25432</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row2" class="row_heading level0 row2" >27</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row2_col0" class="data row2 col0" >Japan</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row2_col1" class="data row2 col1" >23092</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row3" class="row_heading level0 row3" >43</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row3_col0" class="data row3 col0" >Republic of Korea</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row3_col1" class="data row3 col1" >13510</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row4" class="row_heading level0 row4" >7</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row4_col0" class="data row4 col0" >Brazil</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row4_col1" class="data row4 col1" >11163</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row5" class="row_heading level0 row5" >19</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row5_col0" class="data row5 col0" >Germany</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row5_col1" class="data row5 col1" >10088</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row6" class="row_heading level0 row6" >58</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row6_col0" class="data row6 col0" >Ukraine</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row6_col1" class="data row6 col1" >7574</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row7" class="row_heading level0 row7" >35</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row7_col0" class="data row7 col0" >Mexico</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row7_col1" class="data row7 col1" >6234</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row8" class="row_heading level0 row8" >40</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row8_col0" class="data row8 col0" >Poland</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row8_col1" class="data row8 col1" >5420</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row9" class="row_heading level0 row9" >59</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row9_col0" class="data row9 col0" >United Kingdom</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row9_col1" class="data row9 col1" >4910</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row10" class="row_heading level0 row10" >55</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row10_col0" class="data row10 col0" >Thailand</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row10_col1" class="data row10 col1" >4205</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row11" class="row_heading level0 row11" >26</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row11_col0" class="data row11 col0" >Italy</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row11_col1" class="data row11 col1" >3988</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row12" class="row_heading level0 row12" >52</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row12_col0" class="data row12 col0" >Spain</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row12_col1" class="data row12 col1" >3604</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row13" class="row_heading level0 row13" >1</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row13_col0" class="data row13 col0" >Argentina</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row13_col1" class="data row13 col1" >3073</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row14" class="row_heading level0 row14" >3</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row14_col0" class="data row14 col0" >Australia</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row14_col1" class="data row14 col1" >3027</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row15" class="row_heading level0 row15" >28</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row15_col0" class="data row15 col0" >Kazakhstan</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row15_col1" class="data row15 col1" >2872</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row16" class="row_heading level0 row16" >9</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row16_col0" class="data row16 col0" >Colombia</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row16_col1" class="data row16 col1" >2332</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row17" class="row_heading level0 row17" >44</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row17_col0" class="data row17 col0" >Romania</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row17_col1" class="data row17 col1" >2228</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row18" class="row_heading level0 row18" >36</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row18_col0" class="data row18 col0" >Netherlands</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row18_col1" class="data row18 col1" >1873</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row19" class="row_heading level0 row19" >23</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row19_col0" class="data row19 col0" >Hungary</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row19_col1" class="data row19 col1" >1868</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row20" class="row_heading level0 row20" >5</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row20_col0" class="data row20 col0" >Belgium</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row20_col1" class="data row20 col1" >1867</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row21" class="row_heading level0 row21" >8</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row21_col0" class="data row21 col0" >Chile</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row21_col1" class="data row21 col1" >1838</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row22" class="row_heading level0 row22" >56</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row22_col0" class="data row22 col0" >Turkey</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row22_col1" class="data row22 col1" >1532</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row23" class="row_heading level0 row23" >11</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row23_col0" class="data row23 col0" >Cuba</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row23_col1" class="data row23 col1" >1511</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row24" class="row_heading level0 row24" >13</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row24_col0" class="data row24 col0" >Czech Republic</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row24_col1" class="data row24 col1" >1387</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row25" class="row_heading level0 row25" >4</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row25_col0" class="data row25 col0" >Austria</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row25_col1" class="data row25 col1" >1251</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row26" class="row_heading level0 row26" >53</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row26_col0" class="data row26 col0" >Sweden</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row26_col1" class="data row26 col1" >1182</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row27" class="row_heading level0 row27" >54</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row27_col0" class="data row27 col0" >Switzerland</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row27_col1" class="data row27 col1" >1073</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row28" class="row_heading level0 row28" >15</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row28_col0" class="data row28 col0" >Ecuador</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row28_col1" class="data row28 col1" >1073</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row29" class="row_heading level0 row29" >47</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row29_col0" class="data row29 col0" >Serbia</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row29_col1" class="data row29 col1" >1062</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row30" class="row_heading level0 row30" >31</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row30_col0" class="data row30 col0" >Lithuania</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row30_col1" class="data row30 col1" >896</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row31" class="row_heading level0 row31" >10</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row31_col0" class="data row31 col0" >Croatia</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row31_col1" class="data row31 col1" >739</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row32" class="row_heading level0 row32" >17</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row32_col0" class="data row32 col0" >Finland</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row32_col1" class="data row32 col1" >731</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row33" class="row_heading level0 row33" >61</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row33_col0" class="data row33 col0" >Uruguay</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row33_col1" class="data row33 col1" >630</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row34" class="row_heading level0 row34" >38</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row34_col0" class="data row34 col0" >Norway</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row34_col1" class="data row34 col1" >590</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row35" class="row_heading level0 row35" >14</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row35_col0" class="data row35 col0" >Denmark</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row35_col1" class="data row35 col1" >564</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row36" class="row_heading level0 row36" >20</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row36_col0" class="data row36 col0" >Greece</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row36_col1" class="data row36 col1" >529</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row37" class="row_heading level0 row37" >22</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row37_col0" class="data row37 col0" >Guatemala</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row37_col1" class="data row37 col1" >494</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row38" class="row_heading level0 row38" >51</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row38_col0" class="data row38 col0" >South Africa</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row38_col1" class="data row38 col1" >482</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row39" class="row_heading level0 row39" >50</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row39_col0" class="data row39 col0" >Slovenia</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row39_col1" class="data row39 col1" >422</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row40" class="row_heading level0 row40" >29</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row40_col0" class="data row40 col0" >Kyrgyzstan</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row40_col1" class="data row40 col1" >417</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row41" class="row_heading level0 row41" >30</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row41_col0" class="data row41 col0" >Latvia</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row41_col1" class="data row41 col1" >387</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row42" class="row_heading level0 row42" >25</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row42_col0" class="data row42 col0" >Israel</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row42_col1" class="data row42 col1" >342</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row43" class="row_heading level0 row43" >49</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row43_col0" class="data row43 col0" >Singapore</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row43_col1" class="data row43 col1" >329</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row44" class="row_heading level0 row44" >37</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row44_col0" class="data row44 col0" >Nicaragua</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row44_col1" class="data row44 col1" >315</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row45" class="row_heading level0 row45" >41</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row45_col0" class="data row45 col0" >Puerto Rico</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row45_col1" class="data row45 col1" >226</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row46" class="row_heading level0 row46" >16</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row46_col0" class="data row46 col0" >Estonia</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row46_col1" class="data row46 col1" >195</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row47" class="row_heading level0 row47" >18</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row47_col0" class="data row47 col0" >Georgia</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row47_col1" class="data row47 col1" >192</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row48" class="row_heading level0 row48" >57</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row48_col0" class="data row48 col0" >Turkmenistan</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row48_col1" class="data row48 col1" >133</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row49" class="row_heading level0 row49" >39</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row49_col0" class="data row49 col0" >Panama</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row49_col1" class="data row49 col1" >110</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row50" class="row_heading level0 row50" >34</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row50_col0" class="data row50 col0" >Mauritius</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row50_col1" class="data row50 col1" >104</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row51" class="row_heading level0 row51" >2</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row51_col0" class="data row51 col0" >Armenia</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row51_col1" class="data row51 col1" >74</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row52" class="row_heading level0 row52" >42</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row52_col0" class="data row52 col0" >Qatar</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row52_col1" class="data row52 col1" >66</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row53" class="row_heading level0 row53" >32</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row53_col0" class="data row53 col0" >Luxembourg</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row53_col1" class="data row53 col1" >64</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row54" class="row_heading level0 row54" >12</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row54_col0" class="data row54 col0" >Cyprus</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row54_col1" class="data row54 col1" >40</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row55" class="row_heading level0 row55" >24</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row55_col0" class="data row55 col0" >Iceland</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row55_col1" class="data row55 col1" >40</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row56" class="row_heading level0 row56" >33</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row56_col0" class="data row56 col0" >Malta</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row56_col1" class="data row56 col1" >34</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row57" class="row_heading level0 row57" >6</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row57_col0" class="data row57 col0" >Belize</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row57_col1" class="data row57 col1" >26</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row58" class="row_heading level0 row58" >48</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row58_col0" class="data row58 col0" >Seychelles</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row58_col1" class="data row58 col1" >7</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row59" class="row_heading level0 row59" >46</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row59_col0" class="data row59 col0" >Saint Vincent and Grenadines</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row59_col1" class="data row59 col1" >3</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row60" class="row_heading level0 row60" >0</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row60_col0" class="data row60 col0" >Antigua and Barbuda</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row60_col1" class="data row60 col1" >1</td>
            </tr>
            <tr>
                        <th id="T_2d2d5260_f43f_11ea_a187_186590dc81a7level0_row61" class="row_heading level0 row61" >21</th>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row61_col0" class="data row61 col0" >Grenada</td>
                        <td id="T_2d2d5260_f43f_11ea_a187_186590dc81a7row61_col1" class="data row61 col1" >0</td>
            </tr>
    </tbody></table>




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
f,ax = plt.subplots(1,1, figsize=(17,6))
ax = sns.barplot(x = df['year'], y = 'suicides_no', data=df, palette='Spectral')
```


![second pic]({{ site.baseurl }}/images/image19.png)



```python
# Let's investigate the correlation of factors using a heatmap:
f,ax = plt.subplots(1,1,figsize=(10,10))
ax = sns.heatmap(df.corr(),annot=True, cmap='coolwarm')
```


![second pic]({{ site.baseurl }}/images/image20.png)


Nothing really jumping off the page here to be honest. 


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


Ultimately through our analysis we identified a number of interesting insights. Firstly, we can identify that GDP is not a good predictor of suicides. Although the countries with the highest number of suicides have the lowest GDP,  suicide rates don't go up or down as you get to the richer countries on the list. In addition many of the countries that have GDP_for_year below 0.50 have lower suicide rates then countries with much higher GDP. Therefore we can say there is limited evidence of a statistically significant relationship between GDP and suicide rates.

A similar relationship emerged with HDI and suicides, and if anything, as HDI increases suicide numbers also increase (although this must be taken with a significant grain of salt given that we had to fill in the null values for HDI). 

We also found that men commit suicide 3.3x more often then women across the whole data set, but that is even more pronounced by some of the largest countries in the dataset (Brazil/Russia, for example) 

Lastly, we managed to identify that there were two statistically significant clusters of countries (those that commit suicide at higher rates and those that don't) amongst the data set. More analysis would have to be done to identify what ultimately seperates these groups from eachother, however it is clear that it is not necessarily HDI or GDP. Therefore more information would need to be provided regarding demographics, cultural, and sociological factors (race, ethnicity, income inequality, religion, etc.) as well as geographical. 
