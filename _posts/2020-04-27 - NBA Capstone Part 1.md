---
layout: post
title: NBA Project Part 1
---

## Part 2a: EDA and Post-Scraping Clean-up


```python
# Import necessary packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
# in order to look at every column of the dataframes we load in:
pd.set_option('display.max_columns', 100)
```


```python
# Read the csv's in that were scraped in Part 1a:
df20 = pd.read_csv('data/2019-2020.csv')
df19 = pd.read_csv('data/2018-2019.csv')
df18 = pd.read_csv('data/2017-2018.csv')
df17 = pd.read_csv('data/2016-2017.csv')
df16 = pd.read_csv('data/2015-2016.csv')
df15 = pd.read_csv('data/2014-2015.csv')
df14 = pd.read_csv('data/2013-2014.csv')
df13 = pd.read_csv('data/2012-2013.csv')
df12 = pd.read_csv('data/2011-2012.csv')
df11 = pd.read_csv('data/2010-2011.csv')
df10 = pd.read_csv('data/2009-2010.csv')
df09 = pd.read_csv('data/2008-2009.csv')
df08 = pd.read_csv('data/2007-2008.csv')
df07 = pd.read_csv('data/2006-2007.csv')
df06 = pd.read_csv('data/2005-2006.csv')
df05 = pd.read_csv('data/2004-2005.csv')
df04 = pd.read_csv('data/2003-2004.csv')
df03 = pd.read_csv('data/2002-2003.csv')
df02 = pd.read_csv('data/2001-2002.csv')
df01 = pd.read_csv('data/2000-2001.csv')
df00 = pd.read_csv('data/1999-2000.csv')
# Get rid of the 'Unnamed: 0', column that was a result of combining our scraped tables together in Part 1a:
df20 = df20.drop(['Unnamed: 0'], axis=1)
df19 = df19.drop(['Unnamed: 0'], axis=1)
df18 = df18.drop(['Unnamed: 0'], axis=1)
df17 = df17.drop(['Unnamed: 0'], axis=1)
df16 = df16.drop(['Unnamed: 0'], axis=1)
df15 = df15.drop(['Unnamed: 0'], axis=1)
df14 = df14.drop(['Unnamed: 0'], axis=1)
df13 = df13.drop(['Unnamed: 0'], axis=1)
df12 = df12.drop(['Unnamed: 0'], axis=1)
df11 = df11.drop(['Unnamed: 0'], axis=1)
df10 = df10.drop(['Unnamed: 0'], axis=1)
df09 = df09.drop(['Unnamed: 0'], axis=1)
df08 = df08.drop(['Unnamed: 0'], axis=1)
df07 = df07.drop(['Unnamed: 0'], axis=1)
df06 = df06.drop(['Unnamed: 0'], axis=1)
df05 = df05.drop(['Unnamed: 0'], axis=1)
df04 = df04.drop(['Unnamed: 0'], axis=1)
df03 = df03.drop(['Unnamed: 0'], axis=1)
df02 = df02.drop(['Unnamed: 0'], axis=1)
df01 = df01.drop(['Unnamed: 0'], axis=1)
df00 = df00.drop(['Unnamed: 0'], axis=1)
```


```python
# the df20 dataframe has some unique attributes given that there are contract values included (used for the 
# unsupervised learning portion):
df20.drop(['2019-20', 'Guaranteed'], axis=1, inplace=True)
# Convert our All-star dtype to int just to make sure that all 1's are stored as integers and not having some of them
# saved as strings:
df20['All-star'] = df20['All-star'].astype(int)
```

One of the cleaning decisions that still needs to be made is regarding what to do with traded players. Currently, in each of the dataframes that are saved (from df00 to df20) if a player is traded in a given season, three rows exist for this player (the row for the total, the row for the first team, and the row for the second team). 

There are two options to deal with these rows:  
- (1) we can get rid of the total rows for each player and keep the rows for each seperate team, or 
- (2) we can keep only the total row values for players that were traded, and then drop the team column altogther.

Ultimately, option (2) seems like the better choice, given that keeping multiple rows for the same player in a given season will essentially double count them. Also in the rare yet possible cases that a player is traded in a given season and makes the all-star game, we are not double counting them as an all-star with option (2). 


```python
# Let's get rid of all the duplicate rows that have the same player name, in order to not double or triple count 
# players that were traded during the season. First, lets do it for only 2019 so we confirm that the correct rows are dropped
# using this method: 
df19_dropped = df19.drop_duplicates(subset='Player', keep='first')
```


```python
# As we can see we can confirm the previous step was done correctly cause there are 86 players who are on the team 
# "Total" which of course is not a real team, these are all the players that were traded mid season and since we are 
# dropping all the duplicates where we see multiple names, and we are keeping the first row above, then we see that 
# the total column was kept and the other two columns would have been dropped. Assuming no players were traded 
# three times in the season we can assume that there were 86*2 = 172 rows dropped.
df19['Tm'].value_counts()
```




    TOT    86
    MEM    28
    CLE    27
    PHI    26
    WAS    25
    MIL    24
    PHO    24
    NYK    23
    HOU    23
    LAC    22
    ATL    22
    LAL    22
    TOR    22
    CHI    22
    MIN    21
    DAL    21
    SAC    20
    NOP    20
    DET    20
    BRK    19
    OKC    18
    POR    18
    DEN    18
    UTA    18
    MIA    18
    IND    17
    GSW    17
    ORL    17
    CHO    17
    BOS    17
    SAS    16
    Name: Tm, dtype: int64




```python
print(df19.shape)
print(df19_dropped.shape)
print(708-172)
```

    (708, 53)
    (530, 53)
    536


As we can see from the above, there are still 6 rows of a discrepency, however it is possible that this is do to players who get traded more then once in the season:


```python
# Since players that are traded once would have a total row, the first team row and then the second row, we are 
# setting the player_counts we want to look at to be greater then 3, therefore players that played on 3 seperate teams
# throughout the season (since they would have 4 rows: total, team 1, team 2, and team 3)
player_counts = df19['Player'].value_counts()
player_list = player_counts[player_counts > 3].index.tolist()
df19_trades = df19[df19['Player'].isin(player_list)]

df19_trades['Player'].value_counts()
```




    Andrew Harrison    4
    Isaiah Canaan      4
    Wesley Matthews    4
    Greg Monroe        4
    Jason Smith        4
    Alec Burks         4
    Name: Player, dtype: int64



As we can see there are 6 players who unfortunately had to play on three seperate teams throughout the season, poor guys.... however this makes exact sense and shows that our method for getting rid of duplicates worked perfectly. We can now use the same method for each season 


```python
df20 = df20.drop_duplicates(subset='Player', keep='first')
# do it for 2019 for real now:
df19 = df19.drop_duplicates(subset='Player', keep='first')
# and the others in the set:
df18 = df18.drop_duplicates(subset='Player', keep='first')
df17 = df17.drop_duplicates(subset='Player', keep='first')
df16 = df16.drop_duplicates(subset='Player', keep='first')
df15 = df15.drop_duplicates(subset='Player', keep='first')
df14 = df14.drop_duplicates(subset='Player', keep='first')
df13 = df13.drop_duplicates(subset='Player', keep='first')
df12 = df12.drop_duplicates(subset='Player', keep='first')
df11 = df11.drop_duplicates(subset='Player', keep='first')
df10 = df10.drop_duplicates(subset='Player', keep='first')
df09 = df09.drop_duplicates(subset='Player', keep='first')
df08 = df08.drop_duplicates(subset='Player', keep='first')
df07 = df07.drop_duplicates(subset='Player', keep='first')
df06 = df06.drop_duplicates(subset='Player', keep='first')
df05 = df05.drop_duplicates(subset='Player', keep='first')
df04 = df04.drop_duplicates(subset='Player', keep='first')
df03 = df03.drop_duplicates(subset='Player', keep='first')
df02 = df02.drop_duplicates(subset='Player', keep='first')
df01 = df01.drop_duplicates(subset='Player', keep='first')
df00 = df00.drop_duplicates(subset='Player', keep='first')
```


```python
# Let's combine the years into eras before doing anymore cleaning as it will involve less steps to just break them up
# into three:
df14_20 = pd.concat([df14, df15, df16, df17, df18, df19, df20], axis=0, sort=False)
df07_13 = pd.concat([df07, df08, df09, df10, df11, df12, df13], axis=0, sort=False) 
df00_06 = pd.concat([df00, df01, df02, df03, df04, df05, df06], axis=0, sort=False) 
# Need to reset the indexes 
df14_20.reset_index(drop=True, inplace=True)
df07_13.reset_index(drop=True, inplace=True)
df00_06.reset_index(drop=True, inplace=True)
```


```python
df14_20['All-star'] = df14_20['All-star'].astype(int)
df07_13['All-star'] = df07_13['All-star'].astype(int)
df00_06['All-star'] = df00_06['All-star'].astype(int)
```


```python
# As we can see the classification problem is very biased ( a lot more 0's then 1's), which makes sense given the 
# number of players that make the all-star team vs the number of players who do not.

print(df14_20['All-star'].value_counts())
print(df07_13['All-star'].value_counts())
print(df00_06['All-star'].value_counts())
```

    0    3270
    1     184
    Name: All-star, dtype: int64
    0    3008
    1     185
    Name: All-star, dtype: int64
    0    2936
    1     176
    Name: All-star, dtype: int64



```python
# Now it is important that we define the threshold for a player essentially making into the dataset. Given that 
# some players get called up from the NBA development league (the G-league) throughout the season as team need requires,
# More then a quarter of the season played (82 games in a season): 
df14_20 = df14_20[df14_20['G']>20]
# and more then 8 mpg: 
df14_20 = df14_20[df14_20['MP']>8.0]

# For 2007-2013 as well: 
df07_13 = df07_13[df07_13['G']>20]
df07_13 = df07_13[df07_13['MP']>8.0]

# For 2000-2006 as well: 
df00_06 = df00_06[df00_06['G']>20]
df00_06 = df00_06[df00_06['MP']>8.0]
```


```python
# Interestingly the reason that the number of all-stars goes down by 1 for both 2014-2020 and 2007-2013 is because
# kobe bryant made the all star game in ____ despite playing less then 20 games in the season. Yao ming did the same
# in the 2010 season. 

print(df14_20['All-star'].value_counts())
print(df07_13['All-star'].value_counts())
print(df00_06['All-star'].value_counts())
```

    0    2554
    1     183
    Name: All-star, dtype: int64
    0    2427
    1     183
    Name: All-star, dtype: int64
    0    2318
    1     174
    Name: All-star, dtype: int64



```python
percent_missing14_20 = df14_20.isnull().sum() * 100 / len(df14_20)
null_df = pd.DataFrame({'column_name': df14_20.columns,
                                 'percent_missing': percent_missing14_20})
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
      <td>3P%</td>
      <td>3P%</td>
      <td>5.809280</td>
    </tr>
    <tr>
      <td>FT%</td>
      <td>FT%</td>
      <td>0.036536</td>
    </tr>
    <tr>
      <td>Drop 1</td>
      <td>Drop 1</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Drop 2</td>
      <td>Drop 2</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
percent_missing = df07_13.isnull().sum() * 100 / len(df07_13)
null_df = pd.DataFrame({'column_name': df07_13.columns,
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
      <td>3P%</td>
      <td>3P%</td>
      <td>10.651341</td>
    </tr>
    <tr>
      <td>FT%</td>
      <td>FT%</td>
      <td>0.076628</td>
    </tr>
    <tr>
      <td>Drop 1</td>
      <td>Drop 1</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Drop 2</td>
      <td>Drop 2</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
percent_missing = df00_06.isnull().sum() * 100 / len(df00_06)
null_df = pd.DataFrame({'column_name': df00_06.columns,
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
      <td>3P%</td>
      <td>3P%</td>
      <td>12.399679</td>
    </tr>
    <tr>
      <td>Drop 1</td>
      <td>Drop 1</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Drop 2</td>
      <td>Drop 2</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df14_20.drop(['Player', 'Tm', 'Drop 1', 'Drop 2', 'Year'], axis=1, inplace=True)
df07_13.drop(['Player', 'Tm', 'Drop 1', 'Drop 2', 'Year'], axis=1, inplace=True)
df00_06.drop(['Player', 'Tm', 'Drop 1', 'Drop 2', 'Year'], axis=1, inplace=True)
```


```python
# Now we can look at the real null figures remaining: 
percent_missing = df14_20.isnull().sum() * 100 / len(df14_20)
null_df = pd.DataFrame({'column_name': df14_20.columns,
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
      <td>3P%</td>
      <td>3P%</td>
      <td>5.809280</td>
    </tr>
    <tr>
      <td>FT%</td>
      <td>FT%</td>
      <td>0.036536</td>
    </tr>
  </tbody>
</table>
</div>




```python
percent_missing = df07_13.isnull().sum() * 100 / len(df07_13)
null_df = pd.DataFrame({'column_name': df07_13.columns,
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
      <td>3P%</td>
      <td>3P%</td>
      <td>10.651341</td>
    </tr>
    <tr>
      <td>FT%</td>
      <td>FT%</td>
      <td>0.076628</td>
    </tr>
  </tbody>
</table>
</div>




```python
percent_missing = df00_06.isnull().sum() * 100 / len(df00_06)
null_df = pd.DataFrame({'column_name': df00_06.columns,
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
      <td>3P%</td>
      <td>3P%</td>
      <td>12.399679</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Given that 3p% being null is due to centers in the nba who don't shoot them at all, and therefore record a null 3p% 
# stat, we will fill them in with 0's:
df14_20['3P%'] = df14_20['3P%'].fillna(0)
df07_13['3P%'] = df07_13['3P%'].fillna(0)
df00_06['3P%'] = df00_06['3P%'].fillna(0)

# Given that Ft% makes up such a small percentage of nulls in the set, we will fill them in with the mean of the nba
# for these rows. 
df14_20['FT%'] = df14_20['FT%'].fillna(df14_20['FT%'].mean())
df07_13['FT%'] = df07_13['FT%'].fillna(df14_20['FT%'].mean())
```


```python
# Another decision that needs to be made is in regards to positions, currently the data is saved as such: 
df14_20['Pos'].value_counts()
```




    SG       594
    PG       552
    PF       542
    C        540
    SF       481
    SF-SG      7
    PF-SF      5
    SG-SF      3
    SG-PG      3
    C-PF       3
    PG-SG      2
    PF-C       2
    SF-PF      2
    SG-PF      1
    Name: Pos, dtype: int64



Ultimately the best way to deal with this seems to be to remove the secondary position altogether, as it is inconsistent whether a player has more then one listed position. For the purposes of the modelling in this exercise, let's try to isolate the 5 positions, and assume each player mostly plays their primary position.  


```python
df14_20['Pos'] = df14_20['Pos'].astype(str)
```


```python
df14_20['Pos'] = df14_20['Pos'].str.replace(r'-\w\w', '')
df14_20['Pos'] = df14_20['Pos'].str.replace(r'-\w', '')
```


```python
df14_20['Pos'].value_counts()
```




    SG    601
    PG    554
    PF    549
    C     543
    SF    490
    Name: Pos, dtype: int64




```python
# Now let's do it for the other eras: 
# 2007-2013: 
df07_13['Pos'] = df07_13['Pos'].str.replace(r'-\w\w', '')
df07_13['Pos'] = df07_13['Pos'].str.replace(r'-\w', '')

# 2000-2006: 
df00_06['Pos'] = df00_06['Pos'].str.replace(r'-\w\w', '')
df00_06['Pos'] = df00_06['Pos'].str.replace(r'-\w', '')

```


```python
# now the only remaining categorical variables are Position and Year. Let's use pd.get_dummies to encode them 
# individually. drop_first = true because its not necessary to have a column for every option:
df14_20 = pd.get_dummies(df14_20, drop_first=True)
df07_13 = pd.get_dummies(df07_13, drop_first=True)
df00_06 = pd.get_dummies(df00_06, drop_first=True)
```

Part 2b: Modelling


```python
# Filter warnings as many of the models give many unnecessary warnings: 
import warnings
warnings.filterwarnings('ignore')
```


```python
# Now let's state the X and y variables clearly in order for modelling: 
y20 = df14_20['All-star']
X20 = df14_20.drop(['All-star'], axis=1)

y13 = df07_13['All-star']
X13 = df07_13.drop(['All-star'], axis=1)

X06 = df00_06.drop(['All-star'], axis=1)
y06 = df00_06['All-star']
```


```python
from sklearn.model_selection import train_test_split
# Split up our data sets into remainder and test set:
# For the 2014-2020 dataset: 
X_remainder20, X_test20, y_remainder20, y_test20 = train_test_split(X20, y20, test_size=0.25, 
                                                            random_state=1)

# For the 2007-2013 dataset: 
X_remainder13, X_test13, y_remainder13, y_test13 = train_test_split(X13, y13, test_size=0.25, 
                                                            random_state=1)

# For the 2000-2006 dataset:
X_remainder06, X_test06, y_remainder06, y_test06 = train_test_split(X06, y06, test_size=0.25, 
                                                            random_state=1)
```


```python
from imblearn.over_sampling import SMOTE
# The main parameter we can set with SMOTE, is k_neighbors, which is the number of nearest neighbours used to 
# construct synthetic samples, for the purposes of this model, let's pick 2:
#Intiate SMOTE for 2014-2020:
smote = SMOTE(k_neighbors=2, n_jobs = -1, random_state=1)
X_smote20, y_smote20 = smote.fit_resample(X_remainder20,y_remainder20)

# Save X_smote as a df in order to pull out feature names:
X_smote20 = pd.DataFrame(X_smote20, columns = X_remainder20.columns)

# Show how the smote works effectively as another way of oversampling:
unique, counts = np.unique(y_smote20, return_counts = True)
print(np.asarray((unique, counts)).T)
print()


# and for 2007-2013: 
smote2 = SMOTE(k_neighbors=2, n_jobs = -1, random_state=1)
X_smote13, y_smote13 = smote2.fit_resample(X_remainder13,y_remainder13)

# Save X_smote as a df in order to pull out feature names:
X_smote13 = pd.DataFrame(X_smote13, columns = X13.columns)

# Show how the smote works effectively as another way of oversampling:
unique, counts = np.unique(y_smote13, return_counts = True)
print(np.asarray((unique, counts)).T)
print()

# and for 2000-2006: 
smote3 = SMOTE(k_neighbors=2, n_jobs = -1, random_state=1)
X_smote06, y_smote06 = smote3.fit_resample(X_remainder06,y_remainder06)

# Save X_smote as a df in order to pull out feature names:
X_smote06 = pd.DataFrame(X_smote06, columns = X06.columns)

# Show how the smote works effectively as another way of oversampling:
unique, counts = np.unique(y_smote06, return_counts = True)
print(np.asarray((unique, counts)).T)
```

    [[   0 1914]
     [   1 1914]]
    
    [[   0 1806]
     [   1 1806]]
    
    [[   0 1742]
     [   1 1742]]



```python
# For the 2014-2020 dataset: 
X_train20, X_validation20, y_train20, y_validation20 = train_test_split(X_smote20, y_smote20, test_size=0.2, 
                                                            random_state=1, stratify=y_smote20)

# For the 2007 - 2013 dataset: 
X_train13, X_validation13, y_train13, y_validation13 = train_test_split(X_smote13, y_smote13, test_size=0.2, 
                                                            random_state=1, stratify=y_smote13)

# For the 2000-2006 dataset: 
X_train06, X_validation06, y_train06, y_validation06 = train_test_split(X_smote06, y_smote06, test_size=0.2, 
                                                            random_state=1, stratify=y_smote06)
```


```python
from sklearn.preprocessing import StandardScaler
# First dataset: 
scaler = StandardScaler()
scaler = scaler.fit(X_train20)

# transform X_train, however put X_train in to a df in order to easier pull out the coefficients:
scaled_features = scaler.transform(X_train20.values)
X_train20 = pd.DataFrame(scaled_features, index=X_train20.index, columns=X_train20.columns)

# transform X_test and X_validation as well:
X_test20 = scaler.transform(X_test20)
X_validation20 = scaler.transform(X_validation20)



# 2nd Dataset: 
scaler2 = StandardScaler()
scaler2 = scaler2.fit(X_train13)

scaled_features = scaler2.transform(X_train13.values)
X_train13 = pd.DataFrame(scaled_features, index=X_train13.index, columns=X_train13.columns)

X_test13 = scaler2.transform(X_test13)
X_validation13 = scaler2.transform(X_validation13)


# 3rd Dataset: 
scaler3 = StandardScaler()
scaler3 = scaler3.fit(X_train06)

scaled_features = scaler3.transform(X_train06.values)
X_train06 = pd.DataFrame(scaled_features, index=X_train06.index, columns=X_train06.columns)

X_test06 = scaler2.transform(X_test06)
X_validation06 = scaler3.transform(X_validation06)
```


```python
from sklearn.linear_model import LogisticRegression
# Now let's find the optimal C value for our logistic regression model:
C_range = np.array([0.00001, 0.0001, 0.001, 0.01,0.1,1,10,100,1000, 10000])
validation_scores=[]
train_scores=[]
# For loop going through different possible regularization parameters. The lower the value the more regularization,
# given that C is the inverse of regularization. 
for c in C_range:
    # Instantiate and fit a Logistic Regression model to the data. Solver set to 'lbfgs'. n_jobs=-1 in order to 
    # improve processing speeds. 
    myLogc = LogisticRegression(C=c, solver='lbfgs', n_jobs=-1, random_state=1)
    myLogc.fit(X_train20,y_train20)
    
    # append results to score lists for plotting: 
    train_scores.append(myLogc.score(X_train20,y_train20))
    validation_scores.append(myLogc.score(X_validation20,y_validation20))
```


```python
plt.figure()
plt.plot(C_range, train_scores, label='Train accuracies', marker='.')
plt.plot(C_range, validation_scores, label="Validation accuracies",marker='.')
plt.legend()
plt.xscale("log")
plt.xlabel('Regularization Parameter: C')
plt.ylabel('Accuracy Score')
plt.title('Optimal Regularization Parameter for LogReg')
plt.grid()
plt.show();
```


![png](output_41_0.png)



```python
from sklearn.linear_model import LogisticRegression
# Now let's find the optimal C value for our logistic regression model:
C_range = np.array([0.00001, 0.0001, 0.001, 0.01,0.1,1,10,100,1000, 10000])
validation_scores13=[]
train_scores13=[]
# For loop going through different possible regularization parameters. The lower the value the more regularization,
# given that C is the inverse of regularization. 
for c in C_range:
    # Instantiate and fit a Logistic Regression model to the data. Solver set to 'lbfgs'. n_jobs=-1 in order to 
    # improve processing speeds. 
    myLog2 = LogisticRegression(C=c, solver='lbfgs', n_jobs=-1, random_state=1)
    myLog2.fit(X_train13,y_train13)
    
    # append results to score lists for plotting: 
    train_scores13.append(myLog2.score(X_train13,y_train13))
    validation_scores13.append(myLog2.score(X_validation13,y_validation13))
```


```python
plt.figure()
plt.plot(C_range, train_scores13, label='Train accuracies', marker='.')
plt.plot(C_range, validation_scores13, label="Validation accuracies",marker='.')
plt.legend()
plt.xscale("log")
plt.xlabel('Regularization Parameter: C')
plt.ylabel('Accuracy Score')
plt.title('Optimal Regularization Parameter for LogReg')
plt.grid()
plt.show();
```


![png](output_43_0.png)



```python
from sklearn.linear_model import LogisticRegression
# Now let's find the optimal C value for our logistic regression model:
C_range = np.array([0.00001, 0.0001, 0.001, 0.01,0.1,1,10,100,1000, 10000])
validation_scores06=[]
train_scores06=[]
# For loop going through different possible regularization parameters. The lower the value the more regularization,
# given that C is the inverse of regularization. 
for c in C_range:
    # Instantiate and fit a Logistic Regression model to the data. Solver set to 'lbfgs'. n_jobs=-1 in order to 
    # improve processing speeds. 
    myLog3 = LogisticRegression(C=c, solver='lbfgs', n_jobs=-1, random_state=1)
    myLog3.fit(X_train06,y_train06)
    
    # append results to score lists for plotting: 
    train_scores06.append(myLog3.score(X_train06,y_train06))
    validation_scores06.append(myLog3.score(X_validation06,y_validation06))
```


```python
plt.figure()
plt.plot(C_range, train_scores06, label='Train accuracies', marker='.')
plt.plot(C_range, validation_scores06, label="Validation accuracies",marker='.')
plt.legend()
plt.xscale("log")
plt.xlabel('Regularization Parameter: C')
plt.ylabel('Accuracy Score')
plt.title('Optimal Regularization Parameter for LogReg')
plt.grid()
plt.show();
```


![png](output_45_0.png)



```python
from sklearn.model_selection import cross_val_score

# Since we are going to use cross validation to determine the best parameter for C, we will perform a similar loop as
# before, looping through a range of possible C values:
C_range = np.array([0.00001, 0.0001, 0.001, 0.01,0.1,1,10,100,1000, 10000])
cv_scores20=[]
# loop over different C_values in order to find the optimal value for 5 crossfolds. (n_jobs=-1 resulted in an error
# for this block of code):
for c in C_range:
    myLogcv = LogisticRegression(C=c, solver='lbfgs', random_state=1)
    cv_score = np.mean(cross_val_score(myLogcv, X_smote20, y_smote20, cv=5))
    cv_scores20.append(cv_score)

plt.figure()
plt.plot(C_range, cv_scores20, label="Cross Validation Score",marker='.')
plt.legend()
plt.xscale("log")
plt.xlabel('Regularization Parameter: C')
plt.ylabel('Cross Validation Score')
plt.title('Cross Validation Scores for C values')
plt.grid()
plt.show();
```


![png](output_46_0.png)



```python
C_range = np.array([0.00001, 0.0001, 0.001, 0.01,0.1,1,10,100,1000, 10000])
cv_scores13=[]
# loop over different C_values in order to find the optimal value for 5 crossfolds. (n_jobs=-1 resulted in an error
# for this block of code):
for c in C_range:
    myLogcv = LogisticRegression(C=c, solver='lbfgs', random_state=1)
    cv_score = np.mean(cross_val_score(myLogcv, X_smote13, y_smote13, cv=5))
    cv_scores13.append(cv_score)

plt.figure()
plt.plot(C_range, cv_scores13, label="Cross Validation Score",marker='.')
plt.legend()
plt.xscale("log")
plt.xlabel('Regularization Parameter: C')
plt.ylabel('Cross Validation Score')
plt.title('Cross Validation Scores for C values')
plt.grid()
plt.show();
```


![png](output_47_0.png)



```python
C_range = np.array([0.00001, 0.0001, 0.001, 0.01,0.1,1,10,100,1000, 10000])
cv_scores06=[]
# loop over different C_values in order to find the optimal value for 5 crossfolds. (n_jobs=-1 resulted in an error
# for this block of code):
for c in C_range:
    myLogcv = LogisticRegression(C=c, solver='lbfgs', random_state=1)
    cv_score = np.mean(cross_val_score(myLogcv, X_smote06, y_smote06, cv=5))
    cv_scores06.append(cv_score)

plt.figure()
plt.plot(C_range, cv_scores06, label="Cross Validation Score",marker='.')
plt.legend()
plt.xscale("log")
plt.xlabel('Regularization Parameter: C')
plt.ylabel('Cross Validation Score')
plt.title('Cross Validation Scores for C values')
plt.grid()
plt.show();
```


![png](output_48_0.png)



```python
# From the above, it makes sense to use a C value of 10^-1 for all three model. In addition, setting C values lower
# could help mitigate some overfitting issues, as higher C-values generally run a higher risk of overfitting on the
# train set. Therefore let's re-instantiate each of the three models with C=0.1:

myLog = LogisticRegression(C=0.01, solver='lbfgs', n_jobs=-1, random_state=1)
myLog.fit(X_smote20,y_smote20)
print("2014-20: ")
print("Test Accuracy: ", myLog.score(X_test20,y_test20))

myLog2 = LogisticRegression(C=0.01, solver='lbfgs', n_jobs=-1, random_state=1)
myLog2.fit(X_smote13,y_smote13)
print("2007-2013: ")
print("Test Accuracy: ", myLog.score(X_test13,y_test13))

myLog3 = LogisticRegression(C=0.01, solver='lbfgs', n_jobs=-1, random_state=1)
myLog3.fit(X_smote06, y_smote06)
print("2000-2006:")
print("Test Accuracy: ", myLog.score(X_test06,y_test06))
```

    2014-20: 
    Test Accuracy:  0.9284671532846716
    2007-2013: 
    Test Accuracy:  0.9464012251148545
    2000-2006:
    Test Accuracy:  0.9085072231139647


Feature Coefficients:


```python
coefficients = pd.concat([pd.DataFrame(X_smote20.columns),pd.DataFrame(np.transpose(myLog.coef_))], axis = 1)
coefficients.columns=('Feature', 'Coefficients')
Coef14_20 = coefficients.groupby('Coefficients').sum().sort_values('Coefficients', ascending=False)
Coef14_20
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
      <th>Feature</th>
    </tr>
    <tr>
      <th>Coefficients</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0.468161</td>
      <td>WS</td>
    </tr>
    <tr>
      <td>0.378388</td>
      <td>DWS</td>
    </tr>
    <tr>
      <td>0.299855</td>
      <td>AST</td>
    </tr>
    <tr>
      <td>0.242020</td>
      <td>DRB</td>
    </tr>
    <tr>
      <td>0.233175</td>
      <td>VORP</td>
    </tr>
    <tr>
      <td>0.211283</td>
      <td>TRB</td>
    </tr>
    <tr>
      <td>0.170539</td>
      <td>BPM</td>
    </tr>
    <tr>
      <td>0.165742</td>
      <td>Age</td>
    </tr>
    <tr>
      <td>0.163225</td>
      <td>PTS</td>
    </tr>
    <tr>
      <td>0.157497</td>
      <td>FGA</td>
    </tr>
    <tr>
      <td>0.144974</td>
      <td>OBPM</td>
    </tr>
    <tr>
      <td>0.142807</td>
      <td>BLK%</td>
    </tr>
    <tr>
      <td>0.120816</td>
      <td>2PA</td>
    </tr>
    <tr>
      <td>0.120646</td>
      <td>TOV</td>
    </tr>
    <tr>
      <td>0.114022</td>
      <td>FTA</td>
    </tr>
    <tr>
      <td>0.098417</td>
      <td>Pos_PF</td>
    </tr>
    <tr>
      <td>0.088780</td>
      <td>BLK</td>
    </tr>
    <tr>
      <td>0.076086</td>
      <td>2P</td>
    </tr>
    <tr>
      <td>0.075665</td>
      <td>OWS</td>
    </tr>
    <tr>
      <td>0.075276</td>
      <td>USG%</td>
    </tr>
    <tr>
      <td>0.072307</td>
      <td>Pos_SG</td>
    </tr>
    <tr>
      <td>0.065803</td>
      <td>FG</td>
    </tr>
    <tr>
      <td>0.055247</td>
      <td>STL</td>
    </tr>
    <tr>
      <td>0.054403</td>
      <td>FT</td>
    </tr>
    <tr>
      <td>0.046391</td>
      <td>GS</td>
    </tr>
    <tr>
      <td>0.044962</td>
      <td>TRB%</td>
    </tr>
    <tr>
      <td>0.038551</td>
      <td>3PA</td>
    </tr>
    <tr>
      <td>0.031480</td>
      <td>DBPM</td>
    </tr>
    <tr>
      <td>0.013617</td>
      <td>AST%</td>
    </tr>
    <tr>
      <td>-0.001682</td>
      <td>FTr</td>
    </tr>
    <tr>
      <td>-0.003022</td>
      <td>WS/48</td>
    </tr>
    <tr>
      <td>-0.012095</td>
      <td>3P</td>
    </tr>
    <tr>
      <td>-0.020622</td>
      <td>STL%</td>
    </tr>
    <tr>
      <td>-0.020743</td>
      <td>3P%</td>
    </tr>
    <tr>
      <td>-0.023376</td>
      <td>3PAr</td>
    </tr>
    <tr>
      <td>-0.025624</td>
      <td>FG%</td>
    </tr>
    <tr>
      <td>-0.028118</td>
      <td>ORB</td>
    </tr>
    <tr>
      <td>-0.028805</td>
      <td>2P%</td>
    </tr>
    <tr>
      <td>-0.030979</td>
      <td>eFG%</td>
    </tr>
    <tr>
      <td>-0.032322</td>
      <td>TS%</td>
    </tr>
    <tr>
      <td>-0.045667</td>
      <td>FT%</td>
    </tr>
    <tr>
      <td>-0.073791</td>
      <td>DRB%</td>
    </tr>
    <tr>
      <td>-0.077125</td>
      <td>G</td>
    </tr>
    <tr>
      <td>-0.085739</td>
      <td>Pos_PG</td>
    </tr>
    <tr>
      <td>-0.090097</td>
      <td>TOV%</td>
    </tr>
    <tr>
      <td>-0.101489</td>
      <td>Pos_SF</td>
    </tr>
    <tr>
      <td>-0.152194</td>
      <td>PF</td>
    </tr>
    <tr>
      <td>-0.189832</td>
      <td>ORB%</td>
    </tr>
    <tr>
      <td>-0.320805</td>
      <td>MP</td>
    </tr>
    <tr>
      <td>-0.388919</td>
      <td>PER</td>
    </tr>
  </tbody>
</table>
</div>



So we see from the above that the top 10 variables, that were most correlated with making the all star team were:  
1. Win Shares
2. Defensive Winshares
3. Assists 
4. Defensive Rebounds
5. Value Over Replacement Player

The 5 features most negatively affecting the all star team are: 
1. PER
2. Minutes Played
3. ORB%
4. Personal Fouls
5. Position - Small Forward


```python
coefficients = pd.concat([pd.DataFrame(X_smote13.columns),pd.DataFrame(np.transpose(myLog2.coef_))], axis = 1)
coefficients.columns=('Feature', 'Coefficients')
Coef07_13 = coefficients.groupby('Coefficients').sum().sort_values('Coefficients', ascending=False)
Coef07_13
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
      <th>Feature</th>
    </tr>
    <tr>
      <th>Coefficients</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0.515929</td>
      <td>WS</td>
    </tr>
    <tr>
      <td>0.387232</td>
      <td>DWS</td>
    </tr>
    <tr>
      <td>0.194692</td>
      <td>TRB</td>
    </tr>
    <tr>
      <td>0.190574</td>
      <td>GS</td>
    </tr>
    <tr>
      <td>0.171285</td>
      <td>DRB</td>
    </tr>
    <tr>
      <td>0.162878</td>
      <td>VORP</td>
    </tr>
    <tr>
      <td>0.160245</td>
      <td>2PA</td>
    </tr>
    <tr>
      <td>0.127182</td>
      <td>PTS</td>
    </tr>
    <tr>
      <td>0.122552</td>
      <td>AST</td>
    </tr>
    <tr>
      <td>0.120342</td>
      <td>OWS</td>
    </tr>
    <tr>
      <td>0.108189</td>
      <td>TOV</td>
    </tr>
    <tr>
      <td>0.101118</td>
      <td>OBPM</td>
    </tr>
    <tr>
      <td>0.095543</td>
      <td>Age</td>
    </tr>
    <tr>
      <td>0.087656</td>
      <td>BPM</td>
    </tr>
    <tr>
      <td>0.084609</td>
      <td>FGA</td>
    </tr>
    <tr>
      <td>0.081890</td>
      <td>FTA</td>
    </tr>
    <tr>
      <td>0.067940</td>
      <td>2P</td>
    </tr>
    <tr>
      <td>0.056882</td>
      <td>FG</td>
    </tr>
    <tr>
      <td>0.051718</td>
      <td>FT</td>
    </tr>
    <tr>
      <td>0.050971</td>
      <td>Pos_SG</td>
    </tr>
    <tr>
      <td>0.043287</td>
      <td>ORB</td>
    </tr>
    <tr>
      <td>0.043223</td>
      <td>BLK%</td>
    </tr>
    <tr>
      <td>0.031733</td>
      <td>ORB%</td>
    </tr>
    <tr>
      <td>0.031594</td>
      <td>AST%</td>
    </tr>
    <tr>
      <td>0.027682</td>
      <td>BLK</td>
    </tr>
    <tr>
      <td>0.020748</td>
      <td>TOV%</td>
    </tr>
    <tr>
      <td>0.005164</td>
      <td>TRB%</td>
    </tr>
    <tr>
      <td>0.003707</td>
      <td>3P%</td>
    </tr>
    <tr>
      <td>0.000975</td>
      <td>WS/48</td>
    </tr>
    <tr>
      <td>-0.003910</td>
      <td>USG%</td>
    </tr>
    <tr>
      <td>-0.003984</td>
      <td>FTr</td>
    </tr>
    <tr>
      <td>-0.011919</td>
      <td>3P</td>
    </tr>
    <tr>
      <td>-0.012563</td>
      <td>3PAr</td>
    </tr>
    <tr>
      <td>-0.017018</td>
      <td>DBPM</td>
    </tr>
    <tr>
      <td>-0.019462</td>
      <td>FT%</td>
    </tr>
    <tr>
      <td>-0.020061</td>
      <td>FG%</td>
    </tr>
    <tr>
      <td>-0.021823</td>
      <td>eFG%</td>
    </tr>
    <tr>
      <td>-0.023454</td>
      <td>TS%</td>
    </tr>
    <tr>
      <td>-0.023760</td>
      <td>2P%</td>
    </tr>
    <tr>
      <td>-0.030855</td>
      <td>Pos_PG</td>
    </tr>
    <tr>
      <td>-0.074099</td>
      <td>Pos_SF</td>
    </tr>
    <tr>
      <td>-0.083176</td>
      <td>STL</td>
    </tr>
    <tr>
      <td>-0.094785</td>
      <td>3PA</td>
    </tr>
    <tr>
      <td>-0.119970</td>
      <td>Pos_PF</td>
    </tr>
    <tr>
      <td>-0.131116</td>
      <td>DRB%</td>
    </tr>
    <tr>
      <td>-0.142185</td>
      <td>PER</td>
    </tr>
    <tr>
      <td>-0.155014</td>
      <td>MP</td>
    </tr>
    <tr>
      <td>-0.156838</td>
      <td>STL%</td>
    </tr>
    <tr>
      <td>-0.196345</td>
      <td>PF</td>
    </tr>
    <tr>
      <td>-0.284399</td>
      <td>G</td>
    </tr>
  </tbody>
</table>
</div>



For 2007-2013 the top 5 features are: 
- Win shares 
- Defensive Win shares
- Total Rebounds
- Games Started 
- Defensive Rebounds

and the bottom 5 features are: 
- Games
- Personal Fouls
- Steal %
- Minutes Played
- Player Efficiency Rating 


```python
coefficients = pd.concat([pd.DataFrame(X_smote06.columns),pd.DataFrame(np.transpose(myLog3.coef_))], axis = 1)
coefficients.columns=('Feature', 'Coefficients')
Coef00_06 = coefficients.groupby('Coefficients').sum().sort_values('Coefficients', ascending=False)
Coef00_06
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
      <th>Feature</th>
    </tr>
    <tr>
      <th>Coefficients</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0.445622</td>
      <td>DWS</td>
    </tr>
    <tr>
      <td>0.390750</td>
      <td>WS</td>
    </tr>
    <tr>
      <td>0.267192</td>
      <td>VORP</td>
    </tr>
    <tr>
      <td>0.228585</td>
      <td>BPM</td>
    </tr>
    <tr>
      <td>0.212528</td>
      <td>PTS</td>
    </tr>
    <tr>
      <td>0.210850</td>
      <td>2PA</td>
    </tr>
    <tr>
      <td>0.203956</td>
      <td>FGA</td>
    </tr>
    <tr>
      <td>0.203855</td>
      <td>TRB</td>
    </tr>
    <tr>
      <td>0.173782</td>
      <td>OBPM</td>
    </tr>
    <tr>
      <td>0.173006</td>
      <td>DRB</td>
    </tr>
    <tr>
      <td>0.142083</td>
      <td>BLK%</td>
    </tr>
    <tr>
      <td>0.126156</td>
      <td>TOV</td>
    </tr>
    <tr>
      <td>0.125747</td>
      <td>FG</td>
    </tr>
    <tr>
      <td>0.123623</td>
      <td>2P</td>
    </tr>
    <tr>
      <td>0.113611</td>
      <td>BLK</td>
    </tr>
    <tr>
      <td>0.100482</td>
      <td>GS</td>
    </tr>
    <tr>
      <td>0.094746</td>
      <td>AST</td>
    </tr>
    <tr>
      <td>0.083291</td>
      <td>TOV%</td>
    </tr>
    <tr>
      <td>0.062105</td>
      <td>DBPM</td>
    </tr>
    <tr>
      <td>0.059130</td>
      <td>Age</td>
    </tr>
    <tr>
      <td>0.058936</td>
      <td>FTA</td>
    </tr>
    <tr>
      <td>0.050735</td>
      <td>ORB%</td>
    </tr>
    <tr>
      <td>0.032488</td>
      <td>ORB</td>
    </tr>
    <tr>
      <td>0.011965</td>
      <td>TRB%</td>
    </tr>
    <tr>
      <td>0.007508</td>
      <td>FTr</td>
    </tr>
    <tr>
      <td>0.007239</td>
      <td>3PA</td>
    </tr>
    <tr>
      <td>0.006042</td>
      <td>3P</td>
    </tr>
    <tr>
      <td>-0.000157</td>
      <td>WS/48</td>
    </tr>
    <tr>
      <td>-0.006727</td>
      <td>3P%</td>
    </tr>
    <tr>
      <td>-0.007577</td>
      <td>Pos_SF</td>
    </tr>
    <tr>
      <td>-0.012901</td>
      <td>FG%</td>
    </tr>
    <tr>
      <td>-0.016066</td>
      <td>2P%</td>
    </tr>
    <tr>
      <td>-0.016598</td>
      <td>eFG%</td>
    </tr>
    <tr>
      <td>-0.020222</td>
      <td>TS%</td>
    </tr>
    <tr>
      <td>-0.023414</td>
      <td>3PAr</td>
    </tr>
    <tr>
      <td>-0.026249</td>
      <td>AST%</td>
    </tr>
    <tr>
      <td>-0.037161</td>
      <td>FT</td>
    </tr>
    <tr>
      <td>-0.040146</td>
      <td>Pos_PG</td>
    </tr>
    <tr>
      <td>-0.048597</td>
      <td>OWS</td>
    </tr>
    <tr>
      <td>-0.050165</td>
      <td>FT%</td>
    </tr>
    <tr>
      <td>-0.062571</td>
      <td>Pos_SG</td>
    </tr>
    <tr>
      <td>-0.100040</td>
      <td>STL</td>
    </tr>
    <tr>
      <td>-0.102321</td>
      <td>Pos_PF</td>
    </tr>
    <tr>
      <td>-0.118643</td>
      <td>DRB%</td>
    </tr>
    <tr>
      <td>-0.126606</td>
      <td>USG%</td>
    </tr>
    <tr>
      <td>-0.148116</td>
      <td>G</td>
    </tr>
    <tr>
      <td>-0.187055</td>
      <td>PER</td>
    </tr>
    <tr>
      <td>-0.208667</td>
      <td>STL%</td>
    </tr>
    <tr>
      <td>-0.217570</td>
      <td>PF</td>
    </tr>
    <tr>
      <td>-0.223534</td>
      <td>MP</td>
    </tr>
  </tbody>
</table>
</div>



For 2000-2006 the top 5 features are:
- Defensive Win Shares
- Win Shares
- Value Over Replacement Player
- Box Plus Minus
- Points

Bottom 10 features in terms of importance against making the AS game: 
- Minutes Played
- Personal Fouls
- Steal %
- Player Efficiency Rating
- Games

Interestingly, we can see that many of the top 5 features and bottom 5 features are consistent across the different eras. Although there are differences noticed over time, this shows that the best ways of evaluating players has not really changed much over the past 20 years. Also stats that are often quoted as being good metrics for evaluating players, Player Efficiency Rating for example, is consistently one of the worst predictors for determining if a player is elite.


```python
# Now let's look at how the models predict the test set:
y_pred20 = myLog.predict(X_test20)
y_pred13 = myLog2.predict(X_test13)
y_pred06 = myLog3.predict(X_test06)
```


```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
print(confusion_matrix(y_test20, y_pred20))
print()
print(confusion_matrix(y_test13, y_pred13))
print()
print(confusion_matrix(y_test06, y_pred06))
```

    [[592  48]
     [  1  44]]
    
    [[582  39]
     [  5  27]]
    
    [[530  46]
     [  7  40]]


Confusion Matrix for 2020:

|     |  Predicted Non-Allstar  |   Predicted All-star | True Totals |
| --------- |:---:|:---:|:---:|
|True Non-Allstar|594 | 46 | 640 |
|True All-star|1 |44 | 45 |
|Predicted Totals| 595 | 90 | 685 |

From the above, we can conclude that all three of the Logistic Models for the three eras seem to be a bit too prone to predicting all star. Although the true totals are 640 non allstars and 45 true all-stars (for the 2014-2020 model), the model predicted 45 more all-stars then there actually were in the dataset. This isn't actually that surprising, given that every year in the NBA there are many players that are seen as "all-star snubs" (players that media and fans agree should have made the team but don't). This lack of precision is partially due to restrictions on positions in an increasingly position-less NBA (certain number of each position allowed in the NBA but many players don't really fit into traditional position roles), meaning certain players 'deserved' to make the team but didn't. The lack of precision is also due to the nature of upsampling. Since we trained our data on a model where there was an equal number of both classes, the model was not very good at being precise and only picking the best of the best as all-stars. 

However despite the extremely low precision figures (as evidenced below), given that the main purpose of these models are to identify the statistics most important for predicting all star status (the feature coefficients of the model), the model serves its' purpose well. 


```python
print("2014-2020: ")
print("Accuracy score: ", accuracy_score(y_test20,y_pred20))
print("Recall score: ", recall_score(y_test20, y_pred20))
print("Precision Score: ", precision_score(y_test20, y_pred20))
print("2007-2013: ")
print("Accuracy score: ", accuracy_score(y_test13,y_pred13))
print("Recall score: ", recall_score(y_test13, y_pred13))
print("Precision Score: ", precision_score(y_test13, y_pred13))
print("2000-2006")
print("Accuracy score: ", accuracy_score(y_test06,y_pred06))
print("Recall score: ", recall_score(y_test06, y_pred06))
print("Precision Score: ", precision_score(y_test06, y_pred06))
```

    2014-2020: 
    Accuracy score:  0.9284671532846716
    Recall score:  0.9777777777777777
    Precision Score:  0.4782608695652174
    2007-2013: 
    Accuracy score:  0.9326186830015314
    Recall score:  0.84375
    Precision Score:  0.4090909090909091
    2000-2006
    Accuracy score:  0.9149277688603531
    Recall score:  0.851063829787234
    Precision Score:  0.46511627906976744


-- Other Metrics:


```python
y_proba20 = myLog.predict_proba(X_test20)[:,1]
from sklearn.metrics import roc_curve, roc_auc_score

fprs, tprs, thresholds = roc_curve(y_test20, y_proba20)
roc_auc = roc_auc_score(y_test20, y_proba20)

#Get the probability for each point in the train set.
y_proba_train20 = myLog.predict_proba(X_smote20)[:,1]

# Compute ROC curve and AUC for for the one class
fprs_train, tprs_train, thresholds_train = roc_curve(y_smote20, y_proba_train20)
roc_auc_train = roc_auc_score(y_smote20, y_proba_train20)
  
# Plot the ROC curve.
plt.figure()
plt.plot(fprs_train, tprs_train, color='darkorange', lw=2, label='train')
plt.plot(fprs, tprs, lw=2, label='test')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC for LogReg Model 2014-2020')
plt.legend(loc="best")
plt.show()
print(f"Test AUC score: {roc_auc}")
print(f"Train AUC score: {roc_auc_train}")
```


![png](output_64_0.png)


    Test AUC score: 0.9854166666666666
    Train AUC score: 0.9906979753212592



```python
# 2007-2013: 
y_proba13 = myLog2.predict_proba(X_test13)[:,1]
from sklearn.metrics import roc_curve, roc_auc_score

fprs, tprs, thresholds = roc_curve(y_test13, y_proba13)
roc_auc = roc_auc_score(y_test13, y_proba13)

#Get the probability for each point in the train set.
y_proba_train13 = myLog2.predict_proba(X_smote13)[:,1]

# Compute ROC curve and AUC for for the one class
fprs_train, tprs_train, thresholds_train = roc_curve(y_smote13, y_proba_train13)
roc_auc_train = roc_auc_score(y_smote13, y_proba_train13)
  
# Plot the ROC curve.
plt.figure()
plt.plot(fprs_train, tprs_train, color='darkorange', lw=2, label='train')
plt.plot(fprs, tprs, lw=2, label='test')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC for LogReg Model 2007-2013')
plt.legend(loc="best")
plt.show()
print(f"Test AUC score: {roc_auc}")
print(f"Train AUC score: {roc_auc_train}")
```


![png](output_65_0.png)


    Test AUC score: 0.9747886473429952
    Train AUC score: 0.9910078868396105



```python
# 2000-2006: 
y_proba06 = myLog2.predict_proba(X_test06)[:,1]
from sklearn.metrics import roc_curve, roc_auc_score

fprs, tprs, thresholds = roc_curve(y_test06, y_proba06)
roc_auc = roc_auc_score(y_test06, y_proba06)

#Get the probability for each point in the train set.
y_proba_train06 = myLog3.predict_proba(X_smote06)[:,1]

# Compute ROC curve and AUC for for the one class
fprs_train, tprs_train, thresholds_train = roc_curve(y_smote06, y_proba_train06)
roc_auc_train = roc_auc_score(y_smote06, y_proba_train06)
  
# Plot the ROC curve.
plt.figure()
plt.plot(fprs_train, tprs_train, color='darkorange', lw=2, label='train')
plt.plot(fprs, tprs, lw=2, label='test')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC for LogReg Model 2000-2006')
plt.legend(loc="best")
plt.show()
print(f"Test AUC score: {roc_auc}")
print(f"Train AUC score: {roc_auc_train}")
```


![png](output_66_0.png)


    Test AUC score: 0.9829343971631206
    Train AUC score: 0.9920090662118183


The three ROC curves above and the corresponding AUC scores certainly show evidence of overfitting, however the extremely low false positive rate (data points the model classified as all-stars who were in fact not all-stars) is an encouraging sign that the models are effective for their purpose.


```python
# Although the Logistic Model with minimal feature engineering performs very well. Let's see how some other models 
# perform on the data and if further feature engineering improves results:

from sklearn.decomposition import PCA
# Let's first instantiate the PCA modifier including all of the components so we can look at the explained 
# variance (in other words the amount of variance of the total dataset per each feature):
myPCA = PCA(n_components=(50))
myPCA.fit(X_smote20)
X_train20_PCA = myPCA.transform(X_smote20)

expl_var = myPCA.explained_variance_ratio_
plt.figure()
plt.plot(range(1,51),expl_var,marker='.')
plt.xlabel('Number of PCs')
plt.ylabel('Proportion of Variance Explained')
plt.show()
```


![png](output_68_0.png)



```python
expl_var_cumulative = myPCA.explained_variance_ratio_.cumsum()

plt.figure(figsize=(12,6))
plt.plot(range(1,51),expl_var_cumulative,marker='.')
plt.xlabel('Number of PCs')
plt.ylabel('Cumulative Sum of Proportion of Variance Explained')
plt.margins(x=0, y=0)
plt.xticks(range(0,50))
plt.hlines(0.95,xmin=0, xmax=50)
plt.vlines(5,ymin=0, ymax=1)
plt.show()
```


![png](output_69_0.png)


As we can see from the above curve, we have about 95% of the cumulative explained variance with just 5 features. Therefore, let's try to limit our PCA variable to just 5 features and see if it improves results:


```python
myPCA = PCA(n_components=5)
myPCA.fit(X_smote20)
X_remainder20_PCA = myPCA.transform(X_smote20)
X_test20_PCA = myPCA.transform(X_test20)
```


```python
# Let's pick a C-value for 10^-2, in order to be conservative, and also as this was the same C value on the Log model
# without PCA. 
myLog_PCA = LogisticRegression(C = 0.01, solver='lbfgs', n_jobs=-1)
myLog_PCA.fit(X_remainder20_PCA,y_smote20)
print(myLog_PCA.score(X_test20_PCA, y_test20))
print(myLog.score(X_test20,y_test20))
```

    0.9343065693430657
    0.9284671532846716


Virtually, the same exact score, which isn't particularly surprising given that we picked features that explained 90% of the variance. Ultimately, in this use, I would advise against using PCA as the amount of data isn't enough to require it (from a computing time perspective), and using PCA reduces the intrepretability of the data after the fact as it is now virtually impossible to gather insight about which features are driving this model. 


```python
# Now Let's do KNN and Decision Tree's and try out a few other ensemble methods as well.
from sklearn.neighbors import KNeighborsClassifier

# Let's loop over a number of different n_neighbors values to see which one computes best: 
n_neighbors_ = [3,5,9,11, 13, 15, 17, 19, 21, 23, 25]
validation_knn_scores20=[]
train_knn_scores20=[]
#create a for loop to iterate over the possible n_neighbors list above: 
for n in n_neighbors_:
    testKNN = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
    testKNN.fit(X_train20,y_train20)
    train_knn_scores20.append(testKNN.score(X_train20,y_train20))
    validation_knn_scores20.append(testKNN.score(X_validation20,y_validation20))
```


```python
plt.figure()
plt.plot(n_neighbors_, train_knn_scores20, label='Train accuracies', marker='.')
plt.plot(n_neighbors_, validation_knn_scores20, label="Validation accuracies",marker='.')
plt.legend()
plt.xlabel('N_neighbors')
plt.ylabel('Accuracy Score')
plt.title('N_neighbors scores in KNN Model')
plt.grid()
plt.show();
```


![png](output_75_0.png)



```python
# Let's try 11 n_neighbors, since too low number of neighbors seems a bit prone to overfitting, and any lower and 
# our validation accuracies will be higher then train:
myKNN = KNeighborsClassifier(n_neighbors= 11)
myKNN.fit(X_smote20,y_smote20)
print(myKNN.score(X_test20, y_test20))
```

    0.9343065693430657



```python
# Let's use the same n_neighbors value to see the results for the other two 'eras':

myKNN2 = KNeighborsClassifier(n_neighbors= 11)
myKNN2.fit(X_smote13,y_smote13)
print(myKNN2.score(X_test13, y_test13))

myKNN3 = KNeighborsClassifier(n_neighbors= 11)
myKNN3.fit(X_smote06,y_smote06)
print(myKNN3.score(X_test06, y_test06))
```

    0.9509954058192955
    0.9245585874799358


KNN seems to perform very well, and could be used as a substitute for Logistic Regression, however we are losing the ability to figure out which features are most impactful in determining the result of our model, given that KNN classifies unseen data based off their proximity to known data. 


```python
from sklearn.tree import DecisionTreeClassifier

depth_values = list(range(1, 20))
train_scores20 = []
validation_scores20 = []

# Loop over different max_depths
for d in depth_values:
    
    # Instantiate & fit
    my_dt = DecisionTreeClassifier(max_depth = d)
    my_dt.fit(X_train20, y_train20)
    
    # Evaluate on train & test data
    train_scores20.append( my_dt.score(X_train20, y_train20) )
    validation_scores20.append( my_dt.score(X_validation20, y_validation20) )
    

plt.figure()
plt.plot(depth_values, train_scores20, label='train')
plt.plot(depth_values, validation_scores20, label='validation')
plt.legend()
plt.xlabel('max depth')
plt.ylabel('accuracy')
plt.title('DecisionTree Scores')
plt.show()
```


![png](output_79_0.png)



```python
# With the above, anything above ~5 as a max depth is extremely prone to overfitting as the train scores reach ~1.0, 
# and validation scores continue to rise and fall. In addition, decision trees are very prone to overfitting even 
# at the best of times, and it seems that these models in general are prone to overfitting on this data set.

# For the above reasons, it makes most sense to pick a max depth value of lower then 5, therefore let's use 2.

myDT = DecisionTreeClassifier(max_depth = 2)
myDT.fit(X_smote20, y_smote20)
print("2014-2020: ", myDT.score(X_test20,y_test20))

# And let's use the same max_depth value for the other two eras as well:
myDT2 = DecisionTreeClassifier(max_depth = 2)
myDT2.fit(X_smote13, y_smote13)
print("2007-2013: ", myDT2.score(X_test13, y_test13))

myDT3 = DecisionTreeClassifier(max_depth = 2)
myDT3.fit(X_smote06, y_smote06)
print("2000-2006: ", myDT3.score(X_test06, y_test06))
```

    2014-2020:  0.9357664233576642
    2007-2013:  0.9540581929555896
    2000-2006:  0.9245585874799358



```python
# Can a Random Forest Improve on the Decision Tree above? 

from sklearn.ensemble import RandomForestClassifier

myrf = RandomForestClassifier(n_estimators=50)
myrf.fit(X_smote20, y_smote20)

decision_tree_scores = []
for sub_tree in myrf.estimators_:
    decision_tree_scores.append(sub_tree.score(X_test20, y_test20))

print("Performance on Test data:")
print(f"Best Decision Tree: {max(decision_tree_scores)}")
print(f"Worst Decision Tree: {min(decision_tree_scores)}")
print(f"Average Decision Tree: {np.mean(decision_tree_scores)}")
print(f"Random Forest: {myrf.score(X_test20, y_test20)}")
```

    Performance on Test data:
    Best Decision Tree: 0.948905109489051
    Worst Decision Tree: 0.2321167883211679
    Average Decision Tree: 0.9045839416058393
    Random Forest: 0.9343065693430657


Although this may seem like extremely good scores, Decision Trees are extremely prone to overfitting at the best of times, and since this basketball dataset makes it especially easy for overfitting to occur, Decision Trees and Random Forests are probably not the best classifier in this case (even KNN is superior in this case to Decision Trees and Random Forests as there is less evidence of overfitting). The inaccuracy of Random Forests (and thereby Decision Trees as well) is showed when looking at the importance of features:


```python
fi = pd.DataFrame(myrf.feature_importances_,
                  index = X_smote20.columns)
fi
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Age</td>
      <td>0.016136</td>
    </tr>
    <tr>
      <td>G</td>
      <td>0.004277</td>
    </tr>
    <tr>
      <td>GS</td>
      <td>0.012072</td>
    </tr>
    <tr>
      <td>MP</td>
      <td>0.051264</td>
    </tr>
    <tr>
      <td>FG</td>
      <td>0.130662</td>
    </tr>
    <tr>
      <td>FGA</td>
      <td>0.035948</td>
    </tr>
    <tr>
      <td>FG%</td>
      <td>0.004011</td>
    </tr>
    <tr>
      <td>3P</td>
      <td>0.005559</td>
    </tr>
    <tr>
      <td>3PA</td>
      <td>0.005826</td>
    </tr>
    <tr>
      <td>3P%</td>
      <td>0.004537</td>
    </tr>
    <tr>
      <td>2P</td>
      <td>0.003567</td>
    </tr>
    <tr>
      <td>2PA</td>
      <td>0.018948</td>
    </tr>
    <tr>
      <td>2P%</td>
      <td>0.003011</td>
    </tr>
    <tr>
      <td>eFG%</td>
      <td>0.002633</td>
    </tr>
    <tr>
      <td>FT</td>
      <td>0.069432</td>
    </tr>
    <tr>
      <td>FTA</td>
      <td>0.031216</td>
    </tr>
    <tr>
      <td>FT%</td>
      <td>0.004451</td>
    </tr>
    <tr>
      <td>ORB</td>
      <td>0.003785</td>
    </tr>
    <tr>
      <td>DRB</td>
      <td>0.003366</td>
    </tr>
    <tr>
      <td>TRB</td>
      <td>0.003784</td>
    </tr>
    <tr>
      <td>AST</td>
      <td>0.005878</td>
    </tr>
    <tr>
      <td>STL</td>
      <td>0.018015</td>
    </tr>
    <tr>
      <td>BLK</td>
      <td>0.005902</td>
    </tr>
    <tr>
      <td>TOV</td>
      <td>0.003688</td>
    </tr>
    <tr>
      <td>PF</td>
      <td>0.003657</td>
    </tr>
    <tr>
      <td>PTS</td>
      <td>0.098853</td>
    </tr>
    <tr>
      <td>PER</td>
      <td>0.025351</td>
    </tr>
    <tr>
      <td>TS%</td>
      <td>0.003132</td>
    </tr>
    <tr>
      <td>3PAr</td>
      <td>0.004218</td>
    </tr>
    <tr>
      <td>FTr</td>
      <td>0.002709</td>
    </tr>
    <tr>
      <td>ORB%</td>
      <td>0.004799</td>
    </tr>
    <tr>
      <td>DRB%</td>
      <td>0.002827</td>
    </tr>
    <tr>
      <td>TRB%</td>
      <td>0.002880</td>
    </tr>
    <tr>
      <td>AST%</td>
      <td>0.004282</td>
    </tr>
    <tr>
      <td>STL%</td>
      <td>0.003285</td>
    </tr>
    <tr>
      <td>BLK%</td>
      <td>0.004074</td>
    </tr>
    <tr>
      <td>TOV%</td>
      <td>0.003162</td>
    </tr>
    <tr>
      <td>USG%</td>
      <td>0.040394</td>
    </tr>
    <tr>
      <td>OWS</td>
      <td>0.017024</td>
    </tr>
    <tr>
      <td>DWS</td>
      <td>0.013724</td>
    </tr>
    <tr>
      <td>WS</td>
      <td>0.051034</td>
    </tr>
    <tr>
      <td>WS/48</td>
      <td>0.006387</td>
    </tr>
    <tr>
      <td>OBPM</td>
      <td>0.112703</td>
    </tr>
    <tr>
      <td>DBPM</td>
      <td>0.002996</td>
    </tr>
    <tr>
      <td>BPM</td>
      <td>0.026135</td>
    </tr>
    <tr>
      <td>VORP</td>
      <td>0.113733</td>
    </tr>
    <tr>
      <td>Pos_PF</td>
      <td>0.001710</td>
    </tr>
    <tr>
      <td>Pos_PG</td>
      <td>0.000535</td>
    </tr>
    <tr>
      <td>Pos_SF</td>
      <td>0.000963</td>
    </tr>
    <tr>
      <td>Pos_SG</td>
      <td>0.001461</td>
    </tr>
  </tbody>
</table>
</div>



As we can see some of the stats that were given the highest weight in determining the decision tree boundaries anectodally speaking convey much less value. Advanced Metrics that are designed to encompass multiple different areas of basketball are given tiny weightings, stats like FG and PTS are heavily weighted. 

With that being said, it is worth noting that Random Forest vastly outperformed the average decision tree. 


```python
# Now let's try an SVM (for this example I am assuming the same C value as was optimal with the LogReg models):
from sklearn.svm import LinearSVC

mysvm = LinearSVC(C=0.01, random_state=1)
mysvm.fit(X_smote20, y_smote20)
print("2014-2020: ", mysvm.score(X_test20, y_test20))

mysvm2 = LinearSVC(C=0.01, random_state=1)
mysvm2.fit(X_smote13, y_smote13)
print("2007-2013: ", mysvm.score(X_test13, y_test13))

mysvm3 = LinearSVC(C=0.01, random_state=1)
mysvm3.fit(X_smote06, y_smote06)
print("2000-2006: ", mysvm.score(X_test06, y_test06))
```

    2014-2020:  0.9124087591240876
    2007-2013:  0.9065849923430321
    2000-2006:  0.8683788121990369



```python
# Now let's try another ensemble method to see if we can increase performance...Voting:

from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(estimators=[('myKNN', myKNN),
                                        ('myLog', myLog),
                                        ('myDT', myDT)],
                                        voting="soft",
                                        weights = [1,1,1])


scores = cross_val_score(ensemble, X_smote20, y_smote20, cv=5)
scores_mean = round(scores.mean(),4)
print("cross_val: ",scores_mean)

ensemble.fit(X_smote20,y_smote20)
print("test: ",ensemble.score(X_test20,y_test20))
```

    cross_val:  0.9574
    test:  0.9357664233576642


Conclusions of Part 2: 
Final considerations:
- In this example, logistic regression made the most sense, given that it had the best (or close to the best) scores and has the extra intrepretability of being able to easily find the feature coefficients and intepret them. In addition, incorporating domain specific knowledge the results of the most important categories make the most sense with using a Logistic model.

- This classification problem in general seems relatively easy for a model to determine, which is an interesting finding. However despite the very strong accuracy figures, precision is extremely low, therefore the notion that there are many 'snubs' (players who deserve to be all-stars but don't make it) each year is supported by the models. 

- It is worth reiterating again all the logistic models had high recall, which means these models are casting too wide of a net, in other words the models classified the player as an all-star more then the true number of all-stars in the data set. 

- It is worth summarazing again the subjective assumptions/decisions made in the first part of this document including: 
    - the threshold cutoffs (at least 20 games played and 8 minutes played), 
    - the decision to drop the team column entirely after removing the individual team rows for players who were traded in the season.
    - the decision to classify all players as their primary or first-listed position only.

Other decisions could be made in place of these decisions, which could affect the accuracy results greatly. 

- Due to most of the models above showing signs of overfitting, decisions were often made to be more conservative to mitigate the affect of this problem (ex: smaller C values for logistic regression, etc.).

- Many of the stats found to be most impactful in the present era were also found to be the most impactful in previous eras as well. This is a surprising finding; although I understood that stats like WS/DWS were very important I was not expecting them to be #1/#2 respectively for each year. DWS makes some sense as a very impactful statistic given that very few bball stats account for defense (blocks and steals are two of the most quoted) and defense accounts for half of all basketball, however the consistency of it being in the top 2 for each era was a very interesting finding nontheless. 
