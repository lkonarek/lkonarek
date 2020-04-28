---
layout: post
title: NBA & Data Science
---

Introduction: 

As someone who is practically a life-long NBA fan, I have spent hours (perhaps even days) of my life staring at the stats of my favorite players. Obsessively memorizing stats so I could beat my friends in arguments about which player was better was my primary goal I must admit, but what started at what stats are the single best to follow, became so I could understand what all the stats meant. 

Basketball may be the single sport with the most stats (other then perhaps baseball), and anyone who has ever gone to a specific players profile knows that they keep track of an amazing amount of things, everything from scored points to tipped passes. And it is understandable why this is the case, NBA teams pay tens of millions of dollars to sign players on multi-year contracts. It is vital to understand not only on which metrics to evaluate players, but also how much to pay. The tendency is to evaluate players against their peers who play the same position, so it is certainly possible that certain player types or positions are overpaid relative to others.
Due to the clear business case for the NBA, I sought to answer two main questions:
- What are the best ways of evaluating a basketball player’s on court value?
- Could this on court value be used to determine if a player is over or under paid?
Currently NBA teams have comprehensive analytics departments, but most of their analytics work is focused on the teams’ style of play as a supplemental to coaching, for example: taking more 3 pointers, driving to the net more in an attempt to draw fouls, and kick out passes to players in the corners to stretch the defense. However, NBA teams could do more to apply analytics to value for money.

Data Acquisition:

Given that basketball stats are readily available online, I manually scraped all the stats from basketball-reference.com (using beautifulsoup w/ Python). My reasoning for doing this rather then downloading a dataset from Kaggle or another online depository was due to the fact that I wanted all the clean-up and dataframe schemas to be the same for each individual season I processed. 

In a perfect world, it would be ideal to train models on the data for each individual season and evaluate against each other, but given that there are only about 300-400 players who play significant minutes per season (at least 8 minutes per game and 20 games played), there are too few rows for an effective model to be trained on. Therefore, multiple years were combined together into ‘eras’ for modelling: 2014-2020 (present era), 2007-2013, and 2000-2006.

EDA & Summary of Cleaning:
While initially exploring the data, the preliminary goal was to isolate what distinguishes a good player from an elite player. All-star status was determined as the best way to do this. Once the most vital stats were isolated, on-court value vs contracts could be compared.
The dataset required little traditional cleaning. Each row in the dataset corresponded to a player, therefore null values were not a significant issue, as the only nulls were for players who didn’t record any of a percent-based stat (3p%, FT%, etc.). Null values represented less then 15% in any column, and most columns had 0% null values.
However, a number of non-conventional preprocessing decisions were made. First as already alluded to, players that do not play significant minutes and enough games were excluded in order to not skew the data and perhaps make the model more prone to overfitting. Also, by adding this threshold the vast majority of the nulls of the dataset were removed.
Second, another problem were players that were traded within a given season, as these players would have multiple rows (their stat totals for the year, their stats for the first team they were on, and their stats for the second team they were on). To avoid this triple counting, the best solution was to keep only the ‘total’ row for each player and drop the team column altogether.
Additional Feature Engineering and EDA work was primarily driven by domain expertise. Certain statistics were dropped as features from the models in order to mitigate issues with collinearity or because certain features were anecdotally unimportant for predicting the y-variable (all-star status). For example, the age column was dropped given that the average age of the NBA is very young, and rookies and sophomores typically don’t make the all-star game even if their stats warrant it. 

Now that I have explained the primary EDA considerations and preprocessing that was required to get the datasets ready for modelling, I first had to determine how best to deal with the first goal of the project: to identify what causes a player to successfully make the all-star team. However to accurately build a model to predict this, we would need to have a similar number of data points. 

```python
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

As we can see the classification problem is very biased (0 means non all-star and 1 means all-star), which makes sense given the number of players that make the all-star team vs the number of players who do not.

Now it is important that we define the threshold for a player essentially making into the dataset. Given that some players get called up from the NBA development league (the G-league) throughout the season as team need requires. Therefore roughly a quarter of the season (20 games of the total 82 games) and 8 minutes per game (1/6th of a total individual game). As you can also see, the same threshold was chosen for each season: 
```python

df14_20 = df14_20[df14_20['G']>20]
df14_20 = df14_20[df14_20['MP']>8.0]

df07_13 = df07_13[df07_13['G']>20]
df07_13 = df07_13[df07_13['MP']>8.0]

df00_06 = df00_06[df00_06['G']>20]
df00_06 = df00_06[df00_06['MP']>8.0]
```

At this point, now that we have actually created the above thresholds, the only columns that have any null values is 3% and it is only for the players who have recorded no 3 pointers (fairly common for players who are centres and don't shoot long range shots at all):

```python
# Given that 3p% being null is due to centers in the nba who don't shoot them at all, and therefore record a null 3p% 
# stat, we will fill them in with 0's:
df14_20['3P%'] = df14_20['3P%'].fillna(0)
df07_13['3P%'] = df07_13['3P%'].fillna(0)
df00_06['3P%'] = df00_06['3P%'].fillna(0)
```

The Last cleaning decision that needs to be made is regarding positions, at this point the data is saved as such 
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

Modelling:

```python
# Now let's state the X and y variables clearly in order for modelling: 
y20 = df14_20['All-star']
X20 = df14_20.drop(['All-star'], axis=1)

y13 = df07_13['All-star']
X13 = df07_13.drop(['All-star'], axis=1)

X06 = df00_06.drop(['All-star'], axis=1)
y06 = df00_06['All-star']
```

First let's split up our dataset of the stats from 2014-2020 into a remainder and test set. The test set is set aside so there is no data leakage.
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

Next, we need to solve the problem mentioned earlier: there are far more players who do not qualify as an all-star versus players who do. If we do nothing and fit a model to these data points, any model would no-doubt be dramatically overfit, as just having a model assume 100% players are not all-stars would still perform well under most metrics. Therefore resampling needs to be done, more specifically a form of upsampling, given that downsimpling would result in two few datapoints to train on). SMOTE was chosen as it fit the data better. 

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
Now this shows how many data points we have to train and optimize our models on. It is worth repeating that the test set has remained unaltered by resampling methods, as the test set needs to remain unseen data. 


Then a second split needs to be made where we will seperate the remainder set (after smote) into the train and validation sets which will actually be used to tune the primary models. 
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

Scale the data is important, however in order to pull out the coefficients of the best performing model X_train20 needs to remain as a dataframe with the column names present (normally scalers transform dataframes into a np array)
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

![V1](V1.png)

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

The above lists out all the table variables after all preprocessing ordered by their correlation coefficients in the model, or in other words, how highly correlated a given variable is in predicting the outcome of the y-variable according to the model. The top 5 are as such:
1. Win Shares
2. Defensive Winshares
3. Assists 
4. Defensive Rebounds
5. Value Over Replacement Player

And the 5 features most negatively correlated to making the all star team are: 
1. PER
2. Minutes Played
3. ORB%
4. Personal Fouls
5. Position - Small Forward

These are certainly interesting results, especially PER being the least correlated with all-star status, as this is an oft quoted stat. However it is also very prone to high output per minute statistics, so a lot of players who don't play many minutes per game (perhaps because they are horrific defenders) but score quite a few points while they are on the court, often have very high PER stats (Boban Marjanovic is an example).

For the other two eras now:

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

Interestingly, we can see that many of the top 5 features and bottom 5 features are quite consistent across the different eras. Although there are differences noticed over time, this shows that the best ways of evaluating players has not really changed much over the past 20 years. Also stats that are often quoted as being good metrics for evaluating players, Player Efficiency Rating for example, is consistently one of the worst predictors for determining if a player is elite.

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


Unsupervised Modelling:

```python
# Read in the data 
df = pd.read_csv('data/2019-2020.csv')
# In order to determine the right cut off for players, lets look at the middle point of minutes played: 
print(df['MP'].median())
print(df['MP'].mean())
```

    19.0
    19.576268412438615

This time let's use more then 19 mpg as the cut-off (therefore we are including the half of the players in the data set who play more):
```python

df = df[df['G']>20]

df = df[df['MP']>19.0]
```

So far, we have identified a number of different features that were consistently highly involved in success in all of the eras (but especially the recent era):
- Defensive Winshares
- Winshares
- Assists
- Defensive Rebounds
- VORP
- Total Rebounds
- Box Plus Minus
- Points
- Field Goals Attempted
- Offensive Box Plus Minus
- Block %
- 2 Points Attempted
- Turnovers
- Free Throws Attempted
- Block 
- Usage Rate
- Steals

Let's also include other important stats for basketball assessment (the reason for including these is that they are three of the most often quoted advanced stats):
- Free Throw Rate 
- PER 
- TS%
- 3PAr


```python
df2 = df[['DWS', 'WS', 'AST', 'DRB', 'VORP', 'TRB', 'BPM', 'PTS', 'FGA', 'OBPM', 'BLK%', '2PA', 'TOV', 'FTA', 'BLK', 
         'STL', 'USG%', 'FTr', 'PER', 'TS%', '3PAr', '2019-20']]
```

As we can see from a df.describe, the lowest negative number in any column is in BPM with -6.7, therefore let's add +7 to each column before doing a log transform:

```python
df2 = np.log(df2+7)
```

```python
k_range = np.arange(1,20)

inertia_list = []

for k in k_range :
    
    #Specify the model
    k_means_model = KMeans(n_clusters = k)
    k_means_model.fit(df2)
    
    inertia_list.append(k_means_model.inertia_)

plt.plot(k_range,inertia_list,marker = '.')
plt.show()
```


![png](output_24_0.png)



```python
# Let's zoom in closer to the first 9 points on the graph:

plt.figure()
plt.scatter(np.arange(12),inertia_list[0:12])
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show();
```


![png](output_25_0.png)


With the above, we can see that after ~8 clusters our inertia is relatively unchanged, meaning that adding another cluster after that point doesn't make much sense. Therefore picking a value from 4-8 clusters is probably best, as picking a value too low will likely result in too general/broad clusters to generate any interesting analysis. 
For the purposes of our exercise let's use 5 clusters to investigate, as this is the same number of positions in basketball. 


```python
k_means_model = KMeans(n_clusters = 5)
k_means_model.fit(df2)
k_means_model.labels_
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 2, 2, 2, 3, 4, 3, 3, 3, 3, 4, 2, 1, 1, 4, 1, 4, 3, 4, 4,
           4, 0, 4, 4, 3, 3, 4, 2, 3, 1, 1, 1, 4, 4, 4, 3, 4, 1, 4, 4, 1, 2,
           4, 1, 1, 3, 2, 2, 2, 4, 2, 2, 2, 1, 1, 1, 2, 1, 3, 3, 3, 3, 4, 3,
           0, 0, 3, 4, 1, 0, 0, 4, 2, 4, 3, 4, 1, 0, 1, 1, 3, 4, 3, 4, 0, 0,
           1, 4, 1, 4, 3, 3, 1, 3, 4, 3, 3, 2, 4, 3, 1, 3, 1, 4, 1, 3, 3, 4,
           0, 3, 3, 2, 3, 1, 3, 2, 2, 4, 4, 2, 3, 2, 1, 4, 3, 3, 2, 4, 4, 1,
           3, 3, 4, 1, 0, 4, 4, 3, 1, 2, 1, 3, 2, 1, 3, 4, 1, 3, 3, 4, 4, 4,
           3, 1, 3, 4, 1, 1, 1, 4, 4, 4, 4, 3, 1, 1, 3, 1, 4, 4, 4, 3, 4, 4,
           1, 1, 4, 1, 4, 4, 2, 2, 1, 0, 4, 3, 1, 4, 3, 2, 4, 1, 1, 1, 1, 1,
           1, 3, 1, 1, 1, 3, 4, 1, 4, 3, 1, 1, 4, 3, 2, 4, 2, 3, 0, 3, 2, 2,
           1, 0, 3, 4, 1, 4, 4, 3, 2, 1, 1, 4, 1, 3, 3, 3, 3, 2, 3, 3, 1],
          dtype=int32)




```python
df2_km = df2.copy()
df2_km['kmeans_sol'] = k_means_model.labels_
```


```python
# snapshot of the 5 clusters
df2_km.groupby('kmeans_sol').mean().transpose()
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
      <th>kmeans_sol</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DWS</td>
      <td>1.089187</td>
      <td>-0.118166</td>
      <td>0.548687</td>
      <td>-0.024811</td>
      <td>-0.738692</td>
    </tr>
    <tr>
      <td>WS</td>
      <td>1.381156</td>
      <td>0.012983</td>
      <td>0.668815</td>
      <td>-0.029712</td>
      <td>-1.075777</td>
    </tr>
    <tr>
      <td>AST</td>
      <td>1.328651</td>
      <td>0.565980</td>
      <td>-0.791423</td>
      <td>-0.565461</td>
      <td>-0.327528</td>
    </tr>
    <tr>
      <td>DRB</td>
      <td>1.107977</td>
      <td>-0.133984</td>
      <td>0.968799</td>
      <td>-0.381751</td>
      <td>-0.600395</td>
    </tr>
    <tr>
      <td>VORP</td>
      <td>1.657141</td>
      <td>0.111497</td>
      <td>0.307565</td>
      <td>-0.141394</td>
      <td>-1.034738</td>
    </tr>
    <tr>
      <td>TRB</td>
      <td>0.968013</td>
      <td>-0.166451</td>
      <td>1.236890</td>
      <td>-0.392984</td>
      <td>-0.614371</td>
    </tr>
    <tr>
      <td>BPM</td>
      <td>1.084868</td>
      <td>0.240649</td>
      <td>0.461379</td>
      <td>0.068295</td>
      <td>-1.114280</td>
    </tr>
    <tr>
      <td>PTS</td>
      <td>1.543781</td>
      <td>0.554981</td>
      <td>-0.146162</td>
      <td>-0.716270</td>
      <td>-0.609911</td>
    </tr>
    <tr>
      <td>FGA</td>
      <td>1.434590</td>
      <td>0.629381</td>
      <td>-0.436355</td>
      <td>-0.772448</td>
      <td>-0.421867</td>
    </tr>
    <tr>
      <td>OBPM</td>
      <td>1.246605</td>
      <td>0.438353</td>
      <td>0.287421</td>
      <td>-0.204380</td>
      <td>-1.040401</td>
    </tr>
    <tr>
      <td>BLK%</td>
      <td>0.051951</td>
      <td>-0.469193</td>
      <td>1.927399</td>
      <td>-0.105306</td>
      <td>-0.443236</td>
    </tr>
    <tr>
      <td>2PA</td>
      <td>1.442530</td>
      <td>0.480557</td>
      <td>0.332126</td>
      <td>-1.048913</td>
      <td>-0.402953</td>
    </tr>
    <tr>
      <td>TOV</td>
      <td>1.646895</td>
      <td>0.467701</td>
      <td>-0.348762</td>
      <td>-0.829141</td>
      <td>-0.378483</td>
    </tr>
    <tr>
      <td>FTA</td>
      <td>1.743716</td>
      <td>0.302146</td>
      <td>0.126064</td>
      <td>-0.802153</td>
      <td>-0.538401</td>
    </tr>
    <tr>
      <td>BLK</td>
      <td>0.305964</td>
      <td>-0.372197</td>
      <td>1.801321</td>
      <td>-0.221262</td>
      <td>-0.500407</td>
    </tr>
    <tr>
      <td>STL</td>
      <td>0.976409</td>
      <td>0.226101</td>
      <td>-0.575175</td>
      <td>-0.163742</td>
      <td>-0.308733</td>
    </tr>
    <tr>
      <td>USG%</td>
      <td>1.355556</td>
      <td>0.580718</td>
      <td>-0.321595</td>
      <td>-0.889026</td>
      <td>-0.278356</td>
    </tr>
    <tr>
      <td>FTr</td>
      <td>0.927128</td>
      <td>-0.052748</td>
      <td>0.913026</td>
      <td>-0.552527</td>
      <td>-0.385096</td>
    </tr>
    <tr>
      <td>PER</td>
      <td>1.391554</td>
      <td>0.279338</td>
      <td>0.894467</td>
      <td>-0.530246</td>
      <td>-0.961016</td>
    </tr>
    <tr>
      <td>TS%</td>
      <td>0.331474</td>
      <td>-0.193822</td>
      <td>1.116978</td>
      <td>0.409780</td>
      <td>-0.945280</td>
    </tr>
    <tr>
      <td>3PAr</td>
      <td>-0.488314</td>
      <td>-0.003714</td>
      <td>-1.432852</td>
      <td>0.844415</td>
      <td>0.172230</td>
    </tr>
    <tr>
      <td>2019-20</td>
      <td>0.859484</td>
      <td>0.481034</td>
      <td>0.005075</td>
      <td>-0.254501</td>
      <td>-0.679171</td>
    </tr>
  </tbody>
</table>
</div>




```python
from scipy.cluster.hierarchy import dendrogram, linkage

linkagemat = linkage(df2, 'ward')

plt.figure(figsize=(25, 10))
dendrogram(
    linkagemat,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=0.  # font size for the x axis labels
);
```


![png](output_30_0.png)



```python
# Dendrograms don't necessarily have the same interpretability in determining the ideal n_clusters like KMeans does, 
# however for our purposes let's use the same value (n_clusters=5):
from sklearn.cluster import AgglomerativeClustering
hclust_model = AgglomerativeClustering(n_clusters=5, linkage='ward')
hclust_model.fit(df2)
hclust_model.labels_
```




    array([3, 3, 3, 3, 3, 1, 3, 1, 1, 1, 3, 2, 1, 1, 1, 3, 1, 3, 1, 3, 3, 1,
           3, 1, 3, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 4, 1, 4, 0, 4, 4,
           4, 1, 0, 4, 0, 0, 2, 2, 0, 1, 0, 1, 4, 4, 4, 0, 0, 1, 0, 0, 1, 2,
           4, 0, 0, 0, 2, 2, 2, 4, 2, 2, 2, 0, 0, 0, 2, 1, 0, 0, 0, 0, 4, 0,
           1, 1, 0, 4, 1, 3, 3, 0, 2, 4, 0, 0, 0, 1, 0, 1, 0, 4, 0, 0, 1, 1,
           1, 4, 1, 4, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
           1, 0, 0, 2, 0, 1, 0, 2, 2, 0, 4, 1, 0, 2, 0, 0, 0, 2, 2, 4, 0, 0,
           0, 0, 0, 0, 1, 4, 0, 0, 1, 2, 1, 0, 2, 0, 0, 4, 1, 0, 0, 0, 0, 0,
           0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0,
           0, 0, 0, 0, 4, 0, 2, 2, 0, 1, 4, 0, 0, 0, 0, 2, 0, 1, 0, 1, 1, 1,
           1, 0, 0, 1, 1, 0, 4, 1, 0, 0, 0, 0, 0, 0, 2, 4, 1, 0, 3, 0, 2, 2,
           1, 1, 2, 0, 0, 0, 0, 2, 2, 1, 1, 4, 1, 0, 0, 0, 0, 2, 0, 0, 1])




```python
# In order to see how similar the two models are: 
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(k_means_model.labels_,hclust_model.labels_)
```




    0.3423373225160187



Not great Adjusted Rand index score, however this can be expected given how hclust and kmeans are calculated in very different ways. Simply put, kmeans works by attempting to find possible centre points for each cluster, and then puts each data point into whichever cluster has the nearest centre point. The hcluster model which uses agglomerative clustering looks at the variance of the data points and makes clusters based of miminising the variance within clusters. 


```python
df2_hc = df2.copy()
df2_hc['hclust_sol'] = hclust_model.labels_
```


```python
df2_km['ind'] = np.ones(len(df2_km.kmeans_sol))
df2_hc['ind'] = np.zeros(len(df2_km.kmeans_sol))
```


```python
# Stick the two graphs together to visualize: 
full_df2 = pd.concat([df2_km,df2_hc])
# done in this way because seaborn requires data to be this way: 
full_df2.hclust_sol[:len(df2_km.kmeans_sol)] = df2_km.kmeans_sol
```


```python
full_df2
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
      <th>2019-20</th>
      <th>2PA</th>
      <th>3PAr</th>
      <th>AST</th>
      <th>BLK</th>
      <th>BLK%</th>
      <th>BPM</th>
      <th>DRB</th>
      <th>DWS</th>
      <th>FGA</th>
      <th>FTA</th>
      <th>FTr</th>
      <th>OBPM</th>
      <th>PER</th>
      <th>PTS</th>
      <th>STL</th>
      <th>TOV</th>
      <th>TRB</th>
      <th>TS%</th>
      <th>USG%</th>
      <th>VORP</th>
      <th>WS</th>
      <th>hclust_sol</th>
      <th>ind</th>
      <th>kmeans_sol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.641947</td>
      <td>1.088899</td>
      <td>0.869614</td>
      <td>2.179381</td>
      <td>0.821316</td>
      <td>0.271810</td>
      <td>1.907255</td>
      <td>0.805012</td>
      <td>1.437003</td>
      <td>2.345187</td>
      <td>3.817334</td>
      <td>2.217939</td>
      <td>2.275568</td>
      <td>2.359121</td>
      <td>2.747742</td>
      <td>2.180719</td>
      <td>3.166709</td>
      <td>0.655087</td>
      <td>1.032005</td>
      <td>2.302224</td>
      <td>3.763570</td>
      <td>3.012560</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.098279</td>
      <td>1.409578</td>
      <td>0.287032</td>
      <td>2.663907</td>
      <td>-0.733690</td>
      <td>-0.922195</td>
      <td>1.824479</td>
      <td>2.034920</td>
      <td>1.011760</td>
      <td>2.036797</td>
      <td>2.935367</td>
      <td>1.579258</td>
      <td>2.155858</td>
      <td>2.261708</td>
      <td>2.205857</td>
      <td>0.605785</td>
      <td>2.877937</td>
      <td>1.650563</td>
      <td>0.368975</td>
      <td>2.366793</td>
      <td>2.853644</td>
      <td>2.019973</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.632591</td>
      <td>1.787952</td>
      <td>-0.330848</td>
      <td>3.304307</td>
      <td>-0.049520</td>
      <td>-0.258448</td>
      <td>1.866263</td>
      <td>1.562907</td>
      <td>2.140633</td>
      <td>1.881597</td>
      <td>1.585902</td>
      <td>0.315873</td>
      <td>2.052198</td>
      <td>2.016704</td>
      <td>1.884759</td>
      <td>0.876214</td>
      <td>2.681092</td>
      <td>1.194245</td>
      <td>0.327442</td>
      <td>1.751035</td>
      <td>3.435862</td>
      <td>2.453350</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.502731</td>
      <td>1.964731</td>
      <td>-0.526641</td>
      <td>1.157317</td>
      <td>0.172467</td>
      <td>0.049792</td>
      <td>1.907255</td>
      <td>1.289441</td>
      <td>1.744581</td>
      <td>1.928761</td>
      <td>2.099520</td>
      <td>0.756766</td>
      <td>1.988184</td>
      <td>2.119083</td>
      <td>2.016599</td>
      <td>2.432604</td>
      <td>1.307114</td>
      <td>0.985395</td>
      <td>0.389737</td>
      <td>1.942034</td>
      <td>2.792708</td>
      <td>1.855387</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.319202</td>
      <td>2.021994</td>
      <td>-1.017821</td>
      <td>0.191033</td>
      <td>3.734960</td>
      <td>2.779987</td>
      <td>1.838497</td>
      <td>1.651445</td>
      <td>2.798411</td>
      <td>1.637499</td>
      <td>2.645433</td>
      <td>1.729150</td>
      <td>1.833108</td>
      <td>2.331487</td>
      <td>1.994953</td>
      <td>1.668145</td>
      <td>1.079510</td>
      <td>1.681642</td>
      <td>0.990647</td>
      <td>1.513664</td>
      <td>3.033377</td>
      <td>2.684766</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>597</td>
      <td>-1.915297</td>
      <td>-1.393099</td>
      <td>0.879847</td>
      <td>-1.115652</td>
      <td>-0.049520</td>
      <td>0.625891</td>
      <td>-0.290362</td>
      <td>-0.927727</td>
      <td>-1.026690</td>
      <td>-1.431055</td>
      <td>-0.902735</td>
      <td>-0.145317</td>
      <td>-0.680534</td>
      <td>-0.727594</td>
      <td>-1.297768</td>
      <td>-0.798493</td>
      <td>-1.358240</td>
      <td>-0.986821</td>
      <td>0.555734</td>
      <td>-1.111634</td>
      <td>-0.593012</td>
      <td>-0.788673</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>605</td>
      <td>-1.392308</td>
      <td>-0.059673</td>
      <td>-0.585042</td>
      <td>-1.115652</td>
      <td>0.821316</td>
      <td>1.218658</td>
      <td>0.910911</td>
      <td>0.441872</td>
      <td>0.103460</td>
      <td>-0.471836</td>
      <td>0.716441</td>
      <td>1.862084</td>
      <td>1.256225</td>
      <td>1.584242</td>
      <td>0.104494</td>
      <td>-1.090414</td>
      <td>-0.264870</td>
      <td>0.617025</td>
      <td>1.918578</td>
      <td>0.565699</td>
      <td>0.750431</td>
      <td>0.937387</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>606</td>
      <td>0.301147</td>
      <td>-0.672089</td>
      <td>-0.420684</td>
      <td>0.355118</td>
      <td>-0.502516</td>
      <td>-0.258448</td>
      <td>0.755357</td>
      <td>-0.477967</td>
      <td>0.453716</td>
      <td>-1.161815</td>
      <td>-0.765650</td>
      <td>-0.162464</td>
      <td>0.237147</td>
      <td>0.277041</td>
      <td>-1.140965</td>
      <td>0.876214</td>
      <td>-0.797882</td>
      <td>-0.347752</td>
      <td>-0.067661</td>
      <td>-0.935812</td>
      <td>0.668360</td>
      <td>0.603051</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>607</td>
      <td>0.600195</td>
      <td>-0.059673</td>
      <td>-0.094049</td>
      <td>-0.581360</td>
      <td>-0.274488</td>
      <td>-0.180017</td>
      <td>-0.253340</td>
      <td>-0.175588</td>
      <td>0.792229</td>
      <td>-0.152281</td>
      <td>-0.972536</td>
      <td>-1.165194</td>
      <td>-0.574613</td>
      <td>-0.353182</td>
      <td>-0.444279</td>
      <td>1.407317</td>
      <td>-0.007809</td>
      <td>0.051856</td>
      <td>-0.944582</td>
      <td>0.017676</td>
      <td>-0.398504</td>
      <td>-0.563909</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>608</td>
      <td>0.711592</td>
      <td>0.273793</td>
      <td>-1.222535</td>
      <td>-0.775801</td>
      <td>-0.274488</td>
      <td>0.049792</td>
      <td>-0.077168</td>
      <td>0.279482</td>
      <td>-0.259382</td>
      <td>-0.444259</td>
      <td>0.282804</td>
      <td>1.009709</td>
      <td>0.157347</td>
      <td>0.816203</td>
      <td>-0.278910</td>
      <td>-0.510388</td>
      <td>-0.395705</td>
      <td>0.913827</td>
      <td>0.202780</td>
      <td>0.207535</td>
      <td>-0.303213</td>
      <td>0.290485</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>526 rows × 25 columns</p>
</div>




```python
# plotting side by side histograms for all features 
# left is hclust (ind = 0) and right is kmeans (ind = 1)
for col in full_df2.columns[:-3]:
    plt.figure(figsize=(20,30));
    g = sns.FacetGrid(full_df2, col='ind',hue='hclust_sol', margin_titles=True,height=8);
    g.map(sns.distplot, col);
    plt.show();
```


    <Figure size 1440x2160 with 0 Axes>



![png](output_38_1.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_3.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_5.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_7.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_9.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_11.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_13.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_15.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_17.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_19.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_21.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_23.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_25.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_27.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_29.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_31.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_33.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_35.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_37.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_39.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_41.png)



    <Figure size 1440x2160 with 0 Axes>



![png](output_38_43.png)


So which model should we 'trust', Hclust or KMeans? Hard to say definitively. However for the contract value plots above, it seems that both methods generation roughly the same clusters (this is encouraging). 

Ultimately this is where some other logic and/or domain knowledge can be applied. Intuitively speaking given that KMeans classifies points and their clusters on the distance between similar data points (more specifically the distance to the closest centre of the cluster), it is fair to say that this may be a better clustering tool for fitting the data then HClust.


```python
# Unscale the data so we can see the average of the clusters better: 
df2_km_u = df2_km.copy()
df2_km_u.iloc[:,:-2] = scaler2.inverse_transform(df2_km_u.iloc[:,:-2])
df2_km_u.iloc[:,:-2] = np.exp(df2_km_u.iloc[:,:-2])-7
```


```python
# The statistical averages of each cluster: 
df2_km_u.groupby('kmeans_sol').mean().transpose()
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
      <th>kmeans_sol</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DWS</td>
      <td>2.424324</td>
      <td>1.340323</td>
      <td>1.921212</td>
      <td>1.418750</td>
      <td>0.832836</td>
    </tr>
    <tr>
      <td>WS</td>
      <td>6.416216</td>
      <td>3.088710</td>
      <td>4.590909</td>
      <td>2.985937</td>
      <td>1.064179</td>
    </tr>
    <tr>
      <td>AST</td>
      <td>5.527027</td>
      <td>3.951613</td>
      <td>1.487879</td>
      <td>1.860938</td>
      <td>2.270149</td>
    </tr>
    <tr>
      <td>DRB</td>
      <td>6.108108</td>
      <td>3.638710</td>
      <td>5.754545</td>
      <td>3.228125</td>
      <td>2.847761</td>
    </tr>
    <tr>
      <td>VORP</td>
      <td>2.972973</td>
      <td>0.967742</td>
      <td>1.203030</td>
      <td>0.685938</td>
      <td>-0.217910</td>
    </tr>
    <tr>
      <td>TRB</td>
      <td>7.586486</td>
      <td>4.491935</td>
      <td>8.221212</td>
      <td>4.004687</td>
      <td>3.500000</td>
    </tr>
    <tr>
      <td>BPM</td>
      <td>4.270270</td>
      <td>0.506452</td>
      <td>1.378788</td>
      <td>-0.062500</td>
      <td>-2.629851</td>
    </tr>
    <tr>
      <td>PTS</td>
      <td>23.102703</td>
      <td>16.012903</td>
      <td>12.157576</td>
      <td>9.228125</td>
      <td>9.811940</td>
    </tr>
    <tr>
      <td>FGA</td>
      <td>17.110811</td>
      <td>12.945161</td>
      <td>8.651515</td>
      <td>7.257813</td>
      <td>8.592537</td>
    </tr>
    <tr>
      <td>OBPM</td>
      <td>3.694595</td>
      <td>1.048387</td>
      <td>0.690909</td>
      <td>-0.532813</td>
      <td>-2.007463</td>
    </tr>
    <tr>
      <td>BLK%</td>
      <td>1.786486</td>
      <td>1.056452</td>
      <td>4.684848</td>
      <td>1.548438</td>
      <td>1.091045</td>
    </tr>
    <tr>
      <td>2PA</td>
      <td>11.794595</td>
      <td>7.879032</td>
      <td>7.430303</td>
      <td>3.184375</td>
      <td>4.997015</td>
    </tr>
    <tr>
      <td>TOV</td>
      <td>3.035135</td>
      <td>2.000000</td>
      <td>1.345455</td>
      <td>0.982813</td>
      <td>1.322388</td>
    </tr>
    <tr>
      <td>FTA</td>
      <td>6.202703</td>
      <td>3.183871</td>
      <td>2.875758</td>
      <td>1.364063</td>
      <td>1.780597</td>
    </tr>
    <tr>
      <td>BLK</td>
      <td>0.675676</td>
      <td>0.359677</td>
      <td>1.393939</td>
      <td>0.428125</td>
      <td>0.302985</td>
    </tr>
    <tr>
      <td>STL</td>
      <td>1.245946</td>
      <td>0.967742</td>
      <td>0.681818</td>
      <td>0.829688</td>
      <td>0.776119</td>
    </tr>
    <tr>
      <td>USG%</td>
      <td>28.743243</td>
      <td>23.301613</td>
      <td>18.112121</td>
      <td>15.209375</td>
      <td>18.426866</td>
    </tr>
    <tr>
      <td>FTr</td>
      <td>0.364946</td>
      <td>0.249177</td>
      <td>0.364121</td>
      <td>0.190969</td>
      <td>0.210701</td>
    </tr>
    <tr>
      <td>PER</td>
      <td>22.232432</td>
      <td>16.179032</td>
      <td>19.366667</td>
      <td>12.696875</td>
      <td>11.094030</td>
    </tr>
    <tr>
      <td>TS%</td>
      <td>0.582243</td>
      <td>0.557000</td>
      <td>0.620303</td>
      <td>0.586062</td>
      <td>0.521030</td>
    </tr>
    <tr>
      <td>3PAr</td>
      <td>0.295865</td>
      <td>0.387242</td>
      <td>0.119576</td>
      <td>0.551641</td>
      <td>0.421030</td>
    </tr>
    <tr>
      <td>2019-20</td>
      <td>21894.641459</td>
      <td>14510.489919</td>
      <td>11121.900333</td>
      <td>7176.920859</td>
      <td>5218.577522</td>
    </tr>
    <tr>
      <td>ind</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
from scipy import stats
sum_stats = pd.DataFrame()
sum_stats['Variable'] = df2_km.columns[:-2]
sum_stats['Kmeans'] = np.full(len(sum_stats['Variable']),np.nan)
sum_stats['Hclust'] = np.full(len(sum_stats['Variable']),np.nan)
# the above two columns will store p-values from tests of difference between our found groups
# in all the features
```


```python
# Normality Tests: QQ Plot:
for col in df2_km.columns[:-2]:
    data1 = df2_hc.loc[df2_hc.hclust_sol==0,col]
    data2 = df2_hc.loc[df2_hc.hclust_sol==1,col]
    plt.figure()
    stats.probplot(data2, dist="norm", plot = plt);
    plt.show();
```


![png](output_43_0.png)



![png](output_43_1.png)



![png](output_43_2.png)



![png](output_43_3.png)



![png](output_43_4.png)



![png](output_43_5.png)



![png](output_43_6.png)



![png](output_43_7.png)



![png](output_43_8.png)



![png](output_43_9.png)



![png](output_43_10.png)



![png](output_43_11.png)



![png](output_43_12.png)



![png](output_43_13.png)



![png](output_43_14.png)



![png](output_43_15.png)



![png](output_43_16.png)



![png](output_43_17.png)



![png](output_43_18.png)



![png](output_43_19.png)



![png](output_43_20.png)



![png](output_43_21.png)


As we can see from the above, the majority of stats are seemingly normal judging from the plots above, however let's do a Mann-Whitney U Test to confirm this further:


```python
# For kmeans solution: 
for i in range(len(df2_km.columns[:-2])):
    col = df2_km.columns[:-2][i]
    data1 = df2_km.loc[df2_km.kmeans_sol==0,col]
    data2 = df2_km.loc[df2_km.kmeans_sol==1,col]
    test_res = stats.mannwhitneyu(data1,data2)
    sum_stats.iloc[i,1]=round(test_res[1],8)
```


```python
# and for hclust solution: 
for i in range(len(df2_hc.columns[:-2])):
    col = df2_hc.columns[:-2][i]
    data1 = df2_hc.loc[df2_hc.hclust_sol==0,col]
    data2 = df2_hc.loc[df2_hc.hclust_sol==1,col]
    test_res = stats.mannwhitneyu(data1,data2)
    sum_stats.iloc[i,2]=round(test_res[1],8)
```


```python
sum_stats
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
      <th>Variable</th>
      <th>Kmeans</th>
      <th>Hclust</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>DWS</td>
      <td>9.000000e-08</td>
      <td>3.115400e-04</td>
    </tr>
    <tr>
      <td>1</td>
      <td>WS</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>AST</td>
      <td>2.896000e-04</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>DRB</td>
      <td>2.000000e-08</td>
      <td>8.800000e-07</td>
    </tr>
    <tr>
      <td>4</td>
      <td>VORP</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>TRB</td>
      <td>7.000000e-08</td>
      <td>5.070000e-06</td>
    </tr>
    <tr>
      <td>6</td>
      <td>BPM</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>PTS</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>FGA</td>
      <td>1.000000e-08</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>OBPM</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>BLK%</td>
      <td>1.276740e-03</td>
      <td>2.101827e-01</td>
    </tr>
    <tr>
      <td>11</td>
      <td>2PA</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>TOV</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>FTA</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>BLK</td>
      <td>1.312800e-04</td>
      <td>1.476096e-01</td>
    </tr>
    <tr>
      <td>15</td>
      <td>STL</td>
      <td>1.934800e-04</td>
      <td>6.120890e-03</td>
    </tr>
    <tr>
      <td>16</td>
      <td>USG%</td>
      <td>1.000000e-08</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>17</td>
      <td>FTr</td>
      <td>8.000000e-08</td>
      <td>4.000000e-08</td>
    </tr>
    <tr>
      <td>18</td>
      <td>PER</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <td>19</td>
      <td>TS%</td>
      <td>4.620000e-05</td>
      <td>3.428703e-01</td>
    </tr>
    <tr>
      <td>20</td>
      <td>3PAr</td>
      <td>4.145280e-03</td>
      <td>4.370000e-06</td>
    </tr>
    <tr>
      <td>21</td>
      <td>2019-20</td>
      <td>2.992080e-03</td>
      <td>0.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>



The above table shows that we can see statistically significant differences in the found groups for all variables for both the Kmeans and Hclust section.

Final Takeaways from Unsupervised Learning Section:

Notes:
- The assumptions used (more then 20 games played and more then 19 minutes per game) are fairly high thresholds, which leaves ~300 rows. Clustering would be more accurate if more rows were included, however the risk of including players who don't play enough was seen as a higher risk for inaccuracy.
- A log transform was used to try to normalize the data, however more could be done to normalize the data further and perhaps improve accuracy. 
- Other stats could have been included with the 21 columns that were added to the clustering analysis for df2 (+ contract values). Some of the added columns were added in order to have every aspect of basketball covered (rather then them showing up as extremely important in the supervised learning section). For example: FTr, 3PAr and TS% were added in order to have a proxy for how often a player gets to the free throw line, how often they shoot threes and how efficient they are overall. 

Takeaways: 
There are 5 groups from the finalized clustering analysis (which can be put into the following 5 player types):

Cluster 1: “The Superstars”:
- Best Overall stats, highest WS/DWS. Statistical averages: 23.1 PPG/ 7.6TRB/ 5.5AST/ 1.2STL/ 0.7BLK. Average Contract: 21.894MM

Cluster 2: “Jack of All Trades”:
- Well-rounded stats. Stat averages: 16.0PPG / 4.5 TRB / 4.0 AST / 1.0 STL / 0.4 BLK. Average Contract: 14.511MM

Cluster 3: “Big Men”:
- Extremely efficient. Second highest WS/DWS. Statistical averages: 12.1 PPG / 8.2 TRB / 1.5 AST / 0.7 STL / 1.4 BLK. Average Contract: 11.122MM

Cluster 4: “3 and D”:
- Good defenders, third highest DWS, limited offensively. Statistical averages: 9.3 PPG / 4.0 RPG / 1.9 AST / 0.8 STL / 0.4 BLK. Average Contract: 7.177MM

Cluster 5: “Bench Player”:
- Average across the board, clearly worse than the other clusters. Statistical averages: 9.8 PPG / 3.5 RPG / 2.3 AST / 0.7 STL / 0.3 BLK. Average Contract: 5.218MM

----

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



Final Considerations and Business Cases: 
The primary business application in this case is that the clustering model can be used to analyze how much to pay for certain players. Evaluating a player against his most similar player archetype of the five above, would help determine which free agents are overpriced and which are potential bargains.
Not all the goals I hoped to achieve in this analysis could be completed due to the time restrictions. In the future, I would like to look into anciliary stats like jersey sales and social media followers in order to assess the additional value a player brings to a franchise (and likely increase the relative value of the superstar cluster). I would also like to perform clustering analysis on earlier “eras” to see if similar player types emerge.
