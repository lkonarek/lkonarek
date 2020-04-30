---
layout: post
title: Data Science & the NBA: Comprehensive Analysis 
---

## Introduction: 

As someone who is practically a life-long NBA fan, I have spent hours (perhaps even days) of my life staring at the stats of my favorite players. Obsessively memorizing stats so I could beat my friends in arguments about which player was better was my primary goal I must admit, but what started at what stats are the single best to follow, became so I could understand what all the stats meant. 

Basketball may be the single sport with the most stats (other then perhaps baseball), and anyone who has ever gone to a specific players profile knows that they keep track of an amazing amount of things, everything from scored points to tipped passes. And it is understandable why this is the case, NBA teams pay tens of millions of dollars to sign players on multi-year contracts. It is vital to understand not only on which metrics to evaluate players, but also how much to pay. The tendency is to evaluate players against their peers who play the same position, so it is certainly possible that certain player types or positions are overpaid relative to others.
Due to the clear business case for the NBA, I sought to answer two main questions:
- What are the best ways of evaluating a basketball player’s on court value?
- Could this on court value be used to determine if a player is over or under paid?
Currently NBA teams have comprehensive analytics departments, but most of their analytics work is focused on the teams’ style of play as a supplemental to coaching, for example: taking more 3 pointers, driving to the net more in an attempt to draw fouls, and kick out passes to players in the corners to stretch the defense. However, NBA teams could do more to apply analytics to value for money.

## Data Acquisition:

Given that basketball stats are readily available online, I manually scraped all the stats from basketball-reference.com (using beautifulsoup w/ Python). My reasoning for doing this rather then downloading a dataset from Kaggle or another online depository was due to the fact that I wanted all the clean-up and dataframe schemas to be the same for each individual season I processed. 

In a perfect world, it would be ideal to train models on the data for each individual season and evaluate against each other, but given that there are only about 300-400 players who play significant minutes per season (at least 8 minutes per game and 20 games played), there are too few rows for an effective model to be trained on. Therefore, multiple years were combined together into ‘eras’ for modelling: 2014-2020 (present era), 2007-2013, and 2000-2006.

## EDA & Summary of Cleaning:
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
df14_20['3P%'] = df14_20['3P%'].fillna(0)
df07_13['3P%'] = df07_13['3P%'].fillna(0)
df00_06['3P%'] = df00_06['3P%'].fillna(0)
```

The Last cleaning decision that needs to be made is regarding positions, at this point the data is saved as such 
```python
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

Ultimately the best way to deal with this seems to be to remove the secondary position altogether, as it is inconsistent whether a player has more then one listed position. For the purposes of the modelling in this exercise, let's try to isolate the 5 positions, and assume each player mostly plays their primary position.  


```python
df14_20['Pos'] = df14_20['Pos'].astype(str)
df14_20['Pos'] = df14_20['Pos'].str.replace(r'-\w\w', '')
df14_20['Pos'] = df14_20['Pos'].str.replace(r'-\w', '')
df14_20['Pos'].value_counts()
```
Now these are the values of the positions remaining (for all the seasons from 2014-2020):

    SG    601
    PG    554
    PF    549
    C     543
    SF    490
    Name: Pos, dtype: int64

From here on, only the code altering the df14_20 will be shown, however assume all changes that are completed to the df14_20 dataframe are also completed with the other two dataframes: 

Now the only remaining categorical variables are Position and Year. Let's use pd.get_dummies to encode them individually (drop_first = true because its not necessary to have a column for every option):
```python
df14_20 = pd.get_dummies(df14_20, drop_first=True)
```

## Supervised Modelling:

First let's state the X and y variables clearly in order to do modelling:
```python
y20 = df14_20['All-star']
X20 = df14_20.drop(['All-star'], axis=1)
```

Then let's split up the remainder and test set. The test set is set aside so there is no data leakage.
```python
from sklearn.model_selection import train_test_split
# Split up our data sets into remainder and test set:
# For the 2014-2020 dataset: 
X_remainder20, X_test20, y_remainder20, y_test20 = train_test_split(X20, y20, test_size=0.25, 
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
```

    [[   0 1914]
     [   1 1914]]

Now this shows how many data points we have to train and optimize our models on. It is worth repeating that the test set has remained unaltered by resampling methods, as the test set needs to remain unseen data. 


Then a second split needs to be made where we will seperate the remainder set (after smote) into the train and validation sets which will actually be used to tune the primary models. 
```python
X_train20, X_validation20, y_train20, y_validation20 = train_test_split(X_smote20, y_smote20, test_size=0.2, 
                                                            random_state=1, stratify=y_smote20)
```

Scaling the data is important, however in order to pull out the coefficients of the best performing model 'X_train20' needs to remain as a dataframe with the column names present (normally scalers transform dataframes into a np array):
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(X_train20)

# transform X_train, however put X_train in to a df in order to easier pull out the coefficients:
scaled_features = scaler.transform(X_train20.values)
X_train20 = pd.DataFrame(scaled_features, index=X_train20.index, columns=X_train20.columns)

# transform X_test and X_validation as well:
X_test20 = scaler.transform(X_test20)
X_validation20 = scaler.transform(X_validation20)
X_smote20 = scaler.transform(X_smote20)
```

```python
from sklearn.model_selection import cross_val_score

# Since we are going to use cross validation to determine the best parameter for C, we will perform a similar loop as
# before, looping through a range of possible C values:
C_range = np.array([0.00001, 0.0001, 0.001, 0.01,0.1,1,10,100,1000, 10000])
cv_scores20=[]
# loop over different C_values in order to find the optimal value for 5 crossfolds.
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

![first pic]({{ site.baseurl }}/images/image1.png)

From the above, it makes sense to use a C value of 10^-1 for all three model. In addition, setting C values lower could help mitigate some overfitting issues, as higher C-values generally run a higher risk of overfitting on the train set. Therefore let's re-instantiate each of the three models with C=0.1:

```python
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

The above lists out all the table variables after all preprocessing ordered by their correlation coefficients in the model for #2014-2020#, or in other words, how highly correlated a given variable is in predicting the outcome of the y-variable according to the model. The top 5 are as such:
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

For #2007-2013# the top 5 features are: 
1. Win shares 
2. Defensive Win shares
3. Total Rebounds
4. Games Started 
5. Defensive Rebounds

and the bottom 5 features are: 
1. Games
2. Personal Fouls
3. Steal %
4. Minutes Played
5. Player Efficiency Rating 


For #2000-2006# the top 5 features are:
1. Defensive Win Shares
2. Win Shares
3. Value Over Replacement Player
4. Box Plus Minus
5. Points

and the bottom 5 features are: 
1. Minutes Played
2. Personal Fouls
3. Steal %
4. Player Efficiency Rating
5. Games

Interestingly, we can see that many of the top 5 features and bottom 5 features are quite consistent across the different eras. Although there are differences noticed over time, this shows that the best ways of evaluating players has not really changed much over the past 20 years. Also stats that are often quoted as being good metrics for evaluating players, Player Efficiency Rating for example, is consistently one of the worst predictors for determining if a player is elite.

```python
# Now let's look at how the models predict the test set:
y_pred20 = myLog.predict(X_test20)
y_pred13 = myLog2.predict(X_test13)
y_pred06 = myLog3.predict(X_test06)
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

![second pic]({{ site.baseurl }}/images/image2.png)


The same was performed for 2007-2013:
    Test AUC score: 0.9747886473429952
    Train AUC score: 0.9910078868396105
And 2000-2006:
    Test AUC score: 0.9829343971631206
    Train AUC score: 0.9920090662118183


The ROC curve above and the three corresponding AUC scores certainly show evidence of overfitting, however the extremely low false positive rate (data points the model classified as all-stars who were in fact not all-stars) is an encouraging sign that the models are effective for their purpose.

Although the best performing Logistic Model was used as the final best predictive model due to its interpretability and accuracy, there were many other models I considered including other techniques like Principal component analysis, feature engineering, ensembling, building pipelines to hyperoptimize model parameters, as well as try out other simple modelling techniques like decision trees, k-nearest neighbors and SVM. For the purposes of this blog post, these other methods will be left out, but I will make a longer form blog post that includes all the other code and things I tried anyone wants to take a more indepth look. 


Interestingly, by adding the three main models I tried: Decision Trees, Logistic Regression and K-nearest neighbors, the model performed only marginally better then the logistic regression model alone:

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


Now that we have identified some important features from the best model available, we take this information into the second phase of modelling: unsupervised/clustering. 

```python
# Read in the data again, this time just 2019-2020:
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

```
```python
# Let's zoom in closer to the first 9 points on the graph:

plt.figure()
plt.scatter(np.arange(12),inertia_list[0:12])
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show();
```

![third pic]({{ site.baseurl }}/images/image3.png)


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


![second pic]({{ site.baseurl }}/images/image4.png)



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

Now let's combine the two methods together into one dataframe so we can visualize the clusters that the two methods identified: 

```python
df2_hc = df2.copy()
df2_hc['hclust_sol'] = hclust_model.labels_
df2_km['ind'] = np.ones(len(df2_km.kmeans_sol))
df2_hc['ind'] = np.zeros(len(df2_km.kmeans_sol))
```

```python
# Stick the two graphs together to visualize: 
full_df2 = pd.concat([df2_km,df2_hc])
# done in this way because seaborn requires data to be this way: 
full_df2.hclust_sol[:len(df2_km.kmeans_sol)] = df2_km.kmeans_sol
```

This code creates plots for the clusters for each stat category (with the different clusters identified by their colours). Although we found the adjusted rand score was quite low, when we visualize many of the clusters, similar clusters are emerging. (Given the number of different graphs only a few are included in this blog post):
```python
# plotting side by side histograms for all features 
# left is hclust (ind = 0) and right is kmeans (ind = 1)
for col in full_df2.columns[:-3]:
    plt.figure(figsize=(20,30));
    g = sns.FacetGrid(full_df2, col='ind',hue='hclust_sol', margin_titles=True,height=8);
    g.map(sns.distplot, col);
    plt.show();
```
Assists:
![second pic]({{ site.baseurl }}/images/image5.png)
Blocks:
![second pic]({{ site.baseurl }}/images/image6.png)
2019-2020 Contract Values per cluster (The most important):
![second pic]({{ site.baseurl }}/images/image7.png)

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


It is important to check for normality, as many clusters will be misleading if the data is non-normal (one reason for the log transform earlier was because of this). Ultimately I created QQ Plots aswell as a Mann-Whitney U Test, and found that the data was quite normal but that there were statistically significant differences found in the groups for all variables both in the Kmeans and Hclust section, so this puts some of the results into a bit of question, however is also expected given the very low adjusted rand score earlier. 

## Final Notes and Findings of Modelling: 
Supervised Modelling: 
- In this example, logistic regression made the most sense, given that it had the best (or close to the best) scores and has the extra intrepretability of being able to easily find the feature coefficients and intepret them. In addition, incorporating domain specific knowledge the results of the most important categories make the most sense with using a Logistic model.
- This classification problem in general seems relatively easy for a model to determine, which is an interesting finding. However despite the very strong accuracy figures, precision is extremely low, therefore the notion that there are many 'snubs' (players who deserve to be all-stars but don't make it) each year is supported by the models. 
- It is worth reiterating again all the logistic models had high recall, which means these models are casting too wide of a net, in other words the models classified the player as an all-star more then the true number of all-stars in the data set. 
- It is worth summarazing again the subjective assumptions/decisions made in the first part of this document including: 
    - the threshold cutoffs (at least 20 games played and 8 minutes played), 
    - the decision to drop the team column entirely after removing the individual team rows for players who were traded in the season.
    - the decision to classify all players as their primary or first-listed position only.
- Due to most of the models above showing signs of overfitting, decisions were often made to be more conservative to mitigate the affect of this problem (ex: smaller C values for logistic regression, etc.).
- Many of the stats found to be most impactful in the present era were also found to be the most impactful in previous eras as well. This is a surprising finding; although I understood that stats like WS/DWS were very important I was not expecting them to be #1/#2 respectively for each year. DWS makes some sense as a very impactful statistic given that very few bball stats account for defense (blocks and steals are two of the most quoted) and defense accounts for half of all basketball, however the consistency of it being in the top 2 for each era was a very interesting finding nontheless.
- Other more complicated models/ensemble methods were assessed as possible options, however did not perform better then a Logistic Model and involved a lot of time spent on hyperoptimizing parameters. Grid Searches as well could be used in order to get the exact best parameters, however given that the goal of this exercise was primarily to find the most important features and isolate them for the clustering portion, grid searches didn't really fit into that (given that it would be harder to interpet the feature weights/coefficients)

Unsupervised Modelling: 
- The assumptions used (more then 20 games played and more then 19 minutes per game) are fairly high thresholds, which leaves ~300 rows. Clustering would be more accurate if more rows were included, however the risk of including players who don't play enough was seen as a higher risk for inaccuracy.
- A log transform was used to try to normalize the data, however more could be done to normalize the data further and perhaps improve accuracy. 
- Other stats could have been included with the 21 columns that were added to the clustering analysis for df2 (+ contract values). Some of the added columns were added in order to have every aspect of basketball covered (rather then them showing up as extremely important in the supervised learning section). For example: FTr, 3PAr and TS% were added in order to have a proxy for how often a player gets to the free throw line, how often they shoot threes and how efficient they are overall. 

Findings: 
There are 5 groups from the finalized clustering analysis (which can be put into the following 5 player types):

Cluster 1: “The Superstars”:
- Best Overall stats, highest WS/DWS. Statistical averages: 23.1 PPG/ 7.6TRB/ 5.5AST/ 1.2STL/ 0.7BLK. Average Contract: 21.894MM
![second pic]({{ site.baseurl }}/images/image8.png)


Cluster 2: “Jack of All Trades”:
- Well-rounded stats. Stat averages: 16.0PPG / 4.5 TRB / 4.0 AST / 1.0 STL / 0.4 BLK. Average Contract: 14.511MM
![second pic]({{ site.baseurl }}/images/image9.png)


Cluster 3: “Big Men”:
- Extremely efficient. Second highest WS/DWS. Statistical averages: 12.1 PPG / 8.2 TRB / 1.5 AST / 0.7 STL / 1.4 BLK. Average Contract: 11.122MM
![second pic]({{ site.baseurl }}/images/image10.png)


Cluster 4: “3 and D”:
- Good defenders, third highest DWS, limited offensively. Statistical averages: 9.3 PPG / 4.0 RPG / 1.9 AST / 0.8 STL / 0.4 BLK. Average Contract: 7.177MM
![second pic]({{ site.baseurl }}/images/image11.png)


Cluster 5: “Bench Player”:
- Average across the board, clearly worse than the other clusters. Statistical averages: 9.8 PPG / 3.5 RPG / 2.3 AST / 0.7 STL / 0.3 BLK. Average Contract: 5.218MM
![second pic]({{ site.baseurl }}/images/image12.png)


## Conclusion and Business Cases: 
The primary business application in this case is that the clustering model can be used to analyze how much to pay for certain players. Evaluating a player against his most similar player archetype of the five above, would help determine which free agents are overpriced and which are potential bargains.
Not all the goals I hoped to achieve in this analysis could be completed due to the time restrictions. In the future, I would like to look into anciliary stats like jersey sales and social media followers in order to assess the additional value a player brings to a franchise (and likely increase the relative value of the superstar cluster). I would also like to perform clustering analysis on earlier “eras” to see if similar player types emerge.
