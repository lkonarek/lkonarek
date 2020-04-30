---
layout: post
title: Data Science in the last 20 years of the NBA: A summary: 

## Introduction and Business Questions:

NBA teams pay tens of millions of dollars to sign players on multi-year contracts. It is vital to understand not only on which metrics to evaluate players, but also how much to pay. The tendency is to evaluate players against their peers who play the same position, so it is certainly possible that certain player types or positions are overpaid relative to others.
Due to the clear business case for the NBA, I sought to answer two main questions:
- What are the best ways of evaluating a basketball player’s on court value?
- Could this on court value be used to determine if a player is over or under paid?
Currently NBA teams have comprehensive analytics departments, but most of their analytics work is focused on the teams’ style of play as a supplemental to coaching, for example: taking more 3 pointers, driving to the net more in an attempt to draw fouls, and kick out passes to players in the corners to stretch the defense. However, NBA teams could do more to apply analytics to value for money.

## Data Acquisition:

I scraped all the data from basketball-reference.com (or Wikipedia whenever data wasn’t available). It would be ideal to train models on the data for each individual season and evaluate against each other, but given that there are only about 300-400 players who play significant minutes per season (at least 8 minutes per game and 20 games played), there are too few rows for an effective model to be trained on. Therefore, multiple years were combined together into ‘eras’ for modelling: 2014-2020 (present era), 2007-2013, and 2000-2006.

## Exploration & Analysis:

While initially exploring the data, the preliminary goal was to isolate what distinguishes a good player from an elite player. All-star status was determined as the best way to do this. Once the most vital stats were isolated, on-court value vs contracts could be compared.
The dataset required little traditional cleaning. Each row in the dataset corresponded to a player, therefore null values were not a significant issue, as the only nulls were for players who didn’t record any of a percent-based stat (3p%, FT%, etc.). Null values represented less then 15% in any column, and most columns had 0% null values.
However, a number of non-conventional preprocessing decisions were made. First, players that do not play significant minutes and enough games were excluded in order to not skew the data and perhaps make the model more prone to overfitting. Also, by adding this threshold the vast majority of the nulls of the dataset were removed.
Second, another problem were players that were traded within a given season, as these players would have multiple rows (their stat totals for the year, their stats for the first team they were on, and their stats for the second team they were on). To avoid this triple counting, the best solution was to keep only the ‘total’ row for each player and drop the team column altogether.
Additional Feature Engineering and EDA work was primarily driven by domain expertise. Certain statistics were dropped as features from the models in order to mitigate issues with collinearity or because certain features were anecdotally unimportant for predicting the y-variable (all-star status). For example, the age column was dropped given that the average age of the NBA is very young, and rookies and sophomores typically don’t make the all-star game even if their stats warrant it.

## Modelling:

Modelling consisted of two portions: (1) the supervised learning portion in order to assess which statistics are the most meaningful for predicting if a player is an all-star and (2) an unsupervised learning model predicting player classes (clusters) based off the most important stats identified in part (1) and contract values.

For part (1), multiple models were considered (Logistic Regression, KNearestNeighbors, Decision Trees/Random Forests, and other ensemble methods). Ultimately, Logistic Regression was the model that was chosen, due to a combination of high accuracy statistics as well as the most important features aligning with domain expertise. For example, the stat Win Shares was near the top of all three Logistic Models. This makes sense given that Win Shares takes team success heavily into account, and the best teams usually have the most all-stars. Other model types had surprising features weighted near the top, and this was a red flag for the model’s accuracy.
Once the most meaningful statistics were identified, they were aggregated and compared against contract values for the 2019-2020 season to identify clusters in part (2) of the modelling. For this section I took 21 different categories that were identified as highly important in the 2014-2020 model (based on feature coefficients), and evaluated those stats against contract values for the 2019-2020 season only. Modelling methods consisted of Agglomerative Clustering and KMeans. Ultimately the two models varied in their results, so it is hard to know definitively which model to ‘trust’, however given that KMeans classifies data points as their proximity to the centres of clusters (how similar a given player is to a group of players), it logically makes sense to assume that KMeans is the superior model for assessing this sort of data.

## Findings & Insights: Supervised Learning:

![second pic]({{ site.baseurl }}/images/image13.png)

### Insights from Supervised Learning Modelling:
Since the main goal of the supervised modelling portion was to identify the most important features for each ‘era’, the above visual is an example of the main insight gathered. As you can see from the visual above, for the most recent era the five most important features for determining if a player makes the all-star game were: Win Shares, Defensive Win Shares, Assists, Defensive Rebounds, and Value Over Replacement Player.
Interestingly, although certain stats like 3 pointers were shown to have gone up over time, the stats for predicting the All-star status of a player were largely consistent across the three different eras. This shows us that although the game of basketball stylistically has changed dramatically, the best ways to assess on-court value has not.
 
Clustering:
Based off the most important statistics identified, the clustering analysis was performed. Ultimately it resulted in five different clusters, each of which can be identified as a common player type. The following lists include a brief profile of each cluster, the statistical averages (the most quoted statistics) and the average contract in USD:

Cluster 1: “The Superstars”:
- Best Overall stats, highest WS/DWS. Statistical averages: 23.1 PPG/ 7.6TRB/ 5.5AST/ 1.2STL/ 0.7BLK.
Average Contract: $21.894MM

![second pic]({{ site.baseurl }}/images/image8.png)

Cluster 2: “Jack-of-All-Trades”:
- Well-rounded stats. Stat averages: 16.0PPG / 4.5 TRB / 4.0 AST / 1.0 STL / 0.4 BLK. Average Contract:
14.511MM

![second pic]({{ site.baseurl }}/images/image9.png)

Cluster 3: “Big Men”:
- Extremely efficient. Second highest WS/DWS. Statistical averages: 12.1 PPG / 8.2 TRB / 1.5 AST / 0.7
STL / 1.4 BLK. Average Contract: $11.122MM

![second pic]({{ site.baseurl }}/images/image10.png)

Cluster 4: “3 and D”:
- Good defenders, third highest DWS, limited offensively. Statistical averages: 9.3 PPG / 4.0 RPG / 1.9
AST / 0.8 STL / 0.4 BLK. Average Contract: $7.177MM

![second pic]({{ site.baseurl }}/images/image11.png)

Cluster 5: “Bench Player”:
- Average across the board, clearly worse than the other clusters. Statistical averages: 9.8 PPG / 3.5
RPG / 2.3 AST / 0.7 STL / 0.3 BLK. Average Contract: $5.218MM 

![second pic]({{ site.baseurl }}/images/image12.png)


### Insights from Clustering:
- Interestingly, of the five clusters listed above, which are organized by descending average contract
values, we can see that the third cluster “Big Men”, have the second-best overall stats (since win shares and Defensive win shares were identified as the two most important stats) and are on average, ~$3.4MM cheaper. This value disparity is surprising, considering how cluster 2 has seemingly better statistical averages.
- Superstars are clearly a good value cluster; however, the average contract value is very telling. Many superstars in the NBA get paid upwards of $35MM a season, but the average contract is “only” $21.8MM, because a number of elite players are still on rookie-scale contracts (meaning there is a ceiling on the earnings they can get within their first four years). Ultimately, this cluster is less affordable then it seems.
- Judging value for money based off the findings, results in the following order (from best value per $ to worst): “Big Men” -> “Superstars” -> “Jack-of-all-trades” -> “3 and D” -> “Bench Player”.


## Conclusion: Business Applications & Future Goals:
The primary business application in this case is that the clustering model can be used to analyze how much to pay for certain players. Evaluating a player against his most similar player archetype of the five above, would help determine which free agents are overpriced and which are potential bargains.

Not all the goals I hoped to achieve in this analysis could be completed due to the time restrictions. In the future, I would like to look into anciliary stats like jersey sales and social media followers in order to assess the additional value a player brings to a franchise (and likely increase the relative value of the superstar cluster). I would also like to perform clustering analysis on earlier “eras” to see if similar player types emerge.