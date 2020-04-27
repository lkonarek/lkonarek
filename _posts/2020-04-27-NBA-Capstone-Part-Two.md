---
layout: post
title: NBA Project Part 2
---

```python
# Import the necessary packages again: 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Warnings imported due to some of the models causing many warnings: 
import warnings
warnings.filterwarnings('ignore')

# Expand the number of columns we can see: 
pd.set_option('display.max_columns', 30)
```


```python
# Read in the data 
df = pd.read_csv('data/2019-2020.csv')
```


```python
# In order to determine the right cut off for players, lets look at the middle point of minutes played: 
print(df['MP'].median())
print(df['MP'].mean())
```

    19.0
    19.576268412438615



```python
# drop the unnecessary columns for this step: 
df.drop(['Unnamed: 0', 'Drop 1', 'Drop 2', 'Player', 'Tm', 'Pos', 'Year'], axis=1, inplace=True)
#
df['All-star'] = df['All-star'].astype(int)
#
df = df[df['G']>20]
# and more then 19 mpg this time (therefore we are including half of the players in the data set who play more):
df = df[df['MP']>19.0]
```


```python
# Looking at the nulls similarly to how we did in part 1b: 
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
      <td>3P%</td>
      <td>3P%</td>
      <td>2.661597</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Since it's just 3p% (filling in with zero was explained in the previous section : 
df = df.fillna(0)
```


```python
# Column was only needed in part 2: 
df = df.drop(['All-star'], axis=1)
```


```python
# In order to get contract figures into thousands of dollars: 
df['2019-20'] = df['2019-20']/1000
```


```python
# df.describe() in order to look at the distribution: 
df.describe()
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
      <th>Age</th>
      <th>G</th>
      <th>GS</th>
      <th>MP</th>
      <th>FG</th>
      <th>FGA</th>
      <th>FG%</th>
      <th>3P</th>
      <th>3PA</th>
      <th>3P%</th>
      <th>2P</th>
      <th>2PA</th>
      <th>2P%</th>
      <th>eFG%</th>
      <th>FT</th>
      <th>...</th>
      <th>AST%</th>
      <th>STL%</th>
      <th>BLK%</th>
      <th>TOV%</th>
      <th>USG%</th>
      <th>OWS</th>
      <th>DWS</th>
      <th>WS</th>
      <th>WS/48</th>
      <th>OBPM</th>
      <th>DBPM</th>
      <th>BPM</th>
      <th>VORP</th>
      <th>2019-20</th>
      <th>Guaranteed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>...</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>2.630000e+02</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>26.707224</td>
      <td>52.410646</td>
      <td>34.129278</td>
      <td>27.405703</td>
      <td>4.839924</td>
      <td>10.499620</td>
      <td>0.462696</td>
      <td>1.451331</td>
      <td>4.003422</td>
      <td>0.336768</td>
      <td>3.387833</td>
      <td>6.496958</td>
      <td>0.520578</td>
      <td>0.533072</td>
      <td>2.163118</td>
      <td>...</td>
      <td>15.723954</td>
      <td>1.557414</td>
      <td>1.742966</td>
      <td>12.316350</td>
      <td>20.204943</td>
      <td>1.749430</td>
      <td>1.455513</td>
      <td>3.204563</td>
      <td>0.102354</td>
      <td>0.212548</td>
      <td>-0.003042</td>
      <td>0.207985</td>
      <td>0.908745</td>
      <td>10972.404749</td>
      <td>3.025977e+07</td>
    </tr>
    <tr>
      <td>std</td>
      <td>3.996399</td>
      <td>11.068429</td>
      <td>21.806190</td>
      <td>4.961063</td>
      <td>2.049903</td>
      <td>4.292346</td>
      <td>0.066452</td>
      <td>0.899272</td>
      <td>2.301167</td>
      <td>0.097950</td>
      <td>1.878282</td>
      <td>3.424841</td>
      <td>0.064947</td>
      <td>0.053610</td>
      <td>1.551547</td>
      <td>...</td>
      <td>9.355200</td>
      <td>0.565021</td>
      <td>1.498351</td>
      <td>3.018025</td>
      <td>5.801949</td>
      <td>1.635458</td>
      <td>0.865076</td>
      <td>2.189038</td>
      <td>0.056031</td>
      <td>2.372206</td>
      <td>1.271829</td>
      <td>2.690053</td>
      <td>1.201001</td>
      <td>9690.626999</td>
      <td>4.031674e+07</td>
    </tr>
    <tr>
      <td>min</td>
      <td>19.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>19.100000</td>
      <td>1.300000</td>
      <td>2.900000</td>
      <td>0.333000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.600000</td>
      <td>0.900000</td>
      <td>0.368000</td>
      <td>0.406000</td>
      <td>0.300000</td>
      <td>...</td>
      <td>3.600000</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>6.500000</td>
      <td>8.600000</td>
      <td>-1.600000</td>
      <td>-0.100000</td>
      <td>-1.300000</td>
      <td>-0.047000</td>
      <td>-5.200000</td>
      <td>-3.900000</td>
      <td>-6.700000</td>
      <td>-1.700000</td>
      <td>350.189000</td>
      <td>3.501890e+05</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>23.500000</td>
      <td>45.000000</td>
      <td>14.500000</td>
      <td>23.100000</td>
      <td>3.400000</td>
      <td>7.250000</td>
      <td>0.420000</td>
      <td>0.900000</td>
      <td>2.500000</td>
      <td>0.320500</td>
      <td>2.100000</td>
      <td>3.900000</td>
      <td>0.477000</td>
      <td>0.497000</td>
      <td>1.050000</td>
      <td>...</td>
      <td>8.300000</td>
      <td>1.100000</td>
      <td>0.700000</td>
      <td>10.100000</td>
      <td>15.750000</td>
      <td>0.600000</td>
      <td>0.800000</td>
      <td>1.650000</td>
      <td>0.066000</td>
      <td>-1.300000</td>
      <td>-0.950000</td>
      <td>-1.500000</td>
      <td>0.200000</td>
      <td>2661.420000</td>
      <td>5.458822e+06</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>27.000000</td>
      <td>55.000000</td>
      <td>37.000000</td>
      <td>27.500000</td>
      <td>4.400000</td>
      <td>9.500000</td>
      <td>0.448000</td>
      <td>1.400000</td>
      <td>3.800000</td>
      <td>0.360000</td>
      <td>2.900000</td>
      <td>5.900000</td>
      <td>0.513000</td>
      <td>0.526000</td>
      <td>1.700000</td>
      <td>...</td>
      <td>12.300000</td>
      <td>1.500000</td>
      <td>1.300000</td>
      <td>12.100000</td>
      <td>19.600000</td>
      <td>1.400000</td>
      <td>1.300000</td>
      <td>2.700000</td>
      <td>0.096000</td>
      <td>0.000000</td>
      <td>-0.100000</td>
      <td>0.000000</td>
      <td>0.700000</td>
      <td>7815.533000</td>
      <td>1.449036e+07</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>30.000000</td>
      <td>62.000000</td>
      <td>55.000000</td>
      <td>31.700000</td>
      <td>6.100000</td>
      <td>13.450000</td>
      <td>0.488000</td>
      <td>2.000000</td>
      <td>5.500000</td>
      <td>0.386000</td>
      <td>4.500000</td>
      <td>8.700000</td>
      <td>0.555000</td>
      <td>0.557000</td>
      <td>2.900000</td>
      <td>...</td>
      <td>20.300000</td>
      <td>1.900000</td>
      <td>2.200000</td>
      <td>14.150000</td>
      <td>24.050000</td>
      <td>2.500000</td>
      <td>1.900000</td>
      <td>4.300000</td>
      <td>0.129000</td>
      <td>1.600000</td>
      <td>0.900000</td>
      <td>1.600000</td>
      <td>1.400000</td>
      <td>16323.499000</td>
      <td>3.583252e+07</td>
    </tr>
    <tr>
      <td>max</td>
      <td>35.000000</td>
      <td>66.000000</td>
      <td>65.000000</td>
      <td>36.900000</td>
      <td>10.900000</td>
      <td>22.900000</td>
      <td>0.742000</td>
      <td>4.400000</td>
      <td>12.600000</td>
      <td>0.600000</td>
      <td>9.800000</td>
      <td>18.800000</td>
      <td>0.742000</td>
      <td>0.742000</td>
      <td>10.100000</td>
      <td>...</td>
      <td>49.700000</td>
      <td>3.800000</td>
      <td>8.300000</td>
      <td>21.800000</td>
      <td>37.400000</td>
      <td>8.900000</td>
      <td>4.800000</td>
      <td>11.500000</td>
      <td>0.282000</td>
      <td>7.900000</td>
      <td>4.100000</td>
      <td>11.500000</td>
      <td>6.300000</td>
      <td>38506.482000</td>
      <td>2.574293e+08</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 48 columns</p>
</div>




```python
# Let's visualize the first 10 pairplots to see the overall distribution of the dataset prior to clustering:
sns.pairplot(df.iloc[:,0:10]);
```


![png](output_9_0.png)



```python
# In a sense, Player Contracts is the most important column in this stage of the analysis, since the goal is to 
# determine the relative price of different groups of players (based on play style):
plt.hist(df['2019-20'], range=[0,40000], bins=100);
```


![png](output_10_0.png)



```python
# scale the data: 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_clust = df.copy()
df_clust[df_clust.columns] = scaler.fit_transform(df_clust[df_clust.columns])
```


```python
# Import and instantiate a KMeans model, looking at different ranges of n_clusters. 
from sklearn.cluster import KMeans

k_range = np.arange(1,20)

inertia_list = []

for k in k_range :
    
    #Specify the model
    k_means_model = KMeans(n_clusters = k)
    k_means_model.fit(df_clust)
    
    inertia_list.append(k_means_model.inertia_)
```


```python
# Visualize the inertia for each added cluster
plt.plot(k_range,inertia_list,marker = '.')
plt.show()
```


![png](output_13_0.png)



```python
# look at the cluster values: 
k_means_model = KMeans(n_clusters = 5)
k_means_model.fit(df_clust)
k_means_model.labels_
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 3, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0,
           3, 1, 0, 3, 1, 3, 4, 4, 2, 2, 4, 2, 4, 3, 1, 1, 4, 1, 4, 2, 4, 4,
           4, 0, 4, 4, 2, 2, 4, 3, 2, 1, 2, 1, 4, 4, 4, 2, 4, 1, 4, 4, 1, 3,
           4, 1, 1, 2, 3, 3, 3, 4, 3, 3, 3, 2, 2, 2, 3, 1, 2, 2, 2, 4, 4, 2,
           0, 1, 2, 4, 1, 3, 3, 4, 3, 4, 2, 4, 1, 1, 4, 1, 2, 4, 2, 4, 1, 1,
           1, 4, 1, 4, 2, 2, 4, 2, 4, 2, 2, 3, 4, 2, 1, 2, 1, 4, 1, 2, 2, 4,
           1, 2, 4, 3, 2, 1, 2, 3, 3, 4, 4, 3, 2, 3, 4, 4, 2, 2, 3, 4, 4, 1,
           2, 2, 4, 2, 1, 4, 4, 2, 1, 2, 1, 2, 3, 2, 2, 4, 1, 2, 2, 4, 4, 4,
           2, 2, 2, 4, 1, 1, 1, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 2, 4, 4,
           4, 4, 4, 4, 4, 4, 1, 3, 2, 1, 4, 2, 4, 4, 2, 3, 4, 1, 2, 1, 1, 1,
           1, 2, 4, 1, 1, 2, 4, 1, 4, 2, 4, 1, 4, 4, 3, 4, 3, 4, 0, 2, 2, 3,
           1, 0, 3, 4, 1, 4, 4, 2, 3, 1, 1, 4, 1, 2, 2, 2, 2, 3, 2, 4, 3],
          dtype=int32)




```python
df_km = df_clust.copy()
df_km['kmeans_sol'] = k_means_model.labels_
```


```python
df_km.groupby('kmeans_sol').mean().transpose()
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
      <td>Age</td>
      <td>0.037585</td>
      <td>-0.011660</td>
      <td>0.381956</td>
      <td>-0.192050</td>
      <td>-0.211882</td>
    </tr>
    <tr>
      <td>G</td>
      <td>0.320595</td>
      <td>0.142251</td>
      <td>0.156401</td>
      <td>-0.053145</td>
      <td>-0.265030</td>
    </tr>
    <tr>
      <td>GS</td>
      <td>0.996120</td>
      <td>0.592178</td>
      <td>-0.271013</td>
      <td>-0.023507</td>
      <td>-0.409947</td>
    </tr>
    <tr>
      <td>MP</td>
      <td>1.317319</td>
      <td>0.971473</td>
      <td>-0.378029</td>
      <td>-0.303489</td>
      <td>-0.542249</td>
    </tr>
    <tr>
      <td>FG</td>
      <td>1.858714</td>
      <td>0.877707</td>
      <td>-0.593992</td>
      <td>0.046613</td>
      <td>-0.588046</td>
    </tr>
    <tr>
      <td>FGA</td>
      <td>1.697364</td>
      <td>1.024623</td>
      <td>-0.590995</td>
      <td>-0.465372</td>
      <td>-0.445818</td>
    </tr>
    <tr>
      <td>FG%</td>
      <td>0.421718</td>
      <td>-0.270032</td>
      <td>-0.095154</td>
      <td>1.939772</td>
      <td>-0.614960</td>
    </tr>
    <tr>
      <td>3P</td>
      <td>0.489265</td>
      <td>0.770450</td>
      <td>0.268480</td>
      <td>-1.344993</td>
      <td>-0.288979</td>
    </tr>
    <tr>
      <td>3PA</td>
      <td>0.601837</td>
      <td>0.765886</td>
      <td>0.169987</td>
      <td>-1.410107</td>
      <td>-0.214181</td>
    </tr>
    <tr>
      <td>3P%</td>
      <td>-0.019059</td>
      <td>0.317820</td>
      <td>0.501537</td>
      <td>-1.230195</td>
      <td>-0.093919</td>
    </tr>
    <tr>
      <td>2P</td>
      <td>1.797246</td>
      <td>0.589438</td>
      <td>-0.777222</td>
      <td>0.692087</td>
      <td>-0.503014</td>
    </tr>
    <tr>
      <td>2PA</td>
      <td>1.722703</td>
      <td>0.771945</td>
      <td>-0.855580</td>
      <td>0.364845</td>
      <td>-0.416065</td>
    </tr>
    <tr>
      <td>2P%</td>
      <td>0.390703</td>
      <td>-0.502287</td>
      <td>0.331416</td>
      <td>1.470209</td>
      <td>-0.593170</td>
    </tr>
    <tr>
      <td>eFG%</td>
      <td>0.149051</td>
      <td>-0.307046</td>
      <td>0.554140</td>
      <td>1.348095</td>
      <td>-0.779193</td>
    </tr>
    <tr>
      <td>FT</td>
      <td>2.225506</td>
      <td>0.693779</td>
      <td>-0.593120</td>
      <td>-0.097736</td>
      <td>-0.502430</td>
    </tr>
    <tr>
      <td>FTA</td>
      <td>2.192147</td>
      <td>0.635786</td>
      <td>-0.640952</td>
      <td>0.139733</td>
      <td>-0.514117</td>
    </tr>
    <tr>
      <td>FT%</td>
      <td>0.373266</td>
      <td>0.388630</td>
      <td>0.207907</td>
      <td>-1.110608</td>
      <td>-0.061554</td>
    </tr>
    <tr>
      <td>ORB</td>
      <td>0.499544</td>
      <td>-0.254692</td>
      <td>-0.367003</td>
      <td>1.894139</td>
      <td>-0.422680</td>
    </tr>
    <tr>
      <td>DRB</td>
      <td>1.284924</td>
      <td>0.094314</td>
      <td>-0.347611</td>
      <td>1.099212</td>
      <td>-0.540730</td>
    </tr>
    <tr>
      <td>TRB</td>
      <td>1.107010</td>
      <td>-0.014622</td>
      <td>-0.373298</td>
      <td>1.421371</td>
      <td>-0.534375</td>
    </tr>
    <tr>
      <td>AST</td>
      <td>1.724168</td>
      <td>0.735957</td>
      <td>-0.571570</td>
      <td>-0.668206</td>
      <td>-0.201725</td>
    </tr>
    <tr>
      <td>STL</td>
      <td>0.961284</td>
      <td>0.218011</td>
      <td>-0.175967</td>
      <td>-0.340744</td>
      <td>-0.107729</td>
    </tr>
    <tr>
      <td>BLK</td>
      <td>0.193804</td>
      <td>-0.176795</td>
      <td>-0.107974</td>
      <td>1.539676</td>
      <td>-0.454024</td>
    </tr>
    <tr>
      <td>TOV</td>
      <td>1.869047</td>
      <td>0.754455</td>
      <td>-0.745614</td>
      <td>-0.137641</td>
      <td>-0.325917</td>
    </tr>
    <tr>
      <td>PF</td>
      <td>0.586785</td>
      <td>-0.068567</td>
      <td>-0.159984</td>
      <td>0.760906</td>
      <td>-0.275340</td>
    </tr>
    <tr>
      <td>PTS</td>
      <td>1.975879</td>
      <td>0.918824</td>
      <td>-0.533215</td>
      <td>-0.199489</td>
      <td>-0.592024</td>
    </tr>
    <tr>
      <td>PER</td>
      <td>1.966120</td>
      <td>0.401761</td>
      <td>-0.438610</td>
      <td>0.966708</td>
      <td>-0.783282</td>
    </tr>
    <tr>
      <td>TS%</td>
      <td>0.618176</td>
      <td>-0.137920</td>
      <td>0.484733</td>
      <td>1.194772</td>
      <td>-0.889518</td>
    </tr>
    <tr>
      <td>3PAr</td>
      <td>-0.538648</td>
      <td>-0.011180</td>
      <td>0.849320</td>
      <td>-1.596363</td>
      <td>0.126531</td>
    </tr>
    <tr>
      <td>FTr</td>
      <td>1.143664</td>
      <td>0.111044</td>
      <td>-0.528680</td>
      <td>1.061239</td>
      <td>-0.367279</td>
    </tr>
    <tr>
      <td>ORB%</td>
      <td>0.179738</td>
      <td>-0.424112</td>
      <td>-0.305443</td>
      <td>2.060697</td>
      <td>-0.347518</td>
    </tr>
    <tr>
      <td>DRB%</td>
      <td>0.762676</td>
      <td>-0.257411</td>
      <td>-0.280270</td>
      <td>1.479022</td>
      <td>-0.387015</td>
    </tr>
    <tr>
      <td>TRB%</td>
      <td>0.615291</td>
      <td>-0.330335</td>
      <td>-0.297221</td>
      <td>1.773005</td>
      <td>-0.406726</td>
    </tr>
    <tr>
      <td>AST%</td>
      <td>1.571239</td>
      <td>0.620314</td>
      <td>-0.593242</td>
      <td>-0.640418</td>
      <td>-0.085042</td>
    </tr>
    <tr>
      <td>STL%</td>
      <td>0.320387</td>
      <td>-0.190470</td>
      <td>-0.044520</td>
      <td>-0.284346</td>
      <td>0.189652</td>
    </tr>
    <tr>
      <td>BLK%</td>
      <td>-0.063756</td>
      <td>-0.345156</td>
      <td>-0.041075</td>
      <td>1.707853</td>
      <td>-0.399190</td>
    </tr>
    <tr>
      <td>TOV%</td>
      <td>0.310738</td>
      <td>-0.107391</td>
      <td>-0.491132</td>
      <td>0.345098</td>
      <td>0.226191</td>
    </tr>
    <tr>
      <td>USG%</td>
      <td>1.675008</td>
      <td>0.849926</td>
      <td>-0.687872</td>
      <td>-0.318288</td>
      <td>-0.313075</td>
    </tr>
    <tr>
      <td>OWS</td>
      <td>2.067197</td>
      <td>0.262899</td>
      <td>-0.055728</td>
      <td>0.578730</td>
      <td>-0.852735</td>
    </tr>
    <tr>
      <td>DWS</td>
      <td>1.110423</td>
      <td>0.185954</td>
      <td>0.019451</td>
      <td>0.525011</td>
      <td>-0.607436</td>
    </tr>
    <tr>
      <td>WS</td>
      <td>1.987785</td>
      <td>0.271710</td>
      <td>-0.033775</td>
      <td>0.637335</td>
      <td>-0.878543</td>
    </tr>
    <tr>
      <td>WS/48</td>
      <td>1.600425</td>
      <td>0.041254</td>
      <td>0.119120</td>
      <td>1.051820</td>
      <td>-0.912917</td>
    </tr>
    <tr>
      <td>OBPM</td>
      <td>1.965678</td>
      <td>0.633508</td>
      <td>-0.171641</td>
      <td>0.187243</td>
      <td>-0.827187</td>
    </tr>
    <tr>
      <td>DBPM</td>
      <td>0.677626</td>
      <td>-0.432283</td>
      <td>0.290841</td>
      <td>0.516763</td>
      <td>-0.304562</td>
    </tr>
    <tr>
      <td>BPM</td>
      <td>2.049041</td>
      <td>0.358168</td>
      <td>-0.013861</td>
      <td>0.406720</td>
      <td>-0.873733</td>
    </tr>
    <tr>
      <td>VORP</td>
      <td>2.300730</td>
      <td>0.391941</td>
      <td>-0.121520</td>
      <td>0.245426</td>
      <td>-0.812755</td>
    </tr>
    <tr>
      <td>2019-20</td>
      <td>1.562346</td>
      <td>0.555861</td>
      <td>-0.259170</td>
      <td>-0.046347</td>
      <td>-0.523168</td>
    </tr>
    <tr>
      <td>Guaranteed</td>
      <td>1.740806</td>
      <td>0.634229</td>
      <td>-0.348899</td>
      <td>-0.287571</td>
      <td>-0.455379</td>
    </tr>
  </tbody>
</table>
</div>




```python
for col in df_km.columns:
    plt.figure();
    sns.boxplot(y=col,x='kmeans_sol',data=df_km);
    plt.show;
```


![png](output_17_0.png)



![png](output_17_1.png)



![png](output_17_2.png)



![png](output_17_3.png)



![png](output_17_4.png)



![png](output_17_5.png)



![png](output_17_6.png)



![png](output_17_7.png)



![png](output_17_8.png)



![png](output_17_9.png)



![png](output_17_10.png)



![png](output_17_11.png)



![png](output_17_12.png)



![png](output_17_13.png)



![png](output_17_14.png)



![png](output_17_15.png)



![png](output_17_16.png)



![png](output_17_17.png)



![png](output_17_18.png)



![png](output_17_19.png)



![png](output_17_20.png)



![png](output_17_21.png)



![png](output_17_22.png)



![png](output_17_23.png)



![png](output_17_24.png)



![png](output_17_25.png)



![png](output_17_26.png)



![png](output_17_27.png)



![png](output_17_28.png)



![png](output_17_29.png)



![png](output_17_30.png)



![png](output_17_31.png)



![png](output_17_32.png)



![png](output_17_33.png)



![png](output_17_34.png)



![png](output_17_35.png)



![png](output_17_36.png)



![png](output_17_37.png)



![png](output_17_38.png)



![png](output_17_39.png)



![png](output_17_40.png)



![png](output_17_41.png)



![png](output_17_42.png)



![png](output_17_43.png)



![png](output_17_44.png)



![png](output_17_45.png)



![png](output_17_46.png)



![png](output_17_47.png)



![png](output_17_48.png)


In part 2, we identified a number of different features that were highly involved in success in the most recent era:
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

Let's re-run our clustering models with just these features included (with the 2019-2020 contract value)


```python
df2 = df[['DWS', 'WS', 'AST', 'DRB', 'VORP', 'TRB', 'BPM', 'PTS', 'FGA', 'OBPM', 'BLK%', '2PA', 'TOV', 'FTA', 'BLK', 
         'STL', 'USG%', 'FTr', 'PER', 'TS%', '3PAr', '2019-20']]
```


```python
sns.pairplot(df2.iloc[:,0:]);
```


![png](output_20_0.png)



```python
df2.describe()
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
      <th>DWS</th>
      <th>WS</th>
      <th>AST</th>
      <th>DRB</th>
      <th>VORP</th>
      <th>TRB</th>
      <th>BPM</th>
      <th>PTS</th>
      <th>FGA</th>
      <th>OBPM</th>
      <th>BLK%</th>
      <th>2PA</th>
      <th>TOV</th>
      <th>FTA</th>
      <th>BLK</th>
      <th>STL</th>
      <th>USG%</th>
      <th>FTr</th>
      <th>PER</th>
      <th>TS%</th>
      <th>3PAr</th>
      <th>2019-20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
      <td>263.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>1.455513</td>
      <td>3.204563</td>
      <td>2.926996</td>
      <td>3.950190</td>
      <td>0.908745</td>
      <td>5.023954</td>
      <td>0.207985</td>
      <td>13.295817</td>
      <td>10.499620</td>
      <td>0.212548</td>
      <td>1.742966</td>
      <td>6.496958</td>
      <td>1.643346</td>
      <td>2.769582</td>
      <td>0.536122</td>
      <td>0.888593</td>
      <td>20.204943</td>
      <td>0.255920</td>
      <td>15.287833</td>
      <td>0.566403</td>
      <td>0.389414</td>
      <td>10972.404749</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.865076</td>
      <td>2.189038</td>
      <td>1.932366</td>
      <td>1.913928</td>
      <td>1.201001</td>
      <td>2.620223</td>
      <td>2.690053</td>
      <td>5.828584</td>
      <td>4.292346</td>
      <td>2.372206</td>
      <td>1.498351</td>
      <td>3.424841</td>
      <td>0.829342</td>
      <td>1.905192</td>
      <td>0.477936</td>
      <td>0.364272</td>
      <td>5.801949</td>
      <td>0.118337</td>
      <td>4.621589</td>
      <td>0.048201</td>
      <td>0.190761</td>
      <td>9690.626999</td>
    </tr>
    <tr>
      <td>min</td>
      <td>-0.100000</td>
      <td>-1.300000</td>
      <td>0.500000</td>
      <td>0.900000</td>
      <td>-1.700000</td>
      <td>1.500000</td>
      <td>-6.700000</td>
      <td>3.600000</td>
      <td>2.900000</td>
      <td>-5.200000</td>
      <td>0.000000</td>
      <td>0.900000</td>
      <td>0.500000</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>8.600000</td>
      <td>0.063000</td>
      <td>4.100000</td>
      <td>0.430000</td>
      <td>0.000000</td>
      <td>350.189000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>0.800000</td>
      <td>1.650000</td>
      <td>1.500000</td>
      <td>2.600000</td>
      <td>0.200000</td>
      <td>3.200000</td>
      <td>-1.500000</td>
      <td>9.200000</td>
      <td>7.250000</td>
      <td>-1.300000</td>
      <td>0.700000</td>
      <td>3.900000</td>
      <td>1.000000</td>
      <td>1.400000</td>
      <td>0.200000</td>
      <td>0.600000</td>
      <td>15.750000</td>
      <td>0.175000</td>
      <td>12.000000</td>
      <td>0.536000</td>
      <td>0.283000</td>
      <td>2661.420000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>1.300000</td>
      <td>2.700000</td>
      <td>2.100000</td>
      <td>3.700000</td>
      <td>0.700000</td>
      <td>4.500000</td>
      <td>0.000000</td>
      <td>12.000000</td>
      <td>9.500000</td>
      <td>0.000000</td>
      <td>1.300000</td>
      <td>5.900000</td>
      <td>1.400000</td>
      <td>2.300000</td>
      <td>0.400000</td>
      <td>0.800000</td>
      <td>19.600000</td>
      <td>0.231000</td>
      <td>14.700000</td>
      <td>0.563000</td>
      <td>0.404000</td>
      <td>7815.533000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>1.900000</td>
      <td>4.300000</td>
      <td>3.950000</td>
      <td>4.800000</td>
      <td>1.400000</td>
      <td>6.200000</td>
      <td>1.600000</td>
      <td>17.000000</td>
      <td>13.450000</td>
      <td>1.600000</td>
      <td>2.200000</td>
      <td>8.700000</td>
      <td>2.050000</td>
      <td>3.550000</td>
      <td>0.600000</td>
      <td>1.100000</td>
      <td>24.050000</td>
      <td>0.311000</td>
      <td>17.900000</td>
      <td>0.593000</td>
      <td>0.524000</td>
      <td>16323.499000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>4.800000</td>
      <td>11.500000</td>
      <td>10.600000</td>
      <td>11.500000</td>
      <td>6.300000</td>
      <td>15.800000</td>
      <td>11.500000</td>
      <td>34.400000</td>
      <td>22.900000</td>
      <td>7.900000</td>
      <td>8.300000</td>
      <td>18.800000</td>
      <td>4.800000</td>
      <td>11.800000</td>
      <td>3.100000</td>
      <td>2.100000</td>
      <td>37.400000</td>
      <td>0.725000</td>
      <td>31.600000</td>
      <td>0.726000</td>
      <td>0.886000</td>
      <td>38506.482000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# As we can see from the above code the lowest negative number in any column is in BPM with -6.7, therefore let's add
# +7 to each column before doing a log transform. Saving it as df3 so we can compare df2 (non-transformed) to df3 (
# transformed):

df2 = np.log(df2+7)
```


```python
# Same process as above: Scaler and then fit both model methods:
scaler2 = StandardScaler()
df2[df2.columns] = scaler2.fit_transform(df2[df2.columns])
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
-	Best Overall stats, highest WS/DWS. Statistical averages: 23.1 PPG/ 7.6TRB/ 5.5AST/ 1.2STL/ 0.7BLK. Average Contract: 21.894MM

Cluster 2: “Jack of All Trades”:
-	Well-rounded stats. Stat averages: 16.0PPG / 4.5 TRB / 4.0 AST / 1.0 STL / 0.4 BLK. Average Contract: 14.511MM

Cluster 3: “Big Men”:
-	Extremely efficient. Second highest WS/DWS. Statistical averages: 12.1 PPG / 8.2 TRB / 1.5 AST / 0.7 STL / 1.4 BLK. Average Contract: 11.122MM

Cluster 4: “3 and D”:
-	Good defenders, third highest DWS, limited offensively. Statistical averages: 9.3 PPG / 4.0 RPG / 1.9 AST / 0.8 STL / 0.4 BLK. Average Contract: 7.177MM

Cluster 5: “Bench Player”:
-	Average across the board, clearly worse than the other clusters. Statistical averages: 9.8 PPG / 3.5 RPG / 2.3 AST / 0.7 STL / 0.3 BLK. Average Contract: 5.218MM

