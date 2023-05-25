### Title: Analysis of India's Two Wheeler Insurance Market
India two wheeler insurance market can be segmented based on vehicle type, type, source, premium type and region. Based on the type, third party insurance is the dominant segment due to its low cost when compared with its counterpart. On the basis of premium type, the market can be bifurcated into personal insurance premium and commercial insurance premium.
Years considered for this report: (This is a dummy dataset of 7000 instances)

Historical Years: 2014-2017

Base Year: 2018
Key Players:
- List the major companies operating in the India Two Wheeler Insurance Market.

- Mention some of the key players, such as The New India Assurance', 'HDFC ERGO General Insurance','The Oriental Insurance Company','TATA AIG General Insurance','ICICI Lombard General Insurance'.

### Objective of the Study:
- To analyze the market size of India two wheeler insurance market.
- To classify India two wheeler insurance market based on vehicle type, type, source, premium type and regional distribution.
- To identify drivers and challenges for India two wheeler insurance market.
- To identify and analyze the profile of leading players operating in India two wheeler insurance market.
##### Making a dummy dataset, I couldn't find the api for India Two Wheeler Insurance Market


```python
import numpy as np
import pandas as pd

np.random.seed(123)

insurance_providers = [
    'The Oriental Insurance Company',
    'HDFC ERGO General Insurance',
    'TATA AIG General Insurance',
    'The New India Assurance',
    'ICICI Lombard General Insurance'
]

claim_settlement_ratios = [91.76, 91.23, 90.49, 89.60, 87.771]

num_records = 10000
policy_holders = np.random.randint(1000, 5000, size=num_records).astype(float)
premium_amounts = np.random.randint(20000, 50000, size=num_records).astype(float)
claim_amounts = np.random.randint(20000, 100000, size=num_records).astype(float)

# Introduce random empty cells by assigning NaN values
mask = np.random.choice([True, False], size=num_records, p=[0.2, 0.8])
policy_holders[mask] = np.nan
premium_amounts[mask] = np.nan
claim_amounts[mask] = np.nan

# Generate random premium types
premium_types = np.random.choice(['Personal Insurance Premium', 'Commercial Insurance Premium'], size=num_records)

data = {
    'Insurance': np.random.choice(insurance_providers, size=num_records),
    'Policy Holders': policy_holders,
    'Premium Amounts': premium_amounts,
    'Claim Amounts': claim_amounts,
    'Premium Type': premium_types
}

# Create a DataFrame from the data
df = pd.DataFrame(data)
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
      <th>Insurance</th>
      <th>Policy Holders</th>
      <th>Premium Amounts</th>
      <th>Claim Amounts</th>
      <th>Premium Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The New India Assurance</td>
      <td>4582.0</td>
      <td>41970.0</td>
      <td>63849.0</td>
      <td>Commercial Insurance Premium</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TATA AIG General Insurance</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Commercial Insurance Premium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The New India Assurance</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Personal Insurance Premium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HDFC ERGO General Insurance</td>
      <td>2346.0</td>
      <td>47539.0</td>
      <td>23521.0</td>
      <td>Commercial Insurance Premium</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HDFC ERGO General Insurance</td>
      <td>2122.0</td>
      <td>35455.0</td>
      <td>73845.0</td>
      <td>Commercial Insurance Premium</td>
    </tr>
  </tbody>
</table>
</div>




```python
year_mapping = {2014: 2018, 2015: 2017, 2017: 2015, 2018: 2014}
df['Year'] = np.random.choice(['2018','2017','2016', '2015', '2014'], size=len(df))
df['Year'] = df['Year'].replace(year_mapping)

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
      <th>Insurance</th>
      <th>Policy Holders</th>
      <th>Premium Amounts</th>
      <th>Claim Amounts</th>
      <th>Premium Type</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The New India Assurance</td>
      <td>4582.0</td>
      <td>41970.0</td>
      <td>63849.0</td>
      <td>Commercial Insurance Premium</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TATA AIG General Insurance</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Commercial Insurance Premium</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The New India Assurance</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Personal Insurance Premium</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HDFC ERGO General Insurance</td>
      <td>2346.0</td>
      <td>47539.0</td>
      <td>23521.0</td>
      <td>Commercial Insurance Premium</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HDFC ERGO General Insurance</td>
      <td>2122.0</td>
      <td>35455.0</td>
      <td>73845.0</td>
      <td>Commercial Insurance Premium</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.to_csv('D:\Analytics\Aegys Covenant Cust Dataset\Main\insurancedata1.csv')
```

.


```python
df2 = df.copy()
```


```python
df2.drop('Year', axis=1, inplace=True)
```


```python
df2.shape
```




    (7983, 5)




```python
df2=df2.dropna()
df2
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
      <th>Insurance</th>
      <th>Policy Holders</th>
      <th>Premium Amounts</th>
      <th>Claim Amounts</th>
      <th>Premium Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The New India Assurance</td>
      <td>4582.0</td>
      <td>41970.0</td>
      <td>63849.0</td>
      <td>Commercial Insurance Premium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HDFC ERGO General Insurance</td>
      <td>2346.0</td>
      <td>47539.0</td>
      <td>23521.0</td>
      <td>Commercial Insurance Premium</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HDFC ERGO General Insurance</td>
      <td>2122.0</td>
      <td>35455.0</td>
      <td>73845.0</td>
      <td>Commercial Insurance Premium</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HDFC ERGO General Insurance</td>
      <td>2766.0</td>
      <td>36284.0</td>
      <td>30076.0</td>
      <td>Personal Insurance Premium</td>
    </tr>
    <tr>
      <th>6</th>
      <td>The New India Assurance</td>
      <td>4089.0</td>
      <td>23059.0</td>
      <td>79203.0</td>
      <td>Commercial Insurance Premium</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>TATA AIG General Insurance</td>
      <td>4117.0</td>
      <td>42532.0</td>
      <td>40497.0</td>
      <td>Personal Insurance Premium</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>TATA AIG General Insurance</td>
      <td>3608.0</td>
      <td>42243.0</td>
      <td>48466.0</td>
      <td>Personal Insurance Premium</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>TATA AIG General Insurance</td>
      <td>4401.0</td>
      <td>36990.0</td>
      <td>27490.0</td>
      <td>Commercial Insurance Premium</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>TATA AIG General Insurance</td>
      <td>1082.0</td>
      <td>24775.0</td>
      <td>71633.0</td>
      <td>Commercial Insurance Premium</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>TATA AIG General Insurance</td>
      <td>2114.0</td>
      <td>46319.0</td>
      <td>56320.0</td>
      <td>Commercial Insurance Premium</td>
    </tr>
  </tbody>
</table>
<p>7983 rows × 5 columns</p>
</div>



.

.

### Backups


```python
df10=insurance_data.copy()
```

.

.


```python
length = len(df2)
length
```




    7983




```python
# I have added some more information in the data
import pandas as pd
import numpy as np

# Total number of vehicle crashed on specific month/year
# months of casualties
months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

df2['Insurance Year'] = np.random.choice(['2018', '2017', '2016', '2015', '2014'], size=length)
df2['Insurance Year'] = df2['Insurance Year'].astype(int)
df2['Casualty Year'] = np.random.choice(['2014', '2018', '2017', '2016', '2015'], size=length)
df2['Casualty Year'] = df2['Insurance Year'].astype(int)
df2['Casualty Year'] = np.maximum( df2['Insurance Year'] + 1, df2['Casualty Year'])
df2['Casualty Month'] = np.random.choice(months, size=length)

number_of_vehicles_crashed = np.random.randint(1,11,size=length)
sex_of_casualty = np.random.choice(['Female','Male'], size=length)
age_of_casualty = np.random.randint(18,80,size=length)   # Age range from 18-80

df2['Total Vehicles Crashed'] = number_of_vehicles_crashed
df2['Sex'] = sex_of_casualty
df2['Age'] = age_of_casualty
df2['Policy Holders'] = df2['Policy Holders'].astype(int)
df2.head()
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
      <th>Insurance</th>
      <th>Policy Holders</th>
      <th>Premium Amounts</th>
      <th>Claim Amounts</th>
      <th>Premium Type</th>
      <th>Insurance Year</th>
      <th>Casualty Year</th>
      <th>Casualty Month</th>
      <th>Total Vehicles Crashed</th>
      <th>Sex</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The New India Assurance</td>
      <td>4582</td>
      <td>41970.0</td>
      <td>63849.0</td>
      <td>Commercial Insurance Premium</td>
      <td>2015</td>
      <td>2016</td>
      <td>November</td>
      <td>2</td>
      <td>Female</td>
      <td>72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HDFC ERGO General Insurance</td>
      <td>2346</td>
      <td>47539.0</td>
      <td>23521.0</td>
      <td>Commercial Insurance Premium</td>
      <td>2015</td>
      <td>2016</td>
      <td>June</td>
      <td>10</td>
      <td>Female</td>
      <td>48</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HDFC ERGO General Insurance</td>
      <td>2122</td>
      <td>35455.0</td>
      <td>73845.0</td>
      <td>Commercial Insurance Premium</td>
      <td>2018</td>
      <td>2019</td>
      <td>May</td>
      <td>1</td>
      <td>Male</td>
      <td>78</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HDFC ERGO General Insurance</td>
      <td>2766</td>
      <td>36284.0</td>
      <td>30076.0</td>
      <td>Personal Insurance Premium</td>
      <td>2015</td>
      <td>2016</td>
      <td>July</td>
      <td>3</td>
      <td>Female</td>
      <td>42</td>
    </tr>
    <tr>
      <th>6</th>
      <td>The New India Assurance</td>
      <td>4089</td>
      <td>23059.0</td>
      <td>79203.0</td>
      <td>Commercial Insurance Premium</td>
      <td>2015</td>
      <td>2016</td>
      <td>April</td>
      <td>7</td>
      <td>Female</td>
      <td>59</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.to_csv(r'D:\Analytics\Aegys Covenant Cust Dataset\Road Traffic Accident\Main\New folder\df2.csv')
```


```python
df2.shape
```




    (7983, 11)




```python
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7983 entries, 0 to 9999
    Data columns (total 11 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Insurance               7983 non-null   object 
     1   Policy Holders          7983 non-null   int32  
     2   Premium Amounts         7983 non-null   float64
     3   Claim Amounts           7983 non-null   float64
     4   Premium Type            7983 non-null   object 
     5   Insurance Year          7983 non-null   int32  
     6   Casualty Year           7983 non-null   int32  
     7   Casualty Month          7983 non-null   object 
     8   Total Vehicles Crashed  7983 non-null   int32  
     9   Sex                     7983 non-null   object 
     10  Age                     7983 non-null   int32  
    dtypes: float64(2), int32(5), object(4)
    memory usage: 592.5+ KB
    

.

.

.

 ### I made the dummy dataset of 8000 instances with all the info to Analyze India Insurance Maket

#### Step 1: Data Understanding and Preparation:


```python
df3 = df2.copy()
```

.


```python
df3.head()
df3.describe()
df3.info()
df3.isna().sum().sum()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7983 entries, 0 to 9999
    Data columns (total 11 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Insurance               7983 non-null   object 
     1   Policy Holders          7983 non-null   int32  
     2   Premium Amounts         7983 non-null   float64
     3   Claim Amounts           7983 non-null   float64
     4   Premium Type            7983 non-null   object 
     5   Insurance Year          7983 non-null   int32  
     6   Casualty Year           7983 non-null   int32  
     7   Casualty Month          7983 non-null   object 
     8   Total Vehicles Crashed  7983 non-null   int32  
     9   Sex                     7983 non-null   object 
     10  Age                     7983 non-null   int32  
    dtypes: float64(2), int32(5), object(4)
    memory usage: 592.5+ KB
    




    0



#### Step 2: Exploratory Data Analysis (EDA):


```python

```

Key Players:
- List the major companies operating in the India Two Wheeler Insurance Market.

- Mention some of the key players, such as The New India Assurance', 'HDFC ERGO General Insurance','The Oriental Insurance Company','TATA AIG General Insurance','ICICI Lombard General Insurance'.


```python
top_insurance_providers = df3.groupby('Insurance')[['Policy Holders','Premium Amounts','Claim Amounts']].sum().sort_values(by='Policy Holders', ascending=False)
top_insurance_providers
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
      <th>Policy Holders</th>
      <th>Premium Amounts</th>
      <th>Claim Amounts</th>
    </tr>
    <tr>
      <th>Insurance</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>The New India Assurance</th>
      <td>4989641</td>
      <td>57175793.0</td>
      <td>98204764.0</td>
    </tr>
    <tr>
      <th>TATA AIG General Insurance</th>
      <td>4912271</td>
      <td>56827027.0</td>
      <td>98563984.0</td>
    </tr>
    <tr>
      <th>HDFC ERGO General Insurance</th>
      <td>4787139</td>
      <td>56425592.0</td>
      <td>95367921.0</td>
    </tr>
    <tr>
      <th>ICICI Lombard General Insurance</th>
      <td>4746955</td>
      <td>54567671.0</td>
      <td>96020977.0</td>
    </tr>
    <tr>
      <th>The Oriental Insurance Company</th>
      <td>4635722</td>
      <td>54313737.0</td>
      <td>93285465.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df4 = df3.copy()
df4["Region"]=np.random.choice(['West Region', 'East Region', 'North Region', 'South Region'], size=length)
df3['Region']=df4['Region']
df2['Region']=df4['Region']
```


```python
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
# Visualize the distributions of variables
sns.histplot(df3['Premium Amounts'])
plt.show()
```


    
![png](output_40_0.png)
    



```python
# Explore relationships between variables
sns.scatterplot(x='Age', y='Claim Amounts', data=df3)
plt.show()
```


    
![png](output_41_0.png)
    



```python
# Calculate summary statistics
mean_premium = df3['Premium Amounts'].mean()
print("Mean Premium Amount:", mean_premium)
```

    Mean Premium Amount: 34988.077163973445
    


```python

```

#### Step 3: Descriptive Analytics:


```python
# Top insurance companies based on policy holders
top_companies = df3.groupby('Insurance')['Policy Holders'].sum().sort_values(ascending=False)
print("top_insurance_companies:")
print(top_companies)
```

    top_insurance_companies:
    Insurance
    The New India Assurance            4989641
    TATA AIG General Insurance         4912271
    HDFC ERGO General Insurance        4787139
    ICICI Lombard General Insurance    4746955
    The Oriental Insurance Company     4635722
    Name: Policy Holders, dtype: int32
    

Top insurance providers based on total policy holders-->


```python
# Distribution of premium types
premium_type_counts = df3['Premium Type'].value_counts()
print("Premium_Type_Distribution:")
print(premium_type_counts)
```

    Premium_Type_Distribution:
    Commercial Insurance Premium    4000
    Personal Insurance Premium      3983
    Name: Premium Type, dtype: int64
    


```python
# Distribution of casualties across years and months
casualties_by_year = df3.groupby('Casualty Year')['Total Vehicles Crashed'].sum()
casualties_by_month = df3.groupby('Casualty Month')['Total Vehicles Crashed'].sum()
casualties_by_month
casualties_by_year
```




    Casualty Year
    2015    8970
    2016    8614
    2017    8896
    2018    8902
    2019    8835
    Name: Total Vehicles Crashed, dtype: int32



### CAGR = Compond Annual Growth Rate


```python
start = 2014
ending = 2018
number_of_years = 5
```


```python
cagr = (ending/start)**(1/number_of_years)-1
print("CAGR",cagr)
```

    CAGR 0.00039690427267458084
    


```python

```


```python
df2.groupby('Insurance Year').agg({'Policy Holders':'sum','Premium Amounts':'sum', 'Claim Amounts':'sum'})
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
      <th>Policy Holders</th>
      <th>Premium Amounts</th>
      <th>Claim Amounts</th>
    </tr>
    <tr>
      <th>Insurance Year</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014</th>
      <td>4835848</td>
      <td>56672020.0</td>
      <td>97212968.0</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>4733289</td>
      <td>53785000.0</td>
      <td>92750067.0</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>4895915</td>
      <td>56545812.0</td>
      <td>98020471.0</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>4840174</td>
      <td>56335592.0</td>
      <td>96413404.0</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>4766502</td>
      <td>55971396.0</td>
      <td>97046201.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2["Vehicle Type"] = df2["Vehicle Type"].replace({'Used Two Wheeler':'Old','New Two Wheeler':'New'})
```



dxmax(): The idxmax() function returns the index label corresponding to the maximum value in a Series or DataFrame column. It is used to find the label/index of the maximum value in a column.

nlargest(): The nlargest() function is used to retrieve the top N largest values from a Series or DataFrame column. It returns a new Series or DataFrame containing the N largest values, sorted in descending order. By default, it returns the largest values, but you can specify the keep parameter to control the behavior.

max_value = df['A'].max()

max_index = df['A'].idxmax()

top_3_largest = df['A'].nlargest(3)




```python
insurance_providers=df3.groupby('Insurance')['Policy Holders'].sum()
insurance_providers

top_insurance_provider = insurance_providers.nlargest(3)
top_insurance_provider
```




    Insurance
    The New India Assurance        4989641
    TATA AIG General Insurance     4912271
    HDFC ERGO General Insurance    4787139
    Name: Policy Holders, dtype: int32




```python
top_insurance_provider = insurance_providers.idxmax(0)
top_insurance_provider
```




    'The New India Assurance'




```python
top_claim_amounts = df3.groupby('Insurance')['Claim Amounts'].sum().sort_values(ascending=False)
top_claim_amounts
```




    Insurance
    TATA AIG General Insurance         98563984.0
    The New India Assurance            98204764.0
    ICICI Lombard General Insurance    96020977.0
    HDFC ERGO General Insurance        95367921.0
    The Oriental Insurance Company     93285465.0
    Name: Claim Amounts, dtype: float64



TATA AIG General Insurance has more claim amounts than The New India Assurance.
However, The New India Assurance has the highest number of policy holders.
The New India Assurance also has the highest premium amounts.
In summary, while TATA AIG General Insurance may have more claim amounts, The New India Assurance has a larger customer base with the highest number of policy holders and premium amounts.


```python
regions_total = df3.groupby(["Region"])[['Policy Holders', 'Premium Amounts', 'Claim Amounts']].sum().sort_values(by="Premium Amounts", ascending=False)
insurance_type_total = df3['Premium Type'].value_counts()
top_year = df3.groupby('Insurance Year')[['Premium Amounts', 'Policy Holders', 'Claim Amounts']].sum().sort_values(by="Premium Amounts",ascending=False)
top_insurance = df3.groupby('Insurance')['Policy Holders'].sum().nlargest(3).sort_values(ascending=False)

print(regions_total)
print()
print(insurance_type_total)
print()
print(top_year)
print()

print(top_insurance)
```

                  Policy Holders  Premium Amounts  Claim Amounts
    Region                                                      
    North Region         6300608       72840034.0    126175607.0
    South Region         5980896       69068567.0    119251724.0
    West Region          5959525       69027187.0    119168806.0
    East Region          5830699       68374032.0    116846974.0
    
    Commercial Insurance Premium    4000
    Personal Insurance Premium      3983
    Name: Premium Type, dtype: int64
    
                    Premium Amounts  Policy Holders  Claim Amounts
    Insurance Year                                                
    2014                 56672020.0         4835848     97212968.0
    2016                 56545812.0         4895915     98020471.0
    2017                 56335592.0         4840174     96413404.0
    2018                 55971396.0         4766502     97046201.0
    2015                 53785000.0         4733289     92750067.0
    
    Insurance
    The New India Assurance        4989641
    TATA AIG General Insurance     4912271
    HDFC ERGO General Insurance    4787139
    Name: Policy Holders, dtype: int32
    


```python

```


```python
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
df3.columns
```




    Index(['Insurance', 'Policy Holders', 'Premium Amounts', 'Claim Amounts',
           'Premium Type', 'Insurance Year', 'Casualty Year', 'Casualty Month',
           'Total Vehicles Crashed', 'Sex', 'Age'],
          dtype='object')




```python
# Visualize the distributions of variables
sns.histplot(pp2)
plt.show()
```


    
![png](output_66_0.png)
    



```python
# Visualize the distribution of variables

```


```python
df3.columns
```




    Index(['Insurance', 'Policy Holders', 'Premium Amounts', 'Claim Amounts',
           'Premium Type', 'Insurance Year', 'Casualty Year', 'Casualty Month',
           'Total Vehicles Crashed', 'Sex', 'Age', 'Region'],
          dtype='object')




```python
pp = df3.groupby('Casualty Year')[['Claim Amounts','Premium Amounts']].sum()
pp2 = df3.groupby(['Casualty Year','Casualty Month'])['Total Vehicles Crashed'].sum()
pp2
```




    Casualty Year  Casualty Month
    2015           April             848
                   August            727
                   December          625
                   February          784
                   January           694
                   July              708
                   June              765
                   March             591
                   May               777
                   November          792
                   October           841
                   September         818
    2016           April             614
                   August            740
                   December          746
                   February          693
                   January           801
                   July              671
                   June              779
                   March             758
                   May               646
                   November          683
                   October           729
                   September         754
    2017           April             706
                   August            756
                   December          653
                   February          763
                   January           756
                   July              636
                   June              833
                   March             737
                   May               758
                   November          880
                   October           672
                   September         746
    2018           April             842
                   August            814
                   December          784
                   February          798
                   January           720
                   July              715
                   June              725
                   March             783
                   May               570
                   November          842
                   October           702
                   September         607
    2019           April             707
                   August            778
                   December          690
                   February          756
                   January           807
                   July              687
                   June              746
                   March             726
                   May               756
                   November          679
                   October           758
                   September         745
    Name: Total Vehicles Crashed, dtype: int32




```python
# Explore relationships between variables
sns.scatterplot(x='Policy Holders', y='Claim Amounts', data=data)
plt.show()

# Calculate summary statistics
mean_premium = data['Premium Amounts'].mean()
print("Mean Premium Amount:", mean_premium)
```


    
![png](output_70_0.png)
    


    Mean Premium Amount: nan
    


```python
casualityclaim=df3.groupby(['Casualty Year','Sex'])['Claim Amounts'].sum().nlargest(1)
casualityclaim
```




    Casualty Year  Sex   
    2017           Female    50479848.0
    Name: Claim Amounts, dtype: float64




```python
sns.regplot(x='Total Vehicles Crashed', y='Claim Amounts', data=df3)
plt.ylim(0,)
```




    (0.0, 103998.8)




    
![png](output_72_1.png)
    


Correlation


```python
from scipy import stats

```


```python

pearson_coef, p_value = stats.pearsonr(df3['Total Vehicles Crashed'],df3['Policy Holders'])
pearson_coef, p_value

```




    (0.018945690082545645, 0.09052411784587897)



Correlation Causality: Correlation does not imply causation. Even if a strong correlation is observed between two variables, it does not necessarily mean that one variable causes the other. Establishing causality requires further investigation and experimental design.

The calculated Pearson correlation coefficient (pearson_coef) between the 'Insurance' column and the 'Policy Holders' column is approximately 0.019. The corresponding p-value (p_value) is approximately 0.091.

The Pearson correlation coefficient measures the linear relationship between two variables. In this case, the coefficient of 0.019 suggests a very weak positive correlation between 'Insurance' and 'Policy Holders'. The p-value of 0.091 indicates that there is no strong evidence to reject the null hypothesis of no correlation.


```python

```

## Talking about this dataset.
Generalizability: Dummy datasets are typically created for illustrative purposes and may not represent the diversity and variability present in real-world data. The findings and conclusions drawn from the dummy dataset may not be directly applicable to real-world situations.

Assumptions and limitations: The dummy dataset may be based on certain assumptions or simplifications, which may introduce biases or limitations. It is important to be aware of these assumptions and consider their impact on the analysis and interpretation of the results.

Data reliability: As the dataset is dummy, the accuracy and reliability of the data may be questionable. The values and relationships between variables may not accurately represent real-world scenarios.

Lack of causality: The dataset may show correlations between variables, but it does not establish causality. The relationships observed in the data may be coincidental or influenced by unmeasured factors. Further research and analysis would be needed to establish causal relationships.Thankyou,
Parikshit Carpenter