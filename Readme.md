
# Analyzing the ACLED data with Spark

*Luis Carlos Cruz*

*May 2016*

ACLED (Armed Conflict Location and Event Data Project) is designed for disaggregated conflict analysis and crisis mapping. This dataset codes the dates and locations of all reported political violence and protest events in over 60 developing countries in Africa and Asia. Political violence and protest includes events that occur within civil wars and periods of instability, public protest and regime breakdown. The project covers all African countries from 1997 to the present, and South and South-East Asia in real-time. For more information visit the [ACLED website](http://www.acleddata.com/).

This work is solely based on the african data, covering a time lapse between the years 1997 and 2015. The purpose of the work is to extract relevant information from the dataset using the basic features of Spark.


```python
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import pyspark
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql.functions import date_format, col, udf, lit, array_contains

```

    /opt/conda/lib/python3.5/site-packages/matplotlib/font_manager.py:273: UserWarning: Matplotlib is building the font cache using fc-list. This may take a moment.
      warnings.warn('Matplotlib is building the font cache using fc-list. This may take a moment.')
    /opt/conda/lib/python3.5/site-packages/matplotlib/font_manager.py:273: UserWarning: Matplotlib is building the font cache using fc-list. This may take a moment.
      warnings.warn('Matplotlib is building the font cache using fc-list. This may take a moment.')


Create the Spark and SQL contexts.


```python
sc = pyspark.SparkContext('local[*]')
sqlCtx = SQLContext(sc)
```

Load the file with the data. This version contains less columns than the original dataset, for I removed unnecesary for this analysis, like coordinates, notes and source.


```python
africa_rdd = sc.textFile('africa2.csv')
```

For every line, split the columns and map the lines with the Row. Then convert the RDD into a DataFrame.


```python
Event = Row('event_date', 'year', 'event_type', 'actor1', 'inter1', 'actor2', 'inter2', 'country', 'location', 
            'fatalities')

def getEvent(line):
    cells = line.split(',')
    return Event(*cells)

africa = africa_rdd.map(getEvent)
africa_df = africa.toDF()
```

Convert the columns into the proper type and remove NAs. 


```python
africa_df = africa_df.select('*', africa_df.fatalities.cast('integer').alias('fatalities_i'), 
                             africa_df.year.cast('integer').alias('year_i'))
africa_df = africa_df.dropna()
africa_df = africa_df.select('*', date_format(africa_df.event_date,'\"%d/%m/%Y\"').alias('date'))
```

The resulting data frame contains the following attributes:
- event_date: date of the event with the format dd/mm/yyyy
- year: the year in which the event took place
- event_type: a description of the type of the event that could be one of the following values
    * Battle-No change of territory
    * Battle-Non-state actor overtakes territory
    * Battle-Government regains territory
    * Headquarters or base established
    * Strategic Development
    * Riots/Protests
    * Violence against civilians
    * Non-violent transfer of territory
    * Remote violence
- actor1: actor involved in the conflict
- inter1: type of actor1
- actor2: actor involved in the conflict
- inter2: type of actor2
- country: the country in which the event took place
- location: the location in which the event took place
- fatalities: the numer or estimate of fatalities


```python
africa_df.printSchema()
```

    root
     |-- event_date: string (nullable = true)
     |-- year: string (nullable = true)
     |-- event_type: string (nullable = true)
     |-- actor1: string (nullable = true)
     |-- inter1: string (nullable = true)
     |-- actor2: string (nullable = true)
     |-- inter2: string (nullable = true)
     |-- country: string (nullable = true)
     |-- location: string (nullable = true)
     |-- fatalities: string (nullable = true)
     |-- fatalities_i: integer (nullable = true)
     |-- year_i: integer (nullable = true)
     |-- date: string (nullable = true)
    


## Exploratory analysis

This section shows an exploratory analysis to the dataset. First I register a temporary table with the dataframe created in the previous section.


```python
africa_df.registerTempTable("africa")
```

This query gets the top 10 of countries according to the number of conflicts. Somalia shows up as the most troubled country, probably a reflection of the ongoing [Somali Civil War](https://en.wikipedia.org/wiki/Somali_Civil_War).


```python
sqlCtx.sql('SELECT country, COUNT(*) AS count \
           FROM africa GROUP BY country ORDER BY count \
           DESC LIMIT 10').toPandas()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>"Somalia"</td>
      <td>17454</td>
    </tr>
    <tr>
      <th>1</th>
      <td>"Democratic Republic of Congo"</td>
      <td>9825</td>
    </tr>
    <tr>
      <th>2</th>
      <td>"Nigeria"</td>
      <td>8541</td>
    </tr>
    <tr>
      <th>3</th>
      <td>"Sudan"</td>
      <td>8227</td>
    </tr>
    <tr>
      <th>4</th>
      <td>"South Africa"</td>
      <td>7421</td>
    </tr>
    <tr>
      <th>5</th>
      <td>"Egypt"</td>
      <td>6917</td>
    </tr>
    <tr>
      <th>6</th>
      <td>"Zimbabwe"</td>
      <td>5363</td>
    </tr>
    <tr>
      <th>7</th>
      <td>"Kenya"</td>
      <td>5041</td>
    </tr>
    <tr>
      <th>8</th>
      <td>"Uganda"</td>
      <td>4611</td>
    </tr>
    <tr>
      <th>9</th>
      <td>"Sierra Leone"</td>
      <td>4598</td>
    </tr>
  </tbody>
</table>
</div>



Every event has an type that describes the nature of the conflict. The following query counts the number of incidences of every type of event, showing that the most common types are battles without change of territory and violence against civilians.

This query was created using the API, and hereinafter all the queries are created this way.


```python
africa_df.select("country", "event_type", "year")\
        .groupBy('event_type')\
        .agg({"event_type":"count"})\
        .orderBy('count(event_type)',ascending=False)\
        .limit(10)\
        .toPandas()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event_type</th>
      <th>count(event_type)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>"Battle-No change of territory"</td>
      <td>34137</td>
    </tr>
    <tr>
      <th>1</th>
      <td>"Violence against civilians"</td>
      <td>33393</td>
    </tr>
    <tr>
      <th>2</th>
      <td>"Riots/Protests"</td>
      <td>29640</td>
    </tr>
    <tr>
      <th>3</th>
      <td>"Strategic development"</td>
      <td>7217</td>
    </tr>
    <tr>
      <th>4</th>
      <td>"Remote violence"</td>
      <td>6812</td>
    </tr>
    <tr>
      <th>5</th>
      <td>"Non-violent transfer of territory"</td>
      <td>2150</td>
    </tr>
    <tr>
      <th>6</th>
      <td>"Battle-Government regains territory"</td>
      <td>1891</td>
    </tr>
    <tr>
      <th>7</th>
      <td>"Battle-Non-state actor overtakes territory"</td>
      <td>1803</td>
    </tr>
    <tr>
      <th>8</th>
      <td>"Headquarters or base established"</td>
      <td>733</td>
    </tr>
  </tbody>
</table>
</div>



The following query shows the top 10 of factions that take part in conflicts. Armed groups and military forces shows up in the top 10, but also protesters and rioters.


```python
africa_df.select('actor1','inter1')\
            .groupBy('actor1')\
            .agg({'actor1':'count'})\
            .orderBy('count(actor1)', ascending=False)\
            .limit(10)\
            .toPandas()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actor1</th>
      <th>count(actor1)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>"Unidentified Armed Group (Somalia)"</td>
      <td>4543</td>
    </tr>
    <tr>
      <th>1</th>
      <td>"Protesters (South Africa)"</td>
      <td>3428</td>
    </tr>
    <tr>
      <th>2</th>
      <td>"Military Forces of Somalia (2012-)"</td>
      <td>3004</td>
    </tr>
    <tr>
      <th>3</th>
      <td>"Rioters (South Africa)"</td>
      <td>2487</td>
    </tr>
    <tr>
      <th>4</th>
      <td>"ZANU-PF: Zimbabwe African National Union-Patr...</td>
      <td>2460</td>
    </tr>
    <tr>
      <th>5</th>
      <td>"Protesters (Egypt)"</td>
      <td>2449</td>
    </tr>
    <tr>
      <th>6</th>
      <td>"Al Shabaab"</td>
      <td>2385</td>
    </tr>
    <tr>
      <th>7</th>
      <td>"Military Forces of Angola (1975-)"</td>
      <td>2347</td>
    </tr>
    <tr>
      <th>8</th>
      <td>"Military Forces of Democratic Republic of Con...</td>
      <td>2084</td>
    </tr>
    <tr>
      <th>9</th>
      <td>"LRA: Lord's Resistance Army"</td>
      <td>1927</td>
    </tr>
  </tbody>
</table>
</div>



Next, I show the top 10 of most conflicting cities according to the number of events registered per location. For that, I create a dataframe with the registry of cities and countries they belong to (first query). The second query gets the locations aggregated by the count of events, resulting in a second dataframe. Finally, I join both datasets by the 'location' column. The result shows the top 10 of conflicting locations with the country.


```python
locations_df = sqlCtx.sql('SELECT country, location FROM africa GROUP BY location, country')

top_conflicting_locations = africa_df.select('location')\
                                    .groupBy('location')\
                                    .agg({'location':'count'})\
                                    .orderBy('count(location)', ascending=False)\
                                    .limit(10)

joined_df = top_conflicting_locations.join(locations_df, top_conflicting_locations.location == locations_df.location, 'inner')\
                                    .orderBy('count(location)', ascending=False)\
                                    .drop(top_conflicting_locations.location)
joined_df.toPandas()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count(location)</th>
      <th>country</th>
      <th>location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1218</td>
      <td>"Somalia"</td>
      <td>"Mogadishu"</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1080</td>
      <td>"Zimbabwe"</td>
      <td>"Harare"</td>
    </tr>
    <tr>
      <th>2</th>
      <td>892</td>
      <td>"Central African Republic"</td>
      <td>"Bangui"</td>
    </tr>
    <tr>
      <th>3</th>
      <td>730</td>
      <td>"Somalia"</td>
      <td>"Hodan"</td>
    </tr>
    <tr>
      <th>4</th>
      <td>696</td>
      <td>"Libya"</td>
      <td>"Tripoli"</td>
    </tr>
    <tr>
      <th>5</th>
      <td>678</td>
      <td>"Ivory Coast"</td>
      <td>"Abidjan"</td>
    </tr>
    <tr>
      <th>6</th>
      <td>676</td>
      <td>"South Africa"</td>
      <td>"Cape Town"</td>
    </tr>
    <tr>
      <th>7</th>
      <td>676</td>
      <td>"Egypt"</td>
      <td>"Cairo"</td>
    </tr>
    <tr>
      <th>8</th>
      <td>652</td>
      <td>"Libya"</td>
      <td>"Benghazi"</td>
    </tr>
    <tr>
      <th>9</th>
      <td>640</td>
      <td>"Somalia"</td>
      <td>"Kismayo"</td>
    </tr>
  </tbody>
</table>
</div>



## Libya at war

In the past 5 years Libya has faced several armed conflicts, and in this analysis I show the evidence present in the data. I start by creating a dataframe with only events registered in Libya.


```python
libya_df = africa_df.select('*')\
                    .where("country='\"Libya\"'")
```

Aggregating this dataset by the number of conflicts per year shows a clear change between the years 2010 and 2011. In 2011 the [First Libyan Civil War](https://en.wikipedia.org/wiki/Libyan_civil_war_2011) started, leading to a phase of turmoil known as the [Post-civil war violence](https://en.wikipedia.org/wiki/Factional_violence_in_Libya_%282011%E2%80%9314%29) and the still ongoing [Second Libyan Civil War](https://en.wikipedia.org/wiki/Libyan_civil_war_%282014%E2%80%93present%29). This events increased the number of conflicts dramatically, as can be seen in the plot below.


```python
df = libya_df.groupBy('year')\
                .agg({'year':'count'})\
                .toPandas()
labels = [str(i) for i in df['year']]
x = [float(i) for i in df['year']]
fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot(111)
ax.set_xticks(np.arange(1997,2016))
ax.set_xticklabels(labels)
ax.bar(x,df['count(year)'], align='center')
ax.set_title('Lybian conflicts per year')
ax.set_xlabel('Year')
ax.set_ylabel('Number of conflicts')

```




    <matplotlib.text.Text at 0x7f8f916aada0>




![png](output_29_1.png)


The next pieces of code are an analysis of the conflicts in 2015. The query aggregates the data per day and counts the number of fatalities. The total of fatalities in 2015 was of 2705, and the most deadly day was January 3, with a death toll of 101 due to 7 conflicts.


```python
libya_fatalities_per_day = libya_df.select('year','event_date','fatalities_i')\
                                    .where("year=2015")\
                                    .groupBy('event_date')\
                                    .agg({'fatalities_i':'sum','event_date':'count'})\
                                    .orderBy('sum(fatalities_i)',ascending=False)
```


```python
df = libya_fatalities_per_day.toPandas() 
df['event_date'] = np.array([datetime.strptime(x, '\"%d/%m/%Y\"') for x in df['event_date']])
df['sum(fatalities_i)'] = [int(x) for x in df['sum(fatalities_i)']]
df = df.sort(['event_date'],ascending=[1])

fatalities_2015 = sum(df['sum(fatalities_i)'])

print('Fatalities in 2015: ' + str(fatalities_2015))
print('Most deadly day: ', libya_fatalities_per_day.first())
```

    Fatalities in 2015: 2705
    Most deadly day:  Row(event_date='"03/01/2015"', count(event_date)=7, sum(fatalities_i)=101)



```python
fig = plt.figure(figsize=(18,4))
ax = fig.add_subplot(111)
ax.plot_date(df['event_date'],df['sum(fatalities_i)'], fmt='r-', label = "Number of fatalities")
ax.set_title('Fatalities per day')
ax.set_xlabel('Date')
ax.set_ylabel('Number of fatalities')
```




    <matplotlib.text.Text at 0x7f8fac3f1da0>




![png](output_33_1.png)



```python

```
