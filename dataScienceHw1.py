"""^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Exploratory data analysis (EDA) plays a very important role in data science projects. It helps us understand the data and explore the hidden relations between variables. Moreover, it helps us in the selection of the appropriate statistical and machine learning tools and techniques. In this problem we are going to perform an exploratory data analysis on 120 years of Olympics data. (50%)

a) Download the data from this Kaggle data repository and load the athlete_event.csv file into a pandas DataFrame called olympics120. Filter the DataFrame so that it only contains data about the Summer season. (Hint: you can use the pandas query() function).
VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

olympics120 = pd.read_csv('../Data/athlete_events.csv', sep=',')
olympics120.query('Season == \'Summer\'')

"""^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
b) Retrieve some basic descriptive

    List item
    List item

statistics about the dataset using the .describe() method of pandas DataFrames. Now, try to guess which sports have the shortest, tallest (height column), heaviest, lightest (weight column), youngest and oldest (age column) athletes in the Olympics. Then, check your guess based on the data! Print out which sport has the tallest, shortest, heaviest, lightest, youngest and oldest athletes.
VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"""

olympics120.describe()
olympics120['Sport'].unique()
''' Hunch
Shortest: Table Tennis
Tallest: Basketball
Heaviest: Football/Weightlifting
Lightest: Cross Country
Youngest: Very close call. Maybe gymnastics
Oldest: Art Competitions/Croquet
'''

# Sport, height, age, weight data for each sport
print('The sports with the top/bottom height, age, and weight on average are the following. \n' \
      'Hence, I was right in only 3 out of the 6 (namely basketball, artcompetitions, and ' \
      'gymnastics.')
shaw = pd.DataFrame(olympics120, columns = ['Sport','Height','Age','Weight'])
shawGroupByMean = shaw.dropna().groupby(by='Sport').mean()
pd.concat([shawGroupByMean.sort_values('Height', ascending=False).head(1),
           shawGroupByMean.sort_values('Height', ascending=False).tail(1),
           shawGroupByMean.sort_values('Age', ascending=False).head(1),
           shawGroupByMean.sort_values('Age', ascending=False).tail(1),
           shawGroupByMean.sort_values('Weight', ascending=False).head(1),
           shawGroupByMean.sort_values('Weight', ascending=False).tail(1)]) \
               .mask([[False, True, True],
                      [False, True, True],
                      [True, False, True],
                      [True, False, True],
                      [True, True, False],
                      [True, True, False]], np.nan)
                      
# The sports with the top/bottom height, age, and weight on average are the following. 
# Hence, I was right in only 3 out of the 6 (namely basketball, artcompetitions, and gymnastics.

"""^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
c) Consider only the male athletes of the following sports: Basketball, Gymnastics, Wrestling. Make a scatter plot in which the x axis shows the athletes' weights and the y axis shows the athletes' heights. Each sport should be represented by a different color point on the plot. Make one scatter plot like this for the 2012 London Olympics, and another for the 1960 Rome Olympics. (Hints: You can create the scatterplots with the .scatterplot() function of the seaborn package. You can place the scatterplots next to each other with the help of the .subplot() function of the matplotlib.pyplot package. You can fix the scales of the axes with the matplotlib.pyplot.xlim([lower, upper]) function. It is not required to use these hints, but seaborn and matplotlib are good libraries to explore for plotting.) What is the relationship between the heights and weights of athletes in these sports?
VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"""

male_olympics2012 = olympics120.query('Sex == \'M\' and Year == 2012')
male_olympics1960 = olympics120.query('Sex == \'M\' and Year == 1960')
# "Basketball male 2012 olympics"
bm2012o = male_olympics2012.query('Sport == \'Basketball\'')
gm2012o = male_olympics2012.query('Sport == \'Gymnastics\'')
wm2012o = male_olympics2012.query('Sport == \'Wrestling\'')
bm1960o = male_olympics1960.query('Sport == \'Basketball\'')
gm1960o = male_olympics1960.query('Sport == \'Gymnastics\'')
wm1960o = male_olympics1960.query('Sport == \'Wrestling\'')

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))

ax1.scatter(bm1960o['Weight'], bm1960o['Height'], label='Basketball', alpha=.1, marker='o')
ax1.scatter(gm1960o['Weight'], gm1960o['Height'], label='Gymnastics', alpha=.1, marker='o')
ax1.scatter(wm1960o['Weight'], wm1960o['Height'], label='Wrestling', alpha=.1, marker='o')
ax1.set_xlabel('Weight')
ax1.set_ylabel('Height')
ax1.set_title('1960 Male Olympics')
ax1.legend()

ax2.scatter(bm2012o['Weight'], bm2012o['Height'], label='Basketball', alpha=.1, marker='o')
ax2.scatter(gm2012o['Weight'], gm2012o['Height'], label='Gymnastics', alpha=.1, marker='o')
ax2.scatter(wm2012o['Weight'], wm2012o['Height'], label='Wrestling', alpha=.1, marker='o')
ax2.set_xlabel('Weight')
ax2.set_ylabel('Height')
ax2.set_title('2012 Male Olympics')
ax2.legend()

# Answer: The average male height has gone up in the past few years, hence the latter graph is
# slightly higher up. Basketball has the 'biggest players', and gymnastic has the 'smallest' ones,
# whereas wresting height varies some, and weight varies strongly

"""^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
d) We are curious about which countries have the most gold medals per 1 million people.

    Calculate how many gold medals the each country won per year at the Summer Olympics. Hints: You can filter the data with the query() function, keeping only the rows with gold medals. Then you can groupby() the country name and year attributes, and use .apply(lambda x: len(pd.unique(x))) on the Events column:

    gold_medals=DataFramegolds.groupby(['NOC', 'Year']).Event.apply(lambda x: len(pd.unique(x)))

    Describe what the above line of code is doing

    Store this information in a new data frame, something like this:
    	NOC	GOLDMEDALS
    1	ALG	1
    2	ANZ	3
    Load the population data from https://math.bme.hu/~pinterj/BevAdat1/Adatok/OlympicsPopulation.xlsx into a pandas DataFrame without downloading the file, then merge the two tables.
    Add a showing the number of gold metals per one million people. Which countries have the most gold medals per one million people? (Hint: You can use the pandas sort_values() function)
VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"""

pop_data = pd.read_excel('https://math.bme.hu/~pinterj/BevAdat1/Adatok/OlympicsPopulation.xlsx',
                        sheet_name="Sheet1")
pop_data
olympics120.Year.unique()

# My solution
nyg = olympics120[olympics120.Medal == "Gold"].groupby(["NOC", "Year"]).count().loc[:,"Medal"]
nyg

# The given code does the same thing, but the count is altered. Instead of counting gold medals
# that year for that country, it filters that number down by event category that year for that
# country
olympics120[olympics120.Medal == "Gold"].groupby(['NOC', 'Year']).Event.apply(lambda x: len(pd.unique(x)))

nyg2 = nyg.reset_index()
nyg2.columns = ['NOC', 'YEAR', 'GOLDMEDALS']
nyg2 = nyg2[nyg2.YEAR == 2016].iloc[:,[0, 2]]
nyg2
