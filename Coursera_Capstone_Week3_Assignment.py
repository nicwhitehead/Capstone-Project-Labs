#!/usr/bin/env python
# coding: utf-8

# # Notebook for use on the Data Science Capstone Project

# ## Week 3 Assignment: Segmenting and Clustering Neighborhoods in Toronto

# ##### Importing Toronto Suburbs Dataset from Wikipedia - using beautifulsoup

# In[1]:


#install beautiful soup, geopy and folium libraries
get_ipython().system('pip install bs4')
get_ipython().system('pip install geopy')
get_ipython().system('conda install -c conda-forge geopy --yes')
get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')


# In[2]:


import pandas as pd
import requests
from bs4 import BeautifulSoup


# #### Create variable (soup) to hold html data

# In[4]:


T_url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
data  = requests.get(T_url).text


# In[5]:


soup = BeautifulSoup(data, 'html5lib')


# In[6]:


#View the html structure for wikipedia page -- only used while testing code
#soup


# #### Scraping data from wikipedia page - Appending to pandas dataframe

# In[6]:


table_contents=[]
table = soup.find('table')
for row in table.findAll('td'):
    cell = {}
    if row.span.text=='Not assigned':
        pass
    else:
        cell['PostalCode'] = row.p.text[:3]
        cell['Borough'] = (row.span.text).split('(')[0]
        cell['Neighborhood'] = (((((row.span.text).split('(')[1]).strip(')')).replace(' /',',')).replace(')',' ')).strip(' ')
        table_contents.append(cell)

# print(table_contents)
df=pd.DataFrame(table_contents)
df['Borough']=df['Borough'].replace({'Downtown TorontoStn A PO Boxes25 The Esplanade':'Downtown Toronto Stn A',
                                             'East TorontoBusiness reply mail Processing Centre969 Eastern':'East Toronto Business',
                                             'EtobicokeNorthwest':'Etobicoke Northwest','East YorkEast Toronto':'East York/East Toronto',
                                             'MississaugaCanada Post Gateway Processing Centre':'Mississauga'})

#sort ascending by PostalCode
df.sort_values(by=['PostalCode'])


# The dataframe shape is:

# In[7]:


df.shape


# In[8]:


print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(df['Borough'].unique()),
        df.shape[0]))


# ### Retreive Lat & Long data using Geospatial_Coordinates.csv

# In[9]:


df_LL = pd.read_csv (r'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/labs_v1/Geospatial_Coordinates.csv')
print (df_LL)

#https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/labs_v1/Geospatial_Coordinates.csv


# ### Rename Postal Code col.  Merge data on PostalCode field

# In[10]:


df_LL.rename(columns={'Postal Code': 'PostalCode'}, inplace=True)
neighborhoods = pd.merge(df, df_LL, on='PostalCode', how='inner')
neighborhoods


# ## Import libraries to complete the 'Segmenting and Clustering Neighborhoods in Toronto' exercise

# In[16]:


#!pip install geopy


# In[11]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library

print('Libraries imported.')


# ### Identify Longitude & Latitude of Toronto

# In[12]:


address = 'Toronto, Ontario'

geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# ### Create a map of Toronto with neighborhoods superimposed

# In[13]:


# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(neighborhoods['Latitude'], neighborhoods['Longitude'], neighborhoods['Borough'], neighborhoods['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# Identify the Borough with the most postcodes.  Use this as the set for further analysis

# In[14]:


neighborhoods['Borough'].value_counts()


# North York chosen for Analysis.  Co-ordinates for North York are:

# In[15]:


northyork_data = neighborhoods[neighborhoods['Borough'] == 'North York'].reset_index(drop=True)
northyork_data.head()


# In[16]:


address = 'North York, Toronto'

geolocator = Nominatim(user_agent="Toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinates of North York are {}, {}.'.format(latitude, longitude))


# ### Create map of North York using latitude and longitude values

# In[17]:


map_NorthYork = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, label in zip(northyork_data['Latitude'], northyork_data['Longitude'], northyork_data['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_NorthYork)  
    
map_NorthYork


# ## Define Foursquare Credentials and Version

# In[18]:


CLIENT_ID = 'P2MGPUXJG1RUBIVCB2OUICQTJ5WFL0QFN0EGDKE20ESPN1O2' # your Foursquare ID
CLIENT_SECRET = 'WCHHX33L0PD5QQERPYEUURZIITH4MAQE3ZXRS4O5AAWM3YSA' # your Foursquare Secret
VERSION = '20180604' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# ### Top 50 venues that are within a 1000 meter radius of chosen neighborhood

# First entry in the dataframe

# In[19]:


northyork_data.loc[0, 'Neighborhood']


# Find co-ordinates of Parkwoods

# In[20]:


neighborhood_latitude = northyork_data.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = northyork_data.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = northyork_data.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# ### Top 50 Venues within 1000 meters

# In[21]:


#create the GET request URL
LIMIT = 50
radius = 1000
CLIENT_ID = 'P2MGPUXJG1RUBIVCB2OUICQTJ5WFL0QFN0EGDKE20ESPN1O2' # your Foursquare ID
CLIENT_SECRET = 'WCHHX33L0PD5QQERPYEUURZIITH4MAQE3ZXRS4O5AAWM3YSA' # your Foursquare Secret
ACCESS_TOKEN = 'MH3GX20WHR1YYNVGXTNCHXHO2SZFQ3UOFKP5SQ3WMHNR1AIF' # your FourSquare Access Token
VERSION = '20180604'
url='https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
        CLIENT_ID,
        CLIENT_SECRET,
        VERSION,
        neighborhood_latitude, 
        neighborhood_longitude, 
        radius, 
        LIMIT)

url


# In[22]:


results = requests.get(url).json()


# In[23]:


items = results['response']['groups'][0]['items']
items[0]


# In[24]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# clean the json and structure it into a pandas dataframe

# In[25]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues = nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head(10)


# In[26]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# ## Explore Neighborhoods in Toronto

# #### Create a function to repeat the same process on all the neighborhoods in North York

# In[27]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[29]:


# type your answer here
NorthYork_venues = getNearbyVenues(names=northyork_data['Neighborhood'],
                                   latitudes=northyork_data['Latitude'],
                                   longitudes=northyork_data['Longitude']
                                  )


# #### Find the size of the results table

# In[31]:


print(NorthYork_venues.shape)
NorthYork_venues.head()


# #### Check how many venues were returned for each neighborhood

# In[32]:


NorthYork_venues.groupby('Neighborhood').count()


# #### Determine unique venue categories

# In[35]:


print('There are {} unique categories.'.format(len(NorthYork_venues['Venue Category'].unique())))


# ## Analyse each neighbourhood in North York

# In[39]:


# one hot encoding
NorthYork_onehot = pd.get_dummies(NorthYork_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
NorthYork_onehot['Neighborhood'] = NorthYork_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [NorthYork_onehot.columns[-1]] + list(NorthYork_onehot.columns[:-1])
NorthYork_onehot = NorthYork_onehot[fixed_columns]

NorthYork_onehot.head(10)


# In[40]:


NorthYork_onehot.shape


# ### Now group rows by neighborhood and take the mean of the frequency of occurrence of each category

# In[41]:


NorthYork_grouped = NorthYork_onehot.groupby('Neighborhood').mean().reset_index()
NorthYork_grouped


# In[42]:


NorthYork_grouped.shape


# ### Print each neighborhood along with the top 5 most common venues

# In[43]:


num_top_venues = 5

for hood in NorthYork_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = NorthYork_grouped[NorthYork_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# #### Put this into a pandas dataframe

# First, write a function to sort the venues in descending order.

# In[44]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# Create the new dataframe and display the top 10 venues for each neighborhood

# In[45]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = NorthYork_grouped['Neighborhood']

for ind in np.arange(NorthYork_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(NorthYork_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ### Cluster the Neighborhoods

# Run k-means to cluster the neighborhood into 5 clusters.

# In[46]:


# set number of clusters
kclusters = 5

NorthYork_grouped_clustering = NorthYork_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(NorthYork_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# Create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# In[52]:


# add clustering labels

neighborhoods_venues_sorted.drop(['Cluster Labels'], axis=1, inplace=True)

neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

NorthYork_merged = northyork_data

# merge NorthYork_grouped with NorthYork_data to add latitude/longitude for each neighborhood
NorthYork_merged = NorthYork_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

NorthYork_merged.head() # check the last columns!


# Drop all the NaN values to prevent data skew

# In[56]:


NorthYork_merged_nonan = NorthYork_merged.dropna(subset=['Cluster Labels'])


# Plot the clusters on the map

# In[57]:


import matplotlib.cm as cm
import matplotlib.colors as colors


# In[63]:


map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(NorthYork_merged_nonan['Latitude'], NorthYork_merged_nonan['Longitude'], NorthYork_merged_nonan['Neighborhood'], NorthYork_merged_nonan['Cluster Labels']):
    label = folium.Popup('Cluster ' + str(int(cluster) +1) + '\n' + str(poi) , parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster-1)],
        fill=True,
        fill_color=rainbow[int(cluster-1)]
        ).add_to(map_clusters)
        
map_clusters


# ## Examine Clusters

# ### Cluster 1

# In[69]:


NorthYork_merged_nonan.loc[NorthYork_merged_nonan['Cluster Labels'] == 0, NorthYork_merged_nonan.columns[[2] + list(range(5, NorthYork_merged_nonan.shape[1]))]]


# ### Cluster 2

# In[70]:


NorthYork_merged_nonan.loc[NorthYork_merged_nonan['Cluster Labels'] == 1, NorthYork_merged_nonan.columns[[2] + list(range(5, NorthYork_merged_nonan.shape[1]))]]


# ### Cluster 3

# In[71]:


NorthYork_merged_nonan.loc[NorthYork_merged_nonan['Cluster Labels'] == 2, NorthYork_merged_nonan.columns[[2] + list(range(5, NorthYork_merged_nonan.shape[1]))]]


# ### Cluster 4

# In[72]:


NorthYork_merged_nonan.loc[NorthYork_merged_nonan['Cluster Labels'] == 3, NorthYork_merged_nonan.columns[[2] + list(range(5, NorthYork_merged_nonan.shape[1]))]]


# ### Cluster 5

# In[73]:


NorthYork_merged_nonan.loc[NorthYork_merged_nonan['Cluster Labels'] == 4, NorthYork_merged_nonan.columns[[2] + list(range(5, NorthYork_merged_nonan.shape[1]))]]


# In[ ]:




