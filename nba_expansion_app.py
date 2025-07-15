# Import necessary libraries for data analysis and visualization
import geopandas as gpd  # For geospatial data handling (imported but not used in current code)
import pandas as pd  # For data manipulation and analysis using DataFrames
import requests  # For making HTTP requests (imported but not used in current code)
from bs4 import BeautifulSoup  # For web scraping and HTML parsing (imported but not used in current code)
from transformers import pipeline  # For natural language processing tasks (imported but not used in current code)
import matplotlib.pyplot as plt  # For creating plots and visualizations
import matplotlib  # Base matplotlib library (imported but not used directly)
import numpy as np

# ============================================================================
# Population
# ============================================================================
# Create a dataset of all current NBA teams and their locations
nba_teams = pd.DataFrame({
    'team': ['Lakers', 'Clippers', 'Warriors', 'Kings', 'Suns', 'Nuggets', 
             'Trail Blazers', 'Jazz', 'Thunder', 'Mavericks', 'Spurs', 'Rockets',
             'Timberwolves', 'Grizzlies', 'Pelicans', 'Bulls', 'Cavaliers', 
             'Pistons', 'Pacers', 'Bucks', 'Hawks', 'Hornets', 'Heat', 'Magic',
             'Knicks', 'Nets', '76ers', 'Celtics', 'Raptors', 'Wizards'],
    'city': ['Los Angeles', 'Los Angeles', 'San Francisco', 'Sacramento', 'Phoenix',
             'Denver', 'Portland', 'Salt Lake City', 'Oklahoma City', 'Dallas',
             'San Antonio', 'Houston', 'Minneapolis', 'Memphis', 'New Orleans',
             'Chicago', 'Cleveland', 'Detroit', 'Indianapolis', 'Milwaukee',
             'Atlanta', 'Charlotte', 'Miami', 'Orlando', 'New York', 'Brooklyn',
             'Philadelphia', 'Boston', 'Toronto', 'Washington'],
    'latitude': [34.0430, 34.0430, 37.7680, 38.6493, 33.4457, 39.7487,
                 45.5317, 40.7683, 35.4634, 32.7905, 29.4270, 29.7508,
                 44.9633, 35.1381, 29.9490, 41.8807, 41.4993, 42.6970,
                 39.7640, 43.0432, 33.7573, 35.2249, 25.7814, 28.5392,
                 40.7505, 40.6892, 39.9012, 42.3662, 43.6435, 38.8981],
    'longitude': [-118.2673, -118.2673, -122.3892, -121.5177, -112.0712, -105.0178,
                  -122.6664, -111.9011, -97.5151, -96.8104, -98.4698, -95.3621,
                  -93.2683, -90.0505, -90.0821, -87.6742, -81.6944, -83.2459,
                  -86.1556, -87.9073, -84.3963, -80.8394, -80.2089, -81.4232,
                  -73.9934, -73.9442, -75.1720, -71.0275, -79.3791, -77.0209]
})
# Load the complete county dataset from Excel file - don't filter yet!
# Using openpyxl engine to read .xlsx files, skiprows=3 skips the first 3 header rows
df = pd.read_excel('/Users/deadlinefighter/Desktop/co-est2024-pop.xlsx', engine = 'openpyxl', skiprows = 3)

# Clean the data by renaming columns from generic "Unnamed" to descriptive names
# The original Excel file has unnamed columns that pandas auto-names as "Unnamed: X"
df.rename(columns={"Unnamed: 0": "County Name"},inplace=True)  # First column contains county names
df.rename(columns={"Unnamed: 1":"Estimated Base Population Apr 2020"}, inplace=True)  # Second column has base population estimates
df.rename(columns={2020:"Population 2020"}, inplace=True)  # Rename year columns to be more descriptive
df.rename(columns={2021:"Population 2021"}, inplace=True)  # Each year column contains population data for that year
df.rename(columns={2022:"Population 2022"}, inplace=True)  # Making column names consistent and readable
df.rename(columns={2023:"Population 2023"}, inplace=True)  # Using inplace=True modifies the original DataFrame
df.rename(columns={2024:"Population 2024"}, inplace=True)  # Rather than creating a new copy

# Clean county names by removing leading/trailing whitespace and dots
# str.strip() removes whitespace, str.lstrip(".") removes leading dots from county names
df["County Name"] = df['County Name'].str.strip().str.lstrip(".")

# Remove rows with missing data to ensure data quality
# dropna() removes rows where the specified column has NaN/null values
df = df.dropna(subset = ["County Name"])
# Filter out the "United States" row which is a summary row, not a county
df = df[df['County Name'] != "United States"]

# Define list of counties that have NBA teams to exclude from analysis
# This ensures we're only looking at counties without existing NBA franchises
nba_counties_to_exclude = [
    "Los Angeles County, California",      # Lakers & Clippers - two teams in same county
    "Cook County, Illinois",               # Bulls - Chicago's county
    "Harris County, Texas",                # Rockets - Houston's county
    "Dallas County, Texas",                # Mavericks - Dallas's primary county
    "Miami-Dade County, Florida",          # Heat - Miami's county
    "Kings County, New York",              # Nets - Brooklyn's county
    "New York County, New York",           # Knicks - Manhattan's county
    "Maricopa County, Arizona",            # Suns - Phoenix's county
    "Santa Clara County, California",      # Warriors - San Francisco Bay Area
    "Philadelphia County, Pennsylvania",    # 76ers - Philadelphia's county
    "Suffolk County, Massachusetts",       # Celtics - Boston's county
    "Fulton County, Georgia",              # Hawks - Atlanta's county
    "Denver County, Colorado",             # Nuggets - Denver's county
    "Wayne County, Michigan",              # Pistons - Detroit's county
    "Marion County, Indiana",              # Pacers - Indianapolis's county
    "Shelby County, Tennessee",            # Grizzlies - Memphis's county
    "Milwaukee County, Wisconsin",         # Bucks - Milwaukee's county
    "Hennepin County, Minnesota",          # Timberwolves - Minneapolis's county
    "Orleans Parish, Louisiana",           # Pelicans - New Orleans (parishes are Louisiana's counties)
    "Oklahoma County, Oklahoma",           # Thunder - Oklahoma City's county
    "Orange County, Florida",              # Magic - Orlando's county
    "Sacramento County, California",       # Kings - Sacramento's county
    "Bexar County, Texas",                # Spurs - San Antonio's county
    "Salt Lake County, Utah",             # Jazz - Salt Lake City's county
    "District of Columbia, District of Columbia",  # Wizards - Washington DC
    "Mecklenburg County, North Carolina", # Hornets - Charlotte's county
    "Cuyahoga County, Ohio",              # Cavaliers - Cleveland's county
    "Multnomah County, Oregon",           # Trail Blazers - Portland's county
    "Tarrant County, Texas"               # Mavericks also serve Arlington area in this county
]

# Filter out NBA counties using boolean indexing
# ~df['County Name'].isin() creates a boolean mask that's True for counties NOT in the exclusion list
df_available = df[~df['County Name'].isin(nba_counties_to_exclude)].copy()

# Get top 50 most populated counties from available (non-NBA) counties
# nlargest() sorts by specified column and returns top N rows with all their data intact
top_500_population = df_available.nlargest(500, "Population 2024") 

# Display results for user review
print("Top 500 Available Counties by Population:")
# Print only relevant columns for readability - county name and current population
print(top_500_population[["County Name", "Population 2024"]])

# Calculate population growth rate as percentage change from 2020 to 2024
# Formula: ((new_value - old_value) / old_value) * 100 gives percentage change
df_available['Population_Growth_Rate'] = ((df_available['Population 2024'] - df_available["Population 2020"])/df_available['Population 2020']) * 100

# Get top 50 counties by population growth rate (fastest growing)
# This identifies counties with highest percentage growth, not just highest absolute population
top_500_population_growth = df_available.nlargest(500, "Population_Growth_Rate")

# Display growth rate results
print("Top 500 available counties by population growth rate(2020-2024):")
# Show starting population, ending population, and growth rate; round to 2 decimal places for readability
print(top_500_population_growth[["County Name", "Population 2020", "Population 2024", 'Population_Growth_Rate']].round(2))

# ============================================================================
# GDP Analysis
# ============================================================================

file_path = '/Users/deadlinefighter/Desktop/gdpdataset.xlsx'

df = pd.read_excel(file_path, sheet_name = 'Table 1', skiprows=5)
df.columns = df.columns.str.strip()
df = df[[df.columns[0],'Unnamed: 4']]
df.columns = ['Name', 'GDP_2023']

processed_data = []
current_state = None
us_states = {
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
    'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
    'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
    'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
    'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
    'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
    'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
    'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
    'West Virginia', 'Wisconsin', 'Wyoming', 'District of Columbia'
}

for index, row in df.iterrows():
     name = row['Name']
     gdp = row['GDP_2023']

     if name in us_states:
          current_state = name
     else:
          if current_state:
               processed_data.append({'County' : name,
                                 'State' : current_state,
                                 'GDP_2023': gdp})

county_df = pd.DataFrame(processed_data)

county_df['County_State_Name'] = county_df['County'] + ' County' + ', ' + county_df['State']
county_df = county_df[['County_State_Name', 'GDP_2023']]
county_df = county_df.sort_values('GDP_2023', ascending =False)
county_df = county_df.reset_index(drop=True)
county_df = county_df[~county_df['County_State_Name'].isin(nba_counties_to_exclude)]
county_df['Rank'] = county_df.index + 1

top_500_gdp = county_df.head(500)

print('Top 20 Counties with the most GDP:')
print(top_500_gdp.head(20))

# ============================================================================
# MEDIAN HOUSEHOLD INCOME ANALYSIS
# ============================================================================
median_household_income = pd.read_excel('/Users/deadlinefighter/Desktop/median household income by county.xls', engine='xlrd',skiprows=3)

state_names = [
    "Alabama",
    "Alaska", 
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
    "District of Columbia"
]
#First Remove the Trailing Spaces in columns
median_household_income.columns = median_household_income.columns.str.strip()

#Out of all the columns in the data set, select only the relevant ones
median_household_income_selected_columns = median_household_income[["Name", "Median Household Income"]].copy()

#Remove Trailing spaces
median_household_income_selected_columns['Name'] = median_household_income_selected_columns['Name'].str.strip()

#Turn column data in to numeric value, invalid values with NaN
median_household_income_selected_columns["Median Household Income"] = pd.to_numeric(median_household_income_selected_columns["Median Household Income"], errors='coerce')

#Remove rows that are NaN
median_household_income_selected_columns = median_household_income_selected_columns.dropna(subset= ["Median Household Income"])

#Get the top 50
top_500_income = median_household_income_selected_columns.nlargest(500, "Median Household Income")
def add_state_name(median_household_income_selected_columns):
    result_data = []
    current_state = None

    for index, row in median_household_income_selected_columns.iterrows():
        name = str(row['Name']) if pd.notna(row['Name']) else None
        income = row['Median Household Income']

        if name in state_names:
            current_state = name
        elif name != 'United States' and pd.notna(income) and name not in state_names:
            if current_state:
                full_name = f"{name}, {current_state}"
                result_data.append({
                    'Name' : name,
                    'County Name': full_name,
                    'Median Household Income': income
                })
    return pd.DataFrame(result_data)

income_with_states = add_state_name(median_household_income_selected_columns)

income_with_states_filtered = income_with_states[~income_with_states['Name'].isin(nba_counties_to_exclude)].copy()

income_with_states_filtered['Median Household Income'] = pd.to_numeric(income_with_states_filtered["Median Household Income"], errors='coerce')

income_with_states_filtered = income_with_states_filtered.dropna(subset = "Median Household Income")

income_with_states_filtered = income_with_states_filtered.drop(columns=['Name'])

top_500_income_with_states = income_with_states_filtered.nlargest(500, "Median Household Income")

top_500_income_with_states['Rank'] = range(1, len(top_500_income_with_states)+1)

print('Top 500 counties with the most median household income: ')

print(top_500_income_with_states)

# ============================================================================
# COMPREHENSIVE COUNTY NAME STANDARDIZATION AND DATASET INTEGRATION
# ============================================================================
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
# ============================================================================
# DOMAIN ENTITIES & VALUE OBJECTS
# ============================================================================
@dataclass(frozen = True)
class CountyMetrics:
     county_id: str
     county_name: str

     gdp: Optional[float] = None
     population_growth: Optional[float] = None
     population: Optional[float] = None
     median_income: Optional[float] = None

     @property
     def completeness_score(self) -> int:
           metrics = [self.population, self.population_growth, self.gdp, 
                  self.median_income]
           
           return sum(1 for metric in metrics if metric is not None)

@dataclass
class ScoringWeights:
     population: float = 0.60   
     gdp: float = 0.25        
     income: float = 0.15        
     growth: float = 0.00

     def __post_init__(self) :
          total = sum([self.population, self.gdp, self.income, self.growth])
          if abs(total - 1.0) > 1e-6:
               raise ValueError(f"Weights must sum to 1.0, got {total}")

class DataQuality(Enum):
     COMPLETE = 'complete'
     SUBSTANTIAL = 'substantial'
     ADEQUATE = 'adequate'
     INSUFFICIENT = 'insufficient'

# ============================================================================
# ABSTRACT INTERFACES (DEPENDENCY INVERSION)
# ============================================================================
class DataProcessor(Protocol):
     def process(self, data: pd.DataFrame):
        ...

class CountyStandardizer(ABC):
     @abstractmethod
     def standardize(self, county_name: str):
          pass

class MetricNormalizer(ABC):
     @abstractmethod
     def normalize(self, values: pd.Series):
          pass

class ScoreCalculator(ABC):
     @abstractmethod
     def calculate_score(self, metrics: CountyMetrics, weights: ScoringWeights):
          pass

# ============================================================================
# CONCRETE IMPLEMENTATIONS (STRATEGY PATTERN)
# ============================================================================
class StandardCountyStandardizer(CountyStandardizer):
# def the fucntion
#return None values
#split base on comma
#state part and county part
#if special county name, delte them
#deal with washington dc
#return both if state name exiswt, not both if it does not exist

     def standardize(self, county_name: str)-> Optional[str]:
          if pd.isna(county_name) or not county_name:
               return None
          clean_name = county_name.strip()
          if "," in clean_name:
               county_part, state_part = clean_name.split(',', 1)
               county_part = county_part.strip()
               state_part = state_part.strip()
          else:
               county_part = clean_name
               state_part = ""

          for suffix in ['County', 'Parish', 'Borough']:
               county_part = county_part.replace(suffix, '').strip()
            
          if "District of Columbia" in county_part:
               return "District of Columbia, District of Columbia"
          
          return f"{county_part}, {state_part}" if state_part else county_part

class MinMaxNormalizer(MetricNormalizer):
     #Formula: 100 * (value - min) / (max-min)
     def normalize(self, values: pd.Series) -> pd.Series:
          if values.notna().sum() == 0:
               return values
          
          min_value = values.min(skipna=True)
          max_value = values.max(skipna=True)

          if max_value == min_value:
               return values.fillna(50)
          
          return 100 * (values - min_value)/(max_value-min_value)

class WeightedScoreCalculator(ScoreCalculator):
    def __init__(self, normalizer  : MetricNormalizer):
         self.normalizer = normalizer
    
    def calculate_score(self, metrics_df: pd.DataFrame, weights: ScoringWeights):
         normalize_score = {}
         if 'population' in metrics_df.columns:
              normalize_score['pop'] = self.normalizer.normalize(metrics_df['population'])
         if 'gdp' in metrics_df.columns:
              normalize_score['gdp'] = self.normalizer.normalize(metrics_df['gdp'])
         if 'median_income' in metrics_df.columns:
              normalize_score['income'] = self.normalizer.normalize(metrics_df['median_income'])
         if 'population_growth' in metrics_df.columns:
              normalize_score['growth'] = self.normalizer.normalize(metrics_df['population_growth'])

         composite_score = pd.Series(0, index=metrics_df.index)
         weights_mapping = {
            'pop': weights.population,      # Population weight
            'gdp': weights.gdp,            # GDP weight
            'income': weights.income,      # Income weight
            'growth': weights.growth 
         }

         for metric, score in normalize_score.items():
              composite_score += score.fillna(0) * weights_mapping[metric]
         return composite_score
    
# ============================================================================
# DATA PROCESSOR FACTORY (FACTORY PATTERN)
# ============================================================================
class DataProcessorFactory:
     @staticmethod
     def create_processor(data_type: str, standardizer: CountyStandardizer)-> DataProcessor:
          processors = {
               'population' : PopulationProcessor(standardizer),
               'gdp' : GDPProcessor(standardizer),
               'income' : IncomeProcessor(standardizer),
               'growth' : GrowthProcessor(standardizer)
          }

          if data_type not in processors:
               raise ValueError(f"Unknown data type: {data_type}")
          return processors[data_type]

class PopulationProcessor:
     def __init__(self, standardizer: CountyStandardizer):
          self.standardizer = standardizer
     def process(self, data: pd.DataFrame):
          processed = data[['County Name', 'Population 2024']]
          processed['county_id'] = processed['County Name'].apply(self.standardizer.standardize)
          processed = processed.rename(columns={'Population 2024': 'population'})
          return processed.dropna(subset=['county_id'])

class GDPProcessor:
     def __init__(self, standardizer: CountyStandardizer):
          self.standardizer = standardizer
     def process(self, data):
          processed = data[['County_State_Name', 'GDP_2023']].copy()
          processed = processed.rename(columns = {
               'County_State_Name' : 'county_id',
               'GDP_2023' : 'gdp'
          })
          processed['county_id'] = processed['county_id'].apply(self.standardizer.standardize)
          return processed.dropna(subset = ['county_id'])

class IncomeProcessor:
     def __init__(self, standardizer: CountyStandardizer):
          self.standardizer = standardizer
     def process(self, data: pd.DataFrame) -> pd.DataFrame:
          processed = data[['County Name', 'Median Household Income']].copy()
          processed['county_id'] = processed['County Name'].apply(self.standardizer.standardize)
          processed = processed.rename(columns= {'Median Household Income' : 'median_income'})
          return processed.dropna(subset=['county_id'])
     
class GrowthProcessor:
     def __init__(self, standardizer: CountyStandardizer):
          self.standardizer = standardizer
     def process(self, data: pd.DataFrame)-> pd.DataFrame:
          processed = data[['County Name', 'Population_Growth_Rate']].copy()
          processed['county_id'] = processed['County Name'].apply(self.standardizer.standardize)
          processed = processed.rename(columns= {'Population_Growth_Rate' : 'population_growth'})
          return processed.dropna(subset=['county_id'])

# ============================================================================
# MAIN ANALYSIS ENGINE (CLEAN ARCHITECTURE)
# ============================================================================
class NBAExpansionAnalyzer:
     def __init__(self, 
                  normalizer: MetricNormalizer,
                  score_calculator: ScoreCalculator,
                  weights: ScoringWeights,
                  standardizer: CountyStandardizer):
        
        self.standardizer = standardizer
        self.normalizer = normalizer
        self.score_calculator = score_calculator
        self.weights = weights
        self.factory = DataProcessorFactory()

     def analyze(self, datasets: Dict[str, pd.DataFrame])->pd.DataFrame:
      #initialize processed_datasets
      #iterate through data_type, raw_data in the dataset
      #processor, create the function of creating processor, and standardize the county name
      #create a column in processed datasets called data type, and set that eqaul to processed raw data
      #merge all processed datset on standardized county identifiers
      #Calculate composite score
      #access quality
      #Rank them
        processed_datasets = {}
        for data_type, raw_data in datasets.items():
             processor = self.factory.create_processor(data_type, self.standardizer)
             processed_datasets[data_type] = processor.process(raw_data)
        master_df = self._merge_datasets(processed_datasets)
        master_df['composite_score'] = self.score_calculator.calculate_score(master_df, self.weights)
        master_df['data_quality'] = self._assess_data_quality(master_df)
        master_df = master_df.sort_values('composite_score', ascending=False)
        master_df['rank'] = range(1, len(master_df)+1 )
        return master_df
     
     def _merge_datasets(self, datasets: Dict[str, pd.DataFrame])->pd.DataFrame:
          master_df = None
          for data_type, df in datasets.items():
               if master_df is None:
                    master_df = df.copy()
               else:
                    duplicate_cols = [col for col in df.columns if col in master_df.columns and col !='county_id']
                    if duplicate_cols:
                         df = df.drop(columns=duplicate_cols)
                    master_df = pd.merge(master_df, df,on = 'county_id', how = 'outer')
          return master_df
     def _assess_data_quality(self, df: pd.DataFrame) -> List[DataQuality]:
        metric_cols = ['population', 'gdp', 'median_income', 'population_growth']
        available_cols = [col for col in metric_cols if col in df.columns]
        completness = df[available_cols].notna().sum(axis = 1)

        def quality_level(count):
            if count >= 4: return DataQuality.COMPLETE
            elif count >= 3: return DataQuality.SUBSTANTIAL  
            elif count >= 2: return DataQuality.ADEQUATE
            else: return DataQuality.INSUFFICIENT
        return [quality_level(count) for count in completness]
        
# ============================================================================
# Haversine Distance Calculation
# ============================================================================
nba_teams = pd.DataFrame({
    'team_name': [
        'Boston Celtics', 'Brooklyn Nets', 'New York Knicks', 'Philadelphia 76ers', 'Toronto Raptors',
        'Chicago Bulls', 'Cleveland Cavaliers', 'Detroit Pistons', 'Indiana Pacers', 'Milwaukee Bucks',
        'Atlanta Hawks', 'Charlotte Hornets', 'Miami Heat', 'Orlando Magic', 'Washington Wizards',
        'Denver Nuggets', 'Minnesota Timberwolves', 'Oklahoma City Thunder', 'Portland Trail Blazers', 'Utah Jazz',
        'Golden State Warriors', 'Los Angeles Clippers', 'Los Angeles Lakers', 'Phoenix Suns', 'Sacramento Kings',
        'Dallas Mavericks', 'Houston Rockets', 'Memphis Grizzlies', 'New Orleans Pelicans', 'San Antonio Spurs'
    ],
    'latitude': [
        42.366303, 40.682656, 40.750556, 39.901111, 43.643466,
        41.880556, 41.496389, 42.341111, 39.763889, 43.043056,
        33.757222, 35.225000, 25.781389, 28.539167, 38.898056,
        39.748611, 44.979444, 35.463333, 45.531667, 40.768333,
        37.750278, 34.043056, 34.043056, 33.445833, 38.649167,
        32.790556, 29.680833, 35.138333, 29.948889, 29.426944
    ],
    'longitude': [
        -71.062228, -73.974689, -73.993611, -75.171944, -79.379167,
        -87.674167, -81.688056, -83.055556, -86.155556, -87.916944,
        -84.396389, -80.839167, -80.186944, -81.383611, -77.020833,
        -105.007222, -93.276111, -97.515278, -122.666667, -111.901111,
        -122.203056, -118.267222, -118.267222, -112.071389, -121.518056,
        -96.810278, -95.362222, -90.050556, -90.081944, -98.4375
    ]
})

from math import radians, cos, sin, asin, sqrt
from geopy.geocoders import Nominatim
import time
import pandas as pd
def haversine(lat1, lon1, lat2, lon2):
       """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in miles
    """
     #COnvert decimal degrees to radians
       lat1, lon1, lat2, lon2 = map(radians, [lat1,lon1, lat2, lon2])
       #Haversine formula
       dlat = lat2 - lat1 
       dlon = lon2 - lon1
       a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) *sin(dlon/2)**2
       c = 2 * asin(sqrt(a))

       r = 3956
       return c * r

def calculate_min_nba_distances(county_lat, county_lon, nba_teams_df):
     if pd.isna(county_lat) or pd.isna(county_lon):
          return None
     
     distances =[]
     for _, team in nba_teams_df.iterrows():
          dict = haversine(county_lat, county_lon, team['latitude'], team['longitude'])
          distances.append(dict)
     
     return min(distances)


def get_county_coordinates_batch(county_list):
     geolocator = Nominatim(user_agent= 'nba_expansion_analysis')
     failed_counties = []
     results = {}

     for i, county_name in enumerate(county_list):
          try:
               query = f"{county_name}, USA"
               location = geolocator.geocode(query, timeout = 10)

               if location:
                    coords = (location.latitude, location.longitude)
                    results[county_name] = coords
                    print(f"✅ {i+1}/{len(county_list)} {county_name}: {coords}")
               else:
                    results[county_name] = None
                    failed_counties.append(county_name)
                    print(f"⚠️ {i+1}/{len(county_list)} {county_name}: Not Found")
          except Exception as e:
               results[county_name] = None
               failed_counties.append(county_name)
               print(f"⚠️ {i+1}/{len(county_list)} {county_name}: error - {e}")

          time.sleep(1.1)
     print(f"\nSuccessfully processed: {len(county_list) - len(failed_counties)}/{len(county_list)}")
     if failed_counties:
          print(f"Failed to process: {failed_counties}")
     
     return results

def add_coordinates_to_nba_analysis(results_df):
    county_names = results_df['county_id'].tolist()
    coordinates = get_county_coordinates_batch(county_names)

    results_df['latitude'] = results_df['county_id'].map(lambda x: coordinates[x][0] if x in coordinates and coordinates[x] else None)
    results_df['longitude'] = results_df['county_id'].map(lambda x: coordinates[x][1] if x in coordinates and coordinates[x] else None)

    results_df['min_distance_to_nba'] = results_df.apply(lambda row: calculate_min_nba_distances(row['latitude'], row['longitude'], nba_teams) if pd.notna(row['latitude']) else 999, axis = 1)

    return results_df
# ============================================================================
# MAIN
# ============================================================================
def main():
     normalizer = MinMaxNormalizer()
     standardizer = StandardCountyStandardizer()
     score_calculator = WeightedScoreCalculator(normalizer)

     weights = ScoringWeights(
        population=0.60,  # Increased importance of market size
        gdp=0.25,        # Economic strength for revenue
        income=0.15,     # Fan purchasing power  # Corporate sponsorship opportunities
        growth=0.00 
     )

     analyzer = NBAExpansionAnalyzer(
          standardizer=standardizer,
          normalizer=normalizer,
          score_calculator=score_calculator,
          weights=weights
     )
     datasets = {
        'population': top_500_population,        # Your population DataFrame
        'gdp': top_500_gdp,                     # Your GDP DataFrame
        'income': top_500_income_with_states,   # Your income DataFrame
        'growth': top_500_population_growth     # Your growth DataFrame
    }
     results = analyzer.analyze(datasets)
     print("🌍 Getting coordinates for top counties...")
     results_with_coords = add_coordinates_to_nba_analysis(results.head(200))
     eligible_counties = results_with_coords[results_with_coords['min_distance_to_nba'] > 110]
     print("🏆 TOP 200 NBA EXPANSION CANDIDATES:")
     print("=" * 80)

     top_200_short_list = results.head(234)
     top_200_cleaned = top_200_short_list.dropna(subset = ['county_id', 'County Name'])

     for _, row in eligible_counties.head(25).iterrows():
          distance = row['min_distance_to_nba']
          distance_str = f"{distance:.0f} miles" if pd.notna(distance) else "Unknown"
          print(f"{row['rank']:2d}. {row['county_id']:<35}"
                f"Score: {row['composite_score']:.1f} "
                f"Distance: {distance_str}")
     
     return results_with_coords
# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
     result = main()

# ============================================================================
# Interactive Map 
# ============================================================================
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd

# Configure the Streamlit page
st.set_page_config(
    page_title="NBA Expansion Analysis",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for NBA styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div > select {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
    }
    .metric-container {
        background: linear-gradient(135deg, #C8102E, #1D428A);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .candidate-card {
        background: #f8f9fa;
        border-left: 4px solid #C8102E;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .tier-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    .tier-consensus { background: #C8102E; color: white; }
    .tier-strong { background: #ff8c00; color: white; }
    .tier-regional { background: #1D428A; color: white; }
</style>
""", unsafe_allow_html=True)

# NBA Expansion Candidates Data (Your refined rankings)
@st.cache_data
def load_candidates_data():
    candidates_data = {
        'rank': [1, 2, 3, 4, 5, 6, 7, 8],
        'city': ['Seattle', 'Las Vegas', 'Jacksonville', 'Nashville', 'Kansas City', 'Pittsburgh', 'Louisville', 'Raleigh'],
        'county': ['King County', 'Clark County', 'Duval County', 'Davidson County', 'Johnson County', 'Allegheny County', 'Jefferson County', 'Wake County'],
        'state': ['Washington', 'Nevada', 'Florida', 'Tennessee', 'Kansas', 'Pennsylvania', 'Kentucky', 'North Carolina'],
        'latitude': [47.6062, 36.1716, 30.3322, 36.1627, 38.8813, 40.4406, 38.2527, 35.7796],
        'longitude': [-122.3321, -115.1391, -81.6557, -86.7816, -94.8191, -79.9959, -85.7585, -78.6382],
        'score': [73.5, 51.9, 22.4, 16.9, 16.8, 27.5, 16.3, 30.9],
        'distance': [140, 240, 126, 197, 278, 114, 113, 131],
        'tier': ['Consensus #1', 'Entertainment Capital', 'Largest Untapped', 'Music City Rising', 'Geographic Sweet Spot', 'Steeltown Passion', 'Basketball Heritage', 'Research Triangle'],
        'tier_category': ['consensus', 'consensus', 'strong', 'strong', 'strong', 'regional', 'regional', 'regional'],
        'population': ['4,018,762', '2,266,715', '1,605,848', '2,012,649', '2,192,035', '2,370,930', '1,365,557', '1,390,687'],
        'county_population': ['2,269,675', '2,266,715', '995,567', '715,884', '609,863', '1,250,578', '782,969', '1,129,410'],
        'median_income': ['$94,027', '$63,830', '$56,200', '$56,152', '$85,111', '$56,700', '$52,238', '$73,825'],
        'why_viable': [
            'Former SuperSonics fanbase demanding return, Climate Pledge Arena NBA-ready, tech economy powerhouse, Commissioner Silver endorsement',
            'Entertainment capital with proven sports success, T-Mobile Arena with planned $300M NBA upgrades, tourism economy supports premium pricing',
            'Largest US city without NBA team, NFL Jaguars prove sports market viability, 100+ miles from Orlando/Miami prevents cannibalization',
            'Music City sports explosion with Titans/Predators success, Bridgestone Arena available, fastest-growing Southeast market',
            'Critical geographic gap between Denver/Memphis, Chiefs Kingdom proves passionate fanbase, T-Mobile Center NBA-capable',
            'Legendary sports passion (Steelers/Penguins), PPG Paints Arena ready, 2.4M metro undervalued by expansion analysts',
            'Basketball capital with deep college tradition, KFC Yum! Center NBA-ready, no regional competition within 200 miles',
            'Research Triangle economic boom, ACC basketball culture, PNC Arena undergoing renovations for multi-sport use'
        ],
        'challenges': [
            'None significant - consensus choice with infrastructure, fanbase, and official NBA interest already established',
            'Balancing tourist entertainment with local fanbase development, competition from Raiders/Knights for corporate dollars',
            'VyStar Arena needs capacity expansion to NBA standards, questions about sustained attendance beyond opening year novelty',
            'Bridgestone Arena primarily hockey-designed, requires significant modifications for optimal basketball sightlines',
            'Split governance between Kansas/Missouri, T-Mobile Center needs upgrades, individual market smaller than combined metro',
            'Limited official NBA expansion mentions despite strong fundamentals, perception as declining market despite economic growth',
            'Smaller metro market compared to other candidates, relies heavily on regional draw from Kentucky/Southern Indiana',
            'PNC Arena renovations ongoing through 2027, limited professional sports passion compared to college basketball focus'
        ],
        'arena': [
            'Climate Pledge Arena (18,100 basketball capacity) - NBA ready',
            'T-Mobile Arena (20,000 capacity) with planned $300M NBA upgrades',
            'VyStar Veterans Memorial Arena (14,091 capacity) - needs expansion',
            'Bridgestone Arena (17,113 basketball capacity) - needs modifications',
            'T-Mobile Center (17,000+ capacity) - could accommodate NBA',
            'PPG Paints Arena (18,387 capacity) with recent $30M upgrades',
            'KFC Yum! Center (22,000+ capacity) - NBA ready',
            'PNC Arena (18,700 capacity) undergoing renovations through 2027'
        ]
    }
    return pd.DataFrame(candidates_data)

# NBA Teams Data
@st.cache_data
def load_nba_teams_data():
    nba_teams_data = {
        'team': ['Lakers/Clippers', 'Warriors', 'Kings', 'Suns', 'Nuggets', 'Trail Blazers', 'Jazz', 'Thunder', 
                 'Mavericks', 'Spurs', 'Rockets', 'Timberwolves', 'Grizzlies', 'Pelicans', 'Bulls', 'Cavaliers',
                 'Pistons', 'Pacers', 'Bucks', 'Hawks', 'Hornets', 'Heat', 'Magic', 'Knicks', 'Nets', 
                 '76ers', 'Celtics', 'Raptors', 'Wizards'],
        'city': ['Los Angeles', 'San Francisco', 'Sacramento', 'Phoenix', 'Denver', 'Portland', 'Salt Lake City', 'Oklahoma City',
                 'Dallas', 'San Antonio', 'Houston', 'Minneapolis', 'Memphis', 'New Orleans', 'Chicago', 'Cleveland',
                 'Detroit', 'Indianapolis', 'Milwaukee', 'Atlanta', 'Charlotte', 'Miami', 'Orlando', 'New York', 'Brooklyn',
                 'Philadelphia', 'Boston', 'Toronto', 'Washington'],
        'lat': [34.043, 37.75, 38.649, 33.446, 39.749, 45.532, 40.768, 35.463,
                32.791, 29.427, 29.681, 44.979, 35.138, 29.949, 41.881, 41.496,
                42.341, 39.764, 43.043, 33.757, 35.225, 25.781, 28.539, 40.751, 40.683,
                39.901, 42.366, 43.643, 38.898],
        'lon': [-118.267, -122.203, -121.518, -112.071, -105.007, -122.667, -111.901, -97.515,
                -96.81, -98.438, -95.362, -93.276, -90.051, -90.082, -87.674, -81.688,
                -83.056, -86.156, -87.917, -84.396, -80.839, -80.187, -81.384, -73.994, -73.975,
                -75.172, -71.062, -79.379, -77.021]
    }
    return pd.DataFrame(nba_teams_data)

def create_candidate_popup(candidate):
    """Create detailed popup content for expansion candidates"""
    tier_colors = {
        'consensus': '#C8102E',
        'strong': '#ff8c00', 
        'regional': '#1D428A'
    }
    tier_color = tier_colors.get(candidate['tier_category'], '#666')
    
    popup_html = f"""
    <div style="width: 350px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;">
        <div style="background: linear-gradient(135deg, #C8102E, #1D428A); color: white; padding: 15px; margin: -12px -12px 15px -12px; border-radius: 10px 10px 0 0;">
            <h3 style="margin: 0; font-size: 20px; font-weight: 600;">#{candidate['rank']} {candidate['city']}</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 14px;">{candidate['county']}, {candidate['state']}</p>
            <div style="background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 15px; margin-top: 8px; display: inline-block;">
                <span style="font-size: 12px; font-weight: 600;">{candidate['tier']}</span>
            </div>
        </div>
        
        <div style="padding: 12px;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
                <div style="text-align: center; background: #e3f2fd; padding: 10px; border-radius: 8px;">
                    <div style="font-size: 18px; font-weight: bold; color: #1565c0;">{candidate['score']}</div>
                    <div style="font-size: 11px; color: #666; text-transform: uppercase;">Score</div>
                </div>
                <div style="text-align: center; background: #e8f5e8; padding: 10px; border-radius: 8px;">
                    <div style="font-size: 18px; font-weight: bold; color: #2e7d32;">{candidate['distance']} mi</div>
                    <div style="font-size: 11px; color: #666; text-transform: uppercase;">Distance</div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
                <div style="text-align: center; background: #fff3e0; padding: 8px; border-radius: 6px;">
                    <div style="font-size: 14px; font-weight: bold; color: #ef6c00;">{candidate['population']}</div>
                    <div style="font-size: 10px; color: #666;">Metro Population</div>
                </div>
                <div style="text-align: center; background: #f3e5f5; padding: 8px; border-radius: 6px;">
                    <div style="font-size: 14px; font-weight: bold; color: #7b1fa2;">{candidate['median_income']}</div>
                    <div style="font-size: 10px; color: #666;">Median Income</div>
                </div>
            </div>
            
            <div style="margin-bottom: 12px;">
                <h4 style="color: #C8102E; margin: 0 0 6px 0; font-size: 14px; font-weight: 600;">Why This Market Works</h4>
                <p style="margin: 0; font-size: 12px; line-height: 1.5; color: #333;">{candidate['why_viable']}</p>
            </div>
            
            <div style="margin-bottom: 12px;">
                <h4 style="color: #1D428A; margin: 0 0 6px 0; font-size: 14px; font-weight: 600;">Potential Challenges</h4>
                <p style="margin: 0; font-size: 12px; line-height: 1.5; color: #333;">{candidate['challenges']}</p>
            </div>
            
            <div style="background: #f8f9fa; padding: 8px; border-radius: 6px; border-left: 3px solid {tier_color};">
                <h4 style="margin: 0 0 4px 0; font-size: 12px; font-weight: 600; color: #666;">Arena Situation</h4>
                <p style="margin: 0; font-size: 11px; line-height: 1.4; color: #333;">{candidate['arena']}</p>
            </div>
        </div>
    </div>
    """
    return popup_html

def create_expansion_map(candidates_df, nba_teams_df, filter_option, show_nba, selected_candidate_rank=None):
    """Create the interactive NBA expansion map"""
    
    # Initialize map centered on US
    m = folium.Map(
        location=[39.8283, -98.5795],
        zoom_start=4,
        tiles='CartoDB positron',
        attr='NBA Expansion Analysis'
    )
    
    # Filter candidates based on selection
    if filter_option == "Top 2 Only":
        filtered_candidates = candidates_df[candidates_df['rank'] <= 2]
    elif filter_option == "Strong Contenders (Top 5)":
        filtered_candidates = candidates_df[candidates_df['rank'] <= 5]
    elif filter_option == "Consensus Picks Only":
        filtered_candidates = candidates_df[candidates_df['tier_category'] == 'consensus']
    else:
        filtered_candidates = candidates_df
    
    # Color scheme for different tiers
    tier_colors = {
        'consensus': '#C8102E',    # NBA Red
        'strong': '#ff8c00',       # Orange  
        'regional': '#1D428A'      # NBA Blue
    }
    
    # Add expansion candidate markers
    for _, candidate in filtered_candidates.iterrows():
        color = tier_colors.get(candidate['tier_category'], '#666')
        
        # Larger radius for higher-ranked candidates
        if candidate['rank'] <= 2:
            radius = 15
        elif candidate['rank'] <= 5:
            radius = 12
        else:
            radius = 9
            
        # Highlight selected candidate
        if selected_candidate_rank and candidate['rank'] == selected_candidate_rank:
            radius += 3
            weight = 4
        else:
            weight = 3
        
        marker = folium.CircleMarker(
            location=[candidate['latitude'], candidate['longitude']],
            radius=radius,
            popup=folium.Popup(create_candidate_popup(candidate), max_width=380),
            color='white',
            weight=weight,
            fillColor=color,
            fillOpacity=0.9,
            tooltip=f"#{candidate['rank']} {candidate['city']} - {candidate['tier']}"
        )
        marker.add_to(m)
    
    # Add existing NBA team markers
    if show_nba:
        for _, team in nba_teams_df.iterrows():
            folium.CircleMarker(
                location=[team['lat'], team['lon']],
                radius=6,
                popup=folium.Popup(f"<b>{team['team']}</b><br>{team['city']}", max_width=200),
                color='white',
                weight=2,
                fillColor='#1D428A',
                fillOpacity=0.7,
                tooltip=f"{team['team']} ({team['city']})"
            ).add_to(m)
    
    # Add custom legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 280px; 
                background-color: rgba(255, 255, 255, 0.95); 
                border: 2px solid #ddd; border-radius: 12px;
                z-index: 9999; font-size: 13px; padding: 16px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;">
    <h4 style="margin-top: 0; margin-bottom: 12px; color: #333; font-weight: 600;">🏀 NBA Expansion Analysis</h4>
    <p style="margin: 6px 0;"><span style="color: #C8102E; font-size: 16px;">●</span> Consensus Picks (Seattle, Las Vegas)</p>
    <p style="margin: 6px 0;"><span style="color: #ff8c00; font-size: 16px;">●</span> Strong Contenders (Jacksonville, Nashville, KC)</p>
    <p style="margin: 6px 0;"><span style="color: #1D428A; font-size: 16px;">●</span> Regional Markets (Pittsburgh, Louisville, Raleigh)</p>
    <p style="margin: 6px 0;"><span style="color: #1D428A; font-size: 14px;">●</span> Existing NBA Teams</p>
    <hr style="margin: 12px 0; border: 1px solid #eee;">
    <p style="margin: 6px 0; font-size: 11px; color: #666; line-height: 1.4;">
        <strong>Methodology:</strong> 60% Population, 25% GDP, 15% Income<br>
        <strong>Distance Filter:</strong> 110+ miles from existing teams<br>
        <strong>Rankings:</strong> Data + Basketball Culture Analysis
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def display_candidate_details(candidate):
    """Display detailed candidate information in sidebar"""
    
    tier_styles = {
        'consensus': 'tier-consensus',
        'strong': 'tier-strong', 
        'regional': 'tier-regional'
    }
    
    tier_style = tier_styles.get(candidate['tier_category'], 'tier-regional')
    
    st.markdown(f"""
    <div class="candidate-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
            <div>
                <h2 style="margin: 0; color: #C8102E; font-size: 1.8rem;">#{candidate['rank']} {candidate['city']}</h2>
                <p style="margin: 0.2rem 0; color: #666; font-size: 1rem;">{candidate['county']}, {candidate['state']}</p>
            </div>
        </div>
        
        <span class="tier-badge {tier_style}">{candidate['tier']}</span>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.5rem 0;">
            <div class="metric-container">
                <div style="font-size: 1.5rem; font-weight: bold;">{candidate['score']}</div>
                <div style="font-size: 0.8rem; opacity: 0.9;">Composite Score</div>
            </div>
            <div class="metric-container">
                <div style="font-size: 1.5rem; font-weight: bold;">{candidate['distance']} mi</div>
                <div style="font-size: 0.8rem; opacity: 0.9;">Distance to NBA</div>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin: 1rem 0;">
            <div style="text-align: center; background: #f8f9fa; padding: 0.8rem; border-radius: 6px;">
                <div style="font-weight: bold; color: #C8102E;">{candidate['population']}</div>
                <div style="font-size: 0.75rem; color: #666;">Metro Population</div>
            </div>
            <div style="text-align: center; background: #f8f9fa; padding: 0.8rem; border-radius: 6px;">
                <div style="font-weight: bold; color: #1D428A;">{candidate['median_income']}</div>
                <div style="font-size: 0.75rem; color: #666;">Median Income</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed sections
    st.markdown("#### ✅ Why This Market Works")
    st.write(candidate['why_viable'])
    
    st.markdown("#### ⚠️ Potential Challenges")
    st.write(candidate['challenges'])
    
    st.markdown("#### 🏟️ Arena Situation")
    st.write(candidate['arena'])

def main():
    """Main application function"""
    
    # Load data
    candidates_df = load_candidates_data()
    nba_teams_df = load_nba_teams_data()
    
    # Header
    st.title("🏀 NBA Expansion Analysis")
    st.markdown("### Data-Driven Market Evaluation with Basketball Intelligence")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Map Controls")
    
    # Filter options
    filter_option = st.sidebar.selectbox(
        "Show Candidates:",
        ["All 8 Candidates", "Strong Contenders (Top 5)", "Top 2 Only", "Consensus Picks Only"]
    )
    
    # NBA teams toggle
    show_nba = st.sidebar.checkbox("Show Existing NBA Teams", value=True)
    
    # Candidate selection
    candidate_options = ["Overview"] + [f"#{row['rank']} {row['city']}" for _, row in candidates_df.iterrows()]
    selected_candidate = st.sidebar.selectbox(
        "Focus on Candidate:",
        candidate_options
    )
    
    # Reset view button
    if st.sidebar.button("🔄 Reset Map View"):
        st.rerun()
    
    # Main layout
    col1, col2 = st.columns([2.5, 1.5])
    
    with col1:
        st.subheader("🗺️ Interactive Expansion Map")
        
        # Determine selected candidate rank for highlighting
        selected_rank = None
        if selected_candidate != "Overview":
            selected_rank = int(selected_candidate.split('#')[1].split(' ')[0])
        
        # Create and display map
        expansion_map = create_expansion_map(
            candidates_df, 
            nba_teams_df, 
            filter_option, 
            show_nba,
            selected_rank
        )
        
        # Display the map
        map_data = st_folium(
            expansion_map, 
            width=700, 
            height=500,
            returned_objects=["last_object_clicked", "last_clicked"]
        )
    
    with col2:
        if selected_candidate == "Overview":
            st.subheader("📊 Analysis Overview")
            
            st.markdown("""
            **Refined Rankings Balance:**
            - Quantitative scores (60% population, 25% GDP, 15% income)
            - Sports culture intensity and basketball passion
            - Geographic positioning and market gaps  
            - Arena readiness and ownership viability
            
            **Key Strategic Insights:**
            - **Seattle & Las Vegas**: Consensus expansion leaders
            - **Jacksonville**: Largest untapped market with NFL success
            - **Nashville**: Music City's explosive sports growth
            - **Kansas City**: Critical geographic gap with passionate fanbase
            """)
            
            # Quick metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Candidates", "8", "Refined Analysis")
            with col_b:
                st.metric("Current NBA Teams", "30", "Established Markets")
            with col_c:
                st.metric("Distance Threshold", "110+ mi", "Territorial Protection")
            
            # Top candidates summary
            st.markdown("#### 🏆 Candidate Tiers")
            
            # Consensus picks
            consensus_candidates = candidates_df[candidates_df['tier_category'] == 'consensus']
            with st.expander("🥇 Consensus Leaders", expanded=True):
                for _, candidate in consensus_candidates.iterrows():
                    st.markdown(f"**#{candidate['rank']} {candidate['city']}** - {candidate['tier']}")
            
            # Strong contenders  
            strong_candidates = candidates_df[candidates_df['tier_category'] == 'strong']
            with st.expander("⭐ Strong Contenders"):
                for _, candidate in strong_candidates.iterrows():
                    st.markdown(f"**#{candidate['rank']} {candidate['city']}** - {candidate['tier']}")
            
            # Regional markets
            regional_candidates = candidates_df[candidates_df['tier_category'] == 'regional']
            with st.expander("🏀 Regional Markets"):
                for _, candidate in regional_candidates.iterrows():
                    st.markdown(f"**#{candidate['rank']} {candidate['city']}** - {candidate['tier']}")
        
        else:
            # Show detailed candidate information
            rank = int(selected_candidate.split('#')[1].split(' ')[0])
            candidate = candidates_df[candidates_df['rank'] == rank].iloc[0]
            
            display_candidate_details(candidate)
    
    # Footer methodology
    st.markdown("---")
    with st.expander("📈 Methodology & Data Sources"):
        st.markdown("""
        **Quantitative Analysis:**
        - **Population (60%)**: County and metro area population data
        - **GDP (25%)**: Regional economic output and growth
        - **Median Income (15%)**: Household purchasing power metrics
        - **Distance Filter**: 110+ miles from existing NBA teams
        
        **Qualitative Factors:**
        - Sports culture intensity and proven fanbase passion
        - Geographic positioning and market necessity
        - Arena infrastructure and ownership readiness
        - Basketball-specific interest vs. general population metrics
        
        **Data Sources:**
        - U.S. Census Bureau (population data)
        - Bureau of Economic Analysis (GDP data)  
        - American Community Survey (income data)
        - NBA official statements and industry analysis
        
        This analysis combines sophisticated market fundamentals with basketball intelligence to identify the most viable expansion candidates for NBA growth.
        """)

if __name__ == "__main__":
    main()

     
     