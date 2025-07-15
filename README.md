# NBA Expansion Analysis

**Data-driven analysis of potential NBA expansion markets using demographic data and interactive visualization.**

## What It Does

Analyzes every U.S. county to identify the best NBA expansion candidates using population (60%), GDP (25%), and income (15%) data. Filters out counties within 110 miles of existing NBA teams and ranks the top prospects with an interactive map.

## Key Results

- #1 Seattle - Former SuperSonics market with NBA-ready arena
- #2 Las Vegas - Entertainment capital with proven sports success  
- #3 Jacksonville - Largest untapped market (1.6M+ population)
- **Top 8 candidates** identified across three tiers

## Tech Stack

- **Python**: pandas, NumPy for data processing
- **Streamlit**: Interactive web dashboard
- **Folium**: Interactive maps
- **Data**: U.S. Census, Bureau of Economic Analysis

## Quick Start

```bash
# Install dependencies
pip install streamlit pandas folium geopy

# Run the app
streamlit run app.py

# Open browser to http://localhost:8501
```

## Code Structure

**Single file:** `app.py` - Everything organized with comment dividers

```python
# ===== DATA MODELS =====
# CountyMetrics, ScoringWeights, DataQuality classes

# ===== ABSTRACT INTERFACES =====  
# DataProcessor, CountyStandardizer, MetricNormalizer protocols

# ===== CONCRETE IMPLEMENTATIONS =====
# StandardCountyStandardizer, MinMaxNormalizer, WeightedScoreCalculator

# ===== MAIN ANALYSIS ENGINE =====
# Data processing, scoring, ranking, distance filtering

# ===== INTERACTIVE VISUALIZATION =====
# Streamlit dashboard with Folium maps and custom CSS
```

## Methodology

- **60% Population** - Market size for sustained attendance
- **25% GDP** - Economic strength for corporate partnerships  
- **15% Income** - Consumer purchasing power
- **110+ mile filter** - Territorial protection from existing teams

## Data Sources

- U.S. Census Bureau (population)
- Bureau of Economic Analysis (GDP)
- American Community Survey (income)
