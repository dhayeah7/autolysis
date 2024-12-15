# Data Analysis Report

## Data Overview

We analyzed a dataset with the following characteristics:
- **Total Columns**: 11
- **Total Rows**: 2363
- **Columns**: Country name, year, Life Ladder, Log GDP per capita, Social support, Healthy life expectancy at birth, Freedom to make life choices, Generosity, Perceptions of corruption, Positive affect, Negative affect

### Data Types
Country name                         object
year                                  int64
Life Ladder                         float64
Log GDP per capita                  float64
Social support                      float64
Healthy life expectancy at birth    float64
Freedom to make life choices        float64
Generosity                          float64
Perceptions of corruption           float64
Positive affect                     float64
Negative affect                     float64
dtype: object

### Missing Values
Country name                          0
year                                  0
Life Ladder                           0
Log GDP per capita                   28
Social support                       13
Healthy life expectancy at birth     63
Freedom to make life choices         36
Generosity                           81
Perceptions of corruption           125
Positive affect                      24
Negative affect                      16

## Analysis Insights

### Correlation Analysis
We performed a correlation analysis to understand relationships between numeric variables. 

We identified the following strong correlations:
- **Life Ladder** and **Log GDP per capita**: Correlation of 0.78
- **Life Ladder** and **Social support**: Correlation of 0.72
- **Life Ladder** and **Healthy life expectancy at birth**: Correlation of 0.71
- **Life Ladder** and **Freedom to make life choices**: Correlation of 0.54
- **Life Ladder** and **Positive affect**: Correlation of 0.52
- **Log GDP per capita** and **Life Ladder**: Correlation of 0.78
- **Log GDP per capita** and **Social support**: Correlation of 0.69
- **Log GDP per capita** and **Healthy life expectancy at birth**: Correlation of 0.82
- **Social support** and **Life Ladder**: Correlation of 0.72
- **Social support** and **Log GDP per capita**: Correlation of 0.69
- **Social support** and **Healthy life expectancy at birth**: Correlation of 0.60
- **Healthy life expectancy at birth** and **Life Ladder**: Correlation of 0.71
- **Healthy life expectancy at birth** and **Log GDP per capita**: Correlation of 0.82
- **Healthy life expectancy at birth** and **Social support**: Correlation of 0.60
- **Freedom to make life choices** and **Life Ladder**: Correlation of 0.54
- **Freedom to make life choices** and **Positive affect**: Correlation of 0.58
- **Positive affect** and **Life Ladder**: Correlation of 0.52
- **Positive affect** and **Freedom to make life choices**: Correlation of 0.58

### Clustering Analysis
We applied K-means clustering to identify patterns in the data:
- **Number of Clusters**: N/A
- **Cluster Distribution**: N/A

## Visualizations
- `correlation_matrix.png`: A heatmap showing correlations between numeric variables
- `clustering_visualization.png`: A scatter plot of data points colored by cluster membership

## Recommendations
1. Review the correlation matrix to understand variable relationships
2. Examine the clustering visualization to identify potential groupings or segments in your data
3. Consider further in-depth analysis of the identified clusters and highly correlated variables
