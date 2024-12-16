import os
import sys
import pandas as pd
import numpy as np




import subprocess
import sys

# Check if pip is installed
try:
    import pip
except ImportError:
    print("pip not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "ensurepip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

# Install seaborn if not installed
try:
    import seaborn as sns
except ImportError:
    print("Seaborn not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn




class DataAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.df = None
        # Debugging: Print the filename to verify
        print(f"Initializing with file: {filename}")
        print(f"File exists: {os.path.exists(filename)}")
        
        # Use environment variable securely
        self.client = OpenAI(api_key=os.environ.get("AIPROXY_TOKEN"))
    
    def load_data(self):
        try:
            # Remove low_memory parameter or use 'c' engine
            self.df = pd.read_csv(
                self.filename, 
                encoding='iso-8859-1',  # Handle encoding
                na_values=['NA', 'N/A', ''],  # Handle missing values
                engine='c'  # More efficient default engine
            )
            
            # Debugging: Print basic info about loaded data
            print(f"Data loaded successfully!")
            print(f"DataFrame shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            
            return self._get_data_overview()
        except Exception as e:
            # Detailed error logging
            print(f"Error loading data: {str(e)}")
            # If possible, print the full traceback
            import traceback
            traceback.print_exc()
            return f"Error loading data: {str(e)}"
    
    def _get_data_overview(self):
        # Ensure df is not None before processing
        if self.df is None:
            return "No data loaded"
        
        overview = {
            'columns': self.df.columns.tolist(),
            'data_types': str(self.df.dtypes),
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'sample_data': self.df.head().to_dict()
        }
        return overview
    
    def perform_analysis(self):
        # Ensure data is loaded
        if self.df is None:
            print("No data loaded. Skipping analysis.")
            return {}
        
        analyses = {}
        
        # Identify numeric columns with more robust method
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        print(f"Numeric columns found: {numeric_cols}")
        
        # Only proceed if we have numeric columns
        if not numeric_cols:
            print("No numeric columns found for analysis")
            return analyses
        
        # Correlation Analysis
        if len(numeric_cols) > 1:
            try:
                corr_matrix = self.df[numeric_cols].corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Matrix')
                plt.tight_layout()
                plt.savefig('correlation_matrix.png')
                plt.close()
                analyses['correlation'] = corr_matrix.to_dict()
            except Exception as e:
                print(f"Correlation analysis error: {e}")
        
        # Clustering Analysis
        if len(numeric_cols) > 2:
            try:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(self.df[numeric_cols])
                
                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(scaled_data)
                
                kmeans = KMeans(n_clusters=min(3, len(numeric_cols)), random_state=42)
                clusters = kmeans.fit_predict(scaled_data)
                
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis')
                plt.title('Data Clustering Visualization')
                plt.xlabel('First Principal Component')
                plt.ylabel('Second Principal Component')
                plt.colorbar(scatter)
                plt.savefig('clustering_visualization.png')
                plt.close()
                
                analyses['clustering'] = {
                    'n_clusters': len(np.unique(clusters)),
                    'cluster_distribution': np.unique(clusters, return_counts=True)[1].tolist()
                }
            except Exception as e:
                print(f"Clustering analysis error: {e}")
        
        return analyses
    
    def generate_readme(self, data_overview, analysis_results):
        """
        Generate a narrative README.md file describing the data and analysis
        """
        # Construct the narrative
        readme_content = f"""# Data Analysis Report

## Data Overview

We analyzed a dataset with the following characteristics:
- **Total Columns**: {len(data_overview.get('columns', []))}
- **Total Rows**: {data_overview.get('shape', (0,))[0]}
- **Columns**: {', '.join(data_overview.get('columns', []))}

### Data Types
{data_overview.get('data_types', 'No data type information available')}

### Missing Values
{pd.Series(data_overview.get('missing_values', {})).to_string()}

## Analysis Insights

### Correlation Analysis
We performed a correlation analysis to understand relationships between numeric variables. 

{self._generate_correlation_narrative(analysis_results.get('correlation', {}))}

### Clustering Analysis
We applied K-means clustering to identify patterns in the data:
- **Number of Clusters**: {analysis_results.get('clustering', {}).get('n_clusters', 'N/A')}
- **Cluster Distribution**: {analysis_results.get('clustering', {}).get('cluster_distribution', 'N/A')}

## Visualizations
- `correlation_matrix.png`: A heatmap showing correlations between numeric variables
- `clustering_visualization.png`: A scatter plot of data points colored by cluster membership

## Recommendations
1. Review the correlation matrix to understand variable relationships
2. Examine the clustering visualization to identify potential groupings or segments in your data
3. Consider further in-depth analysis of the identified clusters and highly correlated variables
"""
        
        # Write to README.md
        with open('README.md', 'w') as f:
            f.write(readme_content)
        
        return readme_content
    
    def _generate_correlation_narrative(self, correlation_matrix):
        """
        Generate a narrative description of the correlation matrix
        """
        if not correlation_matrix:
            return "No significant correlations were found in the dataset."
        
        # Convert dict to DataFrame for easier manipulation
        corr_df = pd.DataFrame(correlation_matrix)
        
        # Find strong positive and negative correlations
        strong_correlations = []
        for col1 in corr_df.columns:
            for col2 in corr_df.columns:
                if col1 != col2:
                    corr_value = corr_df.loc[col1, col2]
                    if abs(corr_value) > 0.5:  # Threshold for strong correlation
                        strong_correlations.append(
                            f"- **{col1}** and **{col2}**: Correlation of {corr_value:.2f}"
                        )
        
        if strong_correlations:
            return "We identified the following strong correlations:\n" + "\n".join(strong_correlations)
        else:
            return "No strong correlations (>0.5 or <-0.5) were found between variables."
    
    # Add the missing method to run full analysis
    def run_full_analysis(self):
        data_overview = self.load_data()
        analysis_results = self.perform_analysis()
        
        # Generate README
        self.generate_readme(data_overview, analysis_results)
        
        # Combine and return results
        return {
            'data_overview': data_overview,
            'analysis_results': analysis_results
        }

def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Ensure absolute path
    filename = os.path.abspath(filename)
    
    # Create output directory based on dataset name
    dataset_name = os.path.splitext(os.path.basename(filename))[0]
    os.makedirs(dataset_name, exist_ok=True)
    os.chdir(dataset_name)
    
    # Run analysis
    analyzer = DataAnalyzer(filename)
    results = analyzer.run_full_analysis()
    print(results)

if __name__ == "__main__":
    main()
