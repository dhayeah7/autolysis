import subprocess
import sys
import os
import json
from typing import Optional, Dict, Any

def install_packages(packages):
    """Enhanced robust package installation method"""
    python_executable = sys.executable
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([python_executable, '-m', 'pip', 'install', package])
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}")
                sys.exit(1)

# Robust package list
required_packages = [
    "pandas", 
    "numpy", 
    "seaborn", 
    "matplotlib", 
    "openai==1.3.0",  # Pinned version for stability
    "scikit-learn"
]

# Install packages
install_packages(required_packages)

import pandas as pd
import numpy as np
from openai import OpenAI

class LLMIntegration:
    def __init__(self, api_key: Optional[str] = None):
        """
        Robust OpenAI API key retrieval with multiple fallback mechanisms
        """
        self.api_key = self._get_openai_key(api_key)
        self.client = None
        
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"OpenAI Client Initialization Error: {e}")
    
    def _get_openai_key(self, provided_key: Optional[str] = None) -> Optional[str]:
        """
        Hierarchical API key retrieval strategy
        1. Explicitly provided key
        2. Environment variable
        3. Configuration file
        4. User input
        """
        # 1. Explicitly provided key
        if provided_key:
            return provided_key
        
        # 2. Environment variable
        env_key = os.environ.get('OPENAI_API_KEY')
        if env_key:
            return env_key
        
        # 3. Configuration file
        config_paths = [
            os.path.join(os.path.expanduser('~'), '.openai', 'config.json'),
            os.path.join(os.path.dirname(__file__), 'openai_config.json')
        ]
        
        for path in config_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        config = json.load(f)
                        if config.get('api_key'):
                            return config['api_key']
            except Exception:
                continue
        
        # 4. Interactive key input (optional)
        print("No OpenAI API key found through standard methods.")
        user_input = input("Would you like to manually enter your OpenAI API key? (y/n): ").strip().lower()
        
        if user_input == 'y':
            manually_entered_key = input("Please enter your OpenAI API key: ").strip()
            return manually_entered_key
        
        return None
    
    def generate_prompt_insights(
        self, 
        data: pd.DataFrame, 
        prompt_template: Optional[str] = None,
        max_tokens: int = 300,
        model: str = "gpt-3.5-turbo"
    ) -> Optional[str]:
        """
        Flexible LLM insight generation with robust error handling
        
        :param data: Pandas DataFrame to analyze
        :param prompt_template: Custom prompt template (optional)
        :param max_tokens: Maximum tokens for response
        :param model: OpenAI model to use
        :return: Generated insights or None
        """
        # Validate client initialization
        if not self.client:
            print("OpenAI client not initialized. Skipping LLM insights.")
            return None
        
        # Default prompt if not provided
        if not prompt_template:
            prompt_template = """
            Analyze the following dataset and provide professional insights:
            
            Dataset Overview:
            - Total Columns: {column_count}
            - Total Rows: {row_count}
            - Columns: {columns}
            
            Provide actionable insights, potential patterns, and recommendations 
            for data analysis in under 250 words.
            """
        
        # Prepare prompt with dataset details
        formatted_prompt = prompt_template.format(
            column_count=len(data.columns),
            row_count=len(data),
            columns=', '.join(data.columns)
        )
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst."},
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=max_tokens
            )
            
            # Extract and return insight
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"LLM Insight Generation Error: {e}")
            return None
    
    def validate_api_key(self) -> bool:
        """
        Perform a lightweight validation of the OpenAI API key
        
        :return: Boolean indicating key validity
        """
        if not self.client:
            return False
        
        try:
            # Minimal API call to validate key
            self.client.models.list(limit=1)
            return True
        except Exception:
            return False

# Example usage in main script
def main():
    # Initialize LLM Integration
    llm_helper = LLMIntegration()
    
    # Validate API Key
    if not llm_helper.validate_api_key():
        print("Invalid or missing OpenAI API key. Some features will be limited.")
    
    # Your existing data analysis code here...


# Now import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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
