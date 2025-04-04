import matplotlib.pyplot as plt
import seaborn as sns

def plot_Correlation_Heatmap(df):
    try:
        # Visualization 1: Correlation Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('reports/figures/heatmap.png')
        plt.close()

        
    except Exception as e:
        print(f"Visualization error: {e}")
        
def plot_Boxplot(df):
    try:
        # Visualization 2: Boxplot of Price by Property Type
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='property_type', y='price', data=df)
        plt.title('Price Distribution by Property Type')
        plt.ylabel('Price')
        plt.xlabel('Property Type')
        plt.tight_layout()
        plt.savefig('reports/figures/boxplot.png')
        plt.close()
        
    except Exception as e:
        print(f"Visualization error: {e}")