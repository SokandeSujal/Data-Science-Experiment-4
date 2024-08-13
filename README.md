# Data-Science-Experiment-4

## Data Preprocessing: Normalization and Standardization

### Introduction
In this experiment, we explore two crucial data preprocessing techniques: **Normalization** and **Standardization**. These methods are essential for ensuring that features in your dataset are scaled appropriately, which is critical for many machine learning algorithms.

### Concepts

- **Normalization (Min-Max Scaling)**:
  - **Formula**:  
  $$
  x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}
  $$
  
  where:
  - \( x \) is the original value,
  - \( x_{min} \) is the minimum value in the dataset,
  - \( x_{max} \) is the maximum value in the dataset.
  - **Purpose**: Rescales features to a specific range, typically [0, 1].
  - **Impact**: Changes the range of the data but preserves the distribution's shape.

- **Standardization (Z-Score Normalization)**:
  - **Formula**:
  $$
  x_{std} = \frac{x - \mu}{\sigma}
  $$
  
  where:
  - \( x \) is the original value,
  - \( \mu \) is the mean of the dataset,
  - \( \sigma \) is the standard deviation of the dataset.
  - **Purpose**: Transforms features to have a mean of 0 and a standard deviation of 1.
  - **Impact**: Centers the data but preserves the shape of the distribution.

### Steps to Reproduce

1. **Data Loading**:
   - Load the dataset from the specified URL and inspect the first few rows.

    ```python
    import pandas as pd
    df = pd.read_csv('https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv', header=None, usecols=[0,1,2])
    df.columns = ["Class", "Alcohol", "Malic"]
    df.head()
    ```

2. **Normalization**:
   - Apply Min-Max Scaling to normalize the data.

    ```python
    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler()
    df_normalized = df[['Alcohol', 'Malic']].copy()
    df_normalized[['Alcohol', 'Malic']] = scaling.fit_transform(df[['Alcohol', 'Malic']])
    ```

3. **Standardization**:
   - Apply Standardization to normalize the data to have a mean of 0 and standard deviation of 1.

    ```python
    from sklearn.preprocessing import StandardScaler
    standardization = StandardScaler()
    df_standardized = df[['Alcohol', 'Malic']].copy()
    df_standardized[['Alcohol', 'Malic']] = standardization.fit_transform(df[['Alcohol', 'Malic']])
    ```

4. **Visualization**:
   - Visualize the original, normalized, and standardized data using histograms and scatter plots.

    ```python
    import matplotlib.pyplot as plt
    # Histograms and Scatter plots are generated here
    plt.show()
    ```

### Visualization
#### Histograms
![Histogram](exp4%20ds%20histogram.png)

#### Scatter Plot
![Scatter Plot](exp4%20ds%20scatterplot.png)

### Conclusion
Normalization and standardization are essential preprocessing techniques that improve the performance of machine learning models by ensuring that features are on a comparable scale. This experiment demonstrates the impact of these techniques on the dataset.

### How to Run the Experiment
1. Clone the repository.
2. Open the Jupyter Notebook in Google Colab.
3. Run the cells step by step to reproduce the results.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Author
[Sujal Sokande](https://github.com/SokandeSujal)
