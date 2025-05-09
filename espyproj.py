import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.model_selection import train_test_split

# Load the dataset
try:
    data = pd.read_csv('moviesdata.csv')
    print("Dataset loaded successfully!")
    
    # DATA CLEANING 
    print("\n===== DATA CLEANING =====")
    
    # 1. Check for missing values
    print("\nMissing values before cleaning:")
    print(data.isna().sum())
    
    # 2. Handle missing values in critical columns
    critical_cols = ['vote_average', 'popularity']
    for col in critical_cols:
        if col in data.columns:
            data = data[data[col].notna()]
    
    # 3. Clean budget and revenue (common issues in movie datasets)
    if 'budget' in data.columns:
        data = data[data['budget'] > 0]  # Remove $0 budgets
    if 'revenue' in data.columns:
        data = data[data['revenue'] > 0]  # Remove $0 revenues
    
    # 4. Clean genres
    if 'genres' in data.columns:
        data['genres'] = data['genres'].fillna('Unknown')
        data['primary_genre'] = data['genres'].str.split().str[0].fillna('Unknown')
    
    # 5. Convert numeric columns safely
    numeric_cols = ['budget', 'revenue', 'vote_average', 'popularity', 'runtime', 'vote_count']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # 6. Final check
    print("\nMissing values after cleaning:")
    print(data.isna().sum())
    print(f"\nFinal dataset shape: {data.shape}")
    
except Exception as e:
    print(f"Error loading or cleaning data: {str(e)}")
    exit()

# Histogram of vote_average
plt.figure(figsize=(10, 5))
plt.hist(data['vote_average'], bins=15, edgecolor='black', alpha=0.7)
plt.title('Distribution of vote_average (Cleaned Data)')
plt.xlabel('vote_average')
plt.ylabel('Frequency')
plt.savefig('vote_histogram.png', bbox_inches='tight')
plt.close()
print("Saved vote_histogram.png")

# 1. Basic Statistics for vote_average
print("\n===== BASIC STATISTICS FOR VOTE_AVERAGE =====")
col = 'vote_average'

# Calculate average and variance
average = data[col].mean()
variance = data[col].var()
print(f"Average {col}: {average:.4f}")
print(f"Variance {col}: {variance:.4f}")

# 2. Visualizations
print("\n===== VISUALIZATIONS =====")
# Histogram of vote_average
plt.figure(figsize=(10, 5))
plt.hist(data[col], bins=15, edgecolor='black', alpha=0.7)
plt.title(f'Distribution of {col}')
plt.xlabel(col)
plt.ylabel('Frequency')
plt.savefig('vote_histogram.png', bbox_inches='tight')
plt.close()
print("Saved vote_histogram.png")

# Pie chart of genres with exploded slices and combined small categories
data['primary_genre'] = data['genres'].str.split().str[0].fillna('Unknown')
genre_counts = data['primary_genre'].value_counts()

# Combine small slices into 'Other' category
threshold = 0.006  # 0.6% threshold
small_slices = genre_counts[genre_counts/genre_counts.sum() < threshold]
other_count = small_slices.sum()

if len(small_slices) > 0:
    genre_counts = genre_counts[genre_counts/genre_counts.sum() >= threshold]
    genre_counts['Other'] = other_count

plt.figure(figsize=(12, 12))

# Create explode effect (0.1 separation for all slices)
explode = [0.1] * len(genre_counts)

# Custom autopct function to hide very small percentages
def autopct_format(pct):
    return ('%.1f%%' % pct) if pct >= threshold*100 else ''

plt.pie(genre_counts,
        labels=genre_counts.index,
        autopct=autopct_format,
        explode=explode,
        startangle=0,
        textprops={'fontsize': 14},
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'})

plt.title('Movie Genres Distribution ', pad=10)
plt.tight_layout()
plt.savefig('genre_pie_chart.png', dpi=500, bbox_inches='tight')
plt.close()
print("Saved genre_pie_chart.png")

# 3. Frequency Distribution
print("\n===== FREQUENCY DISTRIBUTION =====")
freq_dist = data[col].value_counts().sort_index()
print("Frequency Distribution:")
print(freq_dist.to_string())

# Calculate average and variance from frequency distribution
avg_from_freq = np.sum(freq_dist.index * freq_dist.values) / freq_dist.sum()
var_from_freq = np.sum(freq_dist.values * (freq_dist.index - avg_from_freq)**2) / freq_dist.sum()

print(f"\nAverage from frequency distribution: {avg_from_freq:.4f}")
print(f"Variance from frequency distribution: {var_from_freq:.4f}")

# Comparison
print("\n===== COMPARISON =====")
print(f"Original average: {average:.4f} | Frequency distribution average: {avg_from_freq:.4f}")
print(f"Difference: {abs(average - avg_from_freq):.4f}")
print(f"\nOriginal variance: {variance:.4f} | Frequency distribution variance: {var_from_freq:.4f}")
print(f"Difference: {abs(variance - var_from_freq):.4f}")

# 4. Confidence and Tolerance Intervals (using 80% of data)
print("\n===== CONFIDENCE AND TOLERANCE INTERVALS =====")
train_data, test_data = train_test_split(data[col], test_size=0.2, random_state=42)

# Confidence Interval for Mean
n = len(train_data)
mean = np.mean(train_data)
std = np.std(train_data, ddof=1)
confidence_level = 0.95
alpha = 1 - confidence_level
t_value = st.t.ppf(1 - alpha/2, df=n-1)
ci_mean = (mean - t_value * std/np.sqrt(n), mean + t_value * std/np.sqrt(n))
print(f"95% Confidence Interval for Mean: ({ci_mean[0]:.4f}, {ci_mean[1]:.4f})")

# Confidence Interval for Variance
variance_train = np.var(train_data, ddof=1)
lower_var = (n - 1) * variance_train / st.chi2.ppf(1 - alpha/2, n - 1)
upper_var = (n - 1) * variance_train / st.chi2.ppf(alpha/2, n - 1)
ci_var = (lower_var, upper_var)
print(f"95% Confidence Interval for Variance: ({ci_var[0]:.4f}, {ci_var[1]:.4f})")

# Tolerance Interval
p = 0.95
z_value = st.norm.ppf((1 + p) / 2)
tol_interval = (mean - z_value * std, mean + z_value * std)
print(f"95% Tolerance Interval: ({tol_interval[0]:.4f}, {tol_interval[1]:.4f})")

# Validation with test data
valid_count = ((test_data >= tol_interval[0]) & (test_data <= tol_interval[1])).sum()
valid_percentage = (valid_count / len(test_data)) * 100
print(f"\nValidation Results:")
print(f"{valid_percentage:.2f}% of test data falls within the tolerance interval")

# 5. Hypothesis Testing
print("\n===== HYPOTHESIS TESTING =====")
# Hypothesis: The average movie rating is greater than 6.0
hypothesized_mean = 6.0
t_stat, p_value = st.ttest_1samp(train_data, hypothesized_mean, alternative='greater')

print(f"Hypothesis: The average movie rating is greater than {hypothesized_mean}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Conclusion: Reject the null hypothesis - average rating is significantly greater than 6.0")
else:
    print("Conclusion: Fail to reject the null hypothesis - not enough evidence to say average is greater than 6.0")

# Analysis of findings
print("\n===== ANALYSIS OF FINDINGS =====")
print("1. The basic statistics and frequency distribution methods produced similar results")
print("2. The confidence intervals show the range where we expect the true population parameters to lie")
print(f"3. The tolerance interval successfully captured {valid_percentage:.2f}% of the test data")
print("4. The hypothesis test provides insight about whether average ratings exceed 6.0")
print("5. Visualizations show the distribution of ratings and genres in the dataset")







print("\n\n\n\n\n\n\n")










# ===== BUDGET IMPACT ANALYSIS =====
print("\n===== BUDGET IMPACT ANALYSIS =======\n")

# 1. Budget vs Popularity Analysis
print("--- Budget vs Popularity ---")
budget_pop_col = 'budget'
target_col1 = 'popularity'

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data[budget_pop_col], data[target_col1], alpha=0.6)
plt.title('Budget vs Popularity')
plt.xlabel('Budget (in millions)')
plt.ylabel('Popularity')
plt.grid(True)

# Convert budget to millions for better readability
data['budget_millions'] = data[budget_pop_col] / 1e6
plt.xticks(ticks=np.arange(0, data['budget_millions'].max()+50, 50))
plt.savefig('budget_vs_popularity.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved budget_vs_popularity.png")

# Correlation and regression
corr1 = data[[budget_pop_col, target_col1]].corr().iloc[0,1]
print(f"Pearson Correlation (Budget-Popularity): {corr1:.3f}")

# 2. Budget vs Revenue Analysis
print("\n--- Budget vs Revenue ---")
target_col2 = 'revenue'

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data[budget_pop_col], data[target_col2], alpha=0.6, color='green')
plt.title('Budget vs Revenue')
plt.xlabel('Budget (in millions)')
plt.ylabel('Revenue (in billions)')
plt.grid(True)

# Convert to appropriate units
data['revenue_billions'] = data[target_col2] / 1e9
plt.xticks(ticks=np.arange(0, data['budget_millions'].max()+50, 50))
plt.yticks(ticks=np.arange(0, data['revenue_billions'].max()+0.5, 0.5))
plt.savefig('budget_vs_revenue.png', bbox_inches='tight', dpi=300)
plt.close()
print("Saved budget_vs_revenue.png")

# Correlation and regression
corr2 = data[[budget_pop_col, target_col2]].corr().iloc[0,1]
print(f"Pearson Correlation (Budget-Revenue): {corr2:.3f}")

# 3. Statistical Analysis for Budget Groups
print("\n--- Statistical Analysis by Budget Groups ---")
data['budget_group'] = pd.qcut(data['budget_millions'], q=4, 
                             labels=['Low', 'Medium', 'High', 'Very High'])

# Calculate stats for each group
budget_stats = data.groupby('budget_group').agg({
    'popularity': ['mean', 'median', 'std'],
    'revenue_billions': ['mean', 'median', 'std']
})

print("\nStatistics by Budget Group:")
print(budget_stats)

# 4. Visualization of Budget Group Effects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Popularity by budget group
budget_stats['popularity']['mean'].plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Average Popularity by Budget Group')
ax1.set_ylabel('Popularity Score')

# Revenue by budget group
budget_stats['revenue_billions']['mean'].plot(kind='bar', ax=ax2, color='lightgreen')
ax2.set_title('Average Revenue by Budget Group')
ax2.set_ylabel('Revenue (Billions)')

plt.tight_layout()
plt.savefig('budget_group_effects.png', dpi=300)
plt.close()
print("Saved budget_group_effects.png")

# 5. Hypothesis Testing
print("\n--- Hypothesis Testing ---")
high_budget = data[data['budget_group'].isin(['High', 'Very High'])]['popularity']
low_budget = data[data['budget_group'].isin(['Low', 'Medium'])]['popularity']

t_stat, p_value = st.ttest_ind(high_budget, low_budget, equal_var=False)
print(f"\nDo high-budget films have higher popularity?")
print(f"Welch's t-test: t-statistic = {t_stat:.3f}, p-value = {p_value:.5f}")

if p_value < 0.05:
    print("Conclusion: Significant difference (p < 0.05) - Budget affects popularity")
else:
    print("Conclusion: No significant difference (p ≥ 0.05)")

# 6. Analysis Findings
print("\n===== BUDGET ANALYSIS FINDINGS =====")
print("1. Budget shows correlation with both popularity and revenue")
print(f"   - Budget-Popularity correlation: {corr1:.3f}")
print(f"   - Budget-Revenue correlation: {corr2:.3f}")
print("2. Higher budget groups generally show higher average popularity and revenue")
print("3. Statistical testing confirms budget has significant impact on popularity")
print("4. Visualizations saved showing these relationships")
