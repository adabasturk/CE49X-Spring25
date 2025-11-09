"""
Lab 4: Statistical Analysis
Descriptive Statistics and Probability Distributions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson, uniform, expon, skew, kurtosis
import os
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Get the directory of this script
SCRIPT_DIR = Path(__file__).parent
DATASETS_DIR = SCRIPT_DIR.parent.parent / "datasets"


def load_data(file_path):
    """
    Load dataset from CSV file in root-level datasets/ folder.
    
    Parameters:
    -----------
    file_path : str
        Name of the CSV file (e.g., 'concrete_strength.csv')
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    try:
        full_path = DATASETS_DIR / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"Dataset not found: {full_path}")
        df = pd.read_csv(full_path)
        print(f"Loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def calculate_descriptive_stats(data, column='strength_mpa'):
    """
    Calculate all descriptive statistics.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    column : str
        Column name to analyze
        
    Returns:
    --------
    dict
        Dictionary containing all descriptive statistics
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    values = data[column].dropna()
    
    stats_dict = {
        'count': len(values),
        'mean': values.mean(),
        'median': values.median(),
        'mode': values.mode()[0] if len(values.mode()) > 0 else None,
        'std': values.std(),
        'variance': values.var(),
        'min': values.min(),
        'max': values.max(),
        'range': values.max() - values.min(),
        'q1': values.quantile(0.25),
        'q2': values.quantile(0.50),  # median
        'q3': values.quantile(0.75),
        'iqr': values.quantile(0.75) - values.quantile(0.25),
        'skewness': skew(values),
        'kurtosis': kurtosis(values, fisher=False),  # Pearson's kurtosis
        'percentile_95': values.quantile(0.95),
        'percentile_99': values.quantile(0.99)
    }
    
    return stats_dict


def plot_distribution(data, column, title, save_path=None):
    """
    Create distribution plot with statistics.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    column : str
        Column name to plot
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    values = data[column].dropna()
    mean_val = values.mean()
    median_val = values.median()
    std_val = values.std()
    mode_val = values.mode()[0] if len(values.mode()) > 0 else None
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Histogram with density curve
    ax1 = axes[0]
    n, bins, patches = ax1.hist(values, bins=30, density=True, alpha=0.7, 
                                color='skyblue', edgecolor='black')
    
    # Overlay normal distribution
    x = np.linspace(values.min(), values.max(), 100)
    y = norm.pdf(x, mean_val, std_val)
    ax1.plot(x, y, 'r-', linewidth=2, label=f'Normal(μ={mean_val:.2f}, σ={std_val:.2f})')
    
    # Mark mean, median, mode
    ax1.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax1.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    if mode_val is not None:
        ax1.axvline(mode_val, color='purple', linestyle='--', linewidth=2, label=f'Mode: {mode_val:.2f}')
    
    # Mark ±1σ, ±2σ, ±3σ
    for i, sigma in enumerate([1, 2, 3], 1):
        ax1.axvline(mean_val + sigma * std_val, color='gray', linestyle=':', 
                   alpha=0.7, label=f'μ+{i}σ' if i == 1 else '')
        ax1.axvline(mean_val - sigma * std_val, color='gray', linestyle=':', alpha=0.7)
    
    ax1.set_xlabel(column.replace('_', ' ').title())
    ax1.set_ylabel('Density')
    ax1.set_title(f'{title} - Distribution with Statistics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Boxplot
    ax2 = axes[1]
    box_plot = ax2.boxplot(values, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_ylabel(column.replace('_', ' ').title())
    ax2.set_title(f'{title} - Boxplot with Quartiles')
    ax2.grid(True, alpha=0.3)
    
    # Add quartile labels
    q1 = values.quantile(0.25)
    q2 = values.quantile(0.50)
    q3 = values.quantile(0.75)
    ax2.text(1.1, q1, f'Q1: {q1:.2f}', va='center')
    ax2.text(1.1, q2, f'Q2 (Median): {q2:.2f}', va='center')
    ax2.text(1.1, q3, f'Q3: {q3:.2f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def fit_distribution(data, column, distribution_type='normal'):
    """
    Fit probability distribution to data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    column : str
        Column name to fit
    distribution_type : str
        Type of distribution to fit ('normal', 'exponential', etc.)
        
    Returns:
    --------
    tuple
        Fitted distribution parameters and distribution object
    """
    values = data[column].dropna()
    
    if distribution_type == 'normal':
        # Fit normal distribution
        mu, sigma = norm.fit(values)
        fitted_dist = norm(loc=mu, scale=sigma)
        params = {'mean': mu, 'std': sigma}
    elif distribution_type == 'exponential':
        # Fit exponential distribution
        loc, scale = expon.fit(values, floc=0)
        fitted_dist = expon(loc=loc, scale=scale)
        params = {'mean': scale, 'lambda': 1/scale}
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
    
    return params, fitted_dist


def calculate_probability_binomial(n, p, k):
    """
    Calculate binomial probabilities.
    
    Parameters:
    -----------
    n : int
        Number of trials
    p : float
        Probability of success
    k : int or array-like
        Number of successes
        
    Returns:
    --------
    float or array
        Probability value(s)
    """
    if isinstance(k, (list, np.ndarray)):
        return binom.pmf(k, n, p)
    return binom.pmf(k, n, p)


def calculate_probability_normal(mean, std, x_lower=None, x_upper=None):
    """
    Calculate normal probabilities.
    
    Parameters:
    -----------
    mean : float
        Mean of the distribution
    std : float
        Standard deviation
    x_lower : float, optional
        Lower bound
    x_upper : float, optional
        Upper bound
        
    Returns:
    --------
    float
        Probability
    """
    if x_lower is not None and x_upper is not None:
        # Probability between two values
        return norm.cdf(x_upper, mean, std) - norm.cdf(x_lower, mean, std)
    elif x_lower is not None:
        # Probability greater than x_lower
        return 1 - norm.cdf(x_lower, mean, std)
    elif x_upper is not None:
        # Probability less than x_upper
        return norm.cdf(x_upper, mean, std)
    else:
        raise ValueError("Must specify at least one bound")


def calculate_probability_poisson(lambda_param, k):
    """
    Calculate Poisson probabilities.
    
    Parameters:
    -----------
    lambda_param : float
        Lambda parameter (mean rate)
    k : int or array-like
        Number of events
        
    Returns:
    --------
    float or array
        Probability value(s)
    """
    if isinstance(k, (list, np.ndarray)):
        return poisson.pmf(k, lambda_param)
    return poisson.pmf(k, lambda_param)


def calculate_probability_exponential(mean, x):
    """
    Calculate exponential probabilities.
    
    Parameters:
    -----------
    mean : float
        Mean of the distribution (1/lambda)
    x : float
        Value to evaluate
        
    Returns:
    --------
    float
        Probability of value less than x (CDF)
    """
    lambda_param = 1 / mean
    return expon.cdf(x, scale=mean)


def apply_bayes_theorem(prior, sensitivity, specificity):
    """
    Apply Bayes' theorem for diagnostic test scenario.
    
    Parameters:
    -----------
    prior : float
        Prior probability (base rate)
    sensitivity : float
        True positive rate (P(test+|disease+))
    specificity : float
        True negative rate (P(test-|disease-))
        
    Returns:
    --------
    dict
        Dictionary containing prior, likelihood, posterior, and related probabilities
    """
    # Calculate probabilities
    p_disease = prior
    p_no_disease = 1 - prior
    
    # Likelihoods
    p_test_positive_given_disease = sensitivity
    p_test_negative_given_disease = 1 - sensitivity
    p_test_negative_given_no_disease = specificity
    p_test_positive_given_no_disease = 1 - specificity
    
    # Joint probabilities
    p_test_positive_and_disease = p_disease * p_test_positive_given_disease
    p_test_positive_and_no_disease = p_no_disease * p_test_positive_given_no_disease
    
    # Marginal probability of positive test
    p_test_positive = p_test_positive_and_disease + p_test_positive_and_no_disease
    
    # Posterior probability (Bayes' theorem)
    p_disease_given_test_positive = p_test_positive_and_disease / p_test_positive if p_test_positive > 0 else 0
    
    results = {
        'prior': p_disease,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'p_test_positive': p_test_positive,
        'p_disease_given_test_positive': p_disease_given_test_positive,
        'p_no_disease_given_test_positive': 1 - p_disease_given_test_positive,
        'p_test_positive_given_disease': p_test_positive_given_disease,
        'p_test_positive_given_no_disease': p_test_positive_given_no_disease
    }
    
    return results


def plot_material_comparison(data, column, group_column, save_path=None):
    """
    Create comparative boxplot for material types.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    column : str
        Column name to compare
    group_column : str
        Column name for grouping
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Boxplot
    ax1 = axes[0]
    groups = data[group_column].unique()
    box_data = [data[data[group_column] == group][column].dropna() for group in groups]
    bp = ax1.boxplot(box_data, tick_labels=groups, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    ax1.set_xlabel(group_column.replace('_', ' ').title())
    ax1.set_ylabel(column.replace('_', ' ').title())
    ax1.set_title(f'Boxplot Comparison: {column.replace("_", " ").title()} by {group_column.replace("_", " ").title()}')
    ax1.grid(True, alpha=0.3)
    
    # Violin plot
    ax2 = axes[1]
    sns.violinplot(data=data, x=group_column, y=column, ax=ax2)
    ax2.set_xlabel(group_column.replace('_', ' ').title())
    ax2.set_ylabel(column.replace('_', ' ').title())
    ax2.set_title(f'Distribution Comparison: {column.replace("_", " ").title()} by {group_column.replace("_", " ").title()}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_distribution_fitting(data, column, fitted_dist=None, save_path=None):
    """
    Visualize fitted distribution with synthetic data comparison.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    column : str
        Column name to plot
    fitted_dist : scipy.stats distribution, optional
        Fitted distribution object
    save_path : str, optional
        Path to save the plot
    """
    values = data[column].dropna()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Histogram with fitted distribution
    ax1 = axes[0]
    n, bins, patches = ax1.hist(values, bins=30, density=True, alpha=0.7, 
                                color='skyblue', edgecolor='black', label='Observed Data')
    
    if fitted_dist is not None:
        # Overlay fitted distribution
        x = np.linspace(values.min(), values.max(), 100)
        y_fitted = fitted_dist.pdf(x)
        ax1.plot(x, y_fitted, 'r-', linewidth=2, label='Fitted Distribution')
        
        # Generate synthetic data from fitted distribution
        synthetic_data = fitted_dist.rvs(size=len(values))
        ax1.hist(synthetic_data, bins=30, density=True, alpha=0.5, 
                color='green', edgecolor='black', label='Synthetic Data')
    
    ax1.set_xlabel(column.replace('_', ' ').title())
    ax1.set_ylabel('Density')
    ax1.set_title(f'Distribution Fitting: {column.replace("_", " ").title()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax2 = axes[1]
    if fitted_dist is not None:
        stats.probplot(values, dist=fitted_dist, plot=ax2)
        ax2.set_title('Q-Q Plot: Observed vs Fitted Distribution')
    else:
        stats.probplot(values, dist='norm', plot=ax2)
        ax2.set_title('Q-Q Plot: Observed vs Normal Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_probability_distributions(save_path=None):
    """
    Create comprehensive plots for discrete and continuous probability distributions.
    
    Parameters:
    -----------
    save_path : str, optional
        Path to save the plot
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Discrete distributions
    # Bernoulli
    ax1 = plt.subplot(3, 3, 1)
    p_bernoulli = 0.3
    x_bernoulli = [0, 1]
    y_bernoulli = [1 - p_bernoulli, p_bernoulli]
    ax1.bar(x_bernoulli, y_bernoulli, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('x')
    ax1.set_ylabel('PMF')
    ax1.set_title(f'Bernoulli Distribution (p={p_bernoulli})')
    ax1.set_xticks([0, 1])
    ax1.grid(True, alpha=0.3)
    
    # Binomial
    ax2 = plt.subplot(3, 3, 2)
    n_binom = 20
    p_binom = 0.4
    x_binom = np.arange(0, n_binom + 1)
    y_binom = binom.pmf(x_binom, n_binom, p_binom)
    ax2.bar(x_binom, y_binom, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('k (number of successes)')
    ax2.set_ylabel('PMF')
    ax2.set_title(f'Binomial Distribution (n={n_binom}, p={p_binom})')
    ax2.grid(True, alpha=0.3)
    
    # Poisson
    ax3 = plt.subplot(3, 3, 3)
    lambda_poisson = 5
    x_poisson = np.arange(0, 20)
    y_poisson = poisson.pmf(x_poisson, lambda_poisson)
    ax3.bar(x_poisson, y_poisson, alpha=0.7, color='lightcoral', edgecolor='black')
    ax3.set_xlabel('k (number of events)')
    ax3.set_ylabel('PMF')
    ax3.set_title(f'Poisson Distribution (λ={lambda_poisson})')
    ax3.grid(True, alpha=0.3)
    
    # Continuous distributions
    # Uniform
    ax4 = plt.subplot(3, 3, 4)
    a_uniform = 0
    b_uniform = 10
    x_uniform = np.linspace(a_uniform - 1, b_uniform + 1, 100)
    y_uniform = uniform.pdf(x_uniform, a_uniform, b_uniform - a_uniform)
    ax4.plot(x_uniform, y_uniform, 'b-', linewidth=2)
    ax4.fill_between(x_uniform, y_uniform, alpha=0.3, color='blue')
    ax4.set_xlabel('x')
    ax4.set_ylabel('PDF')
    ax4.set_title(f'Uniform Distribution (a={a_uniform}, b={b_uniform})')
    ax4.grid(True, alpha=0.3)
    
    # Normal
    ax5 = plt.subplot(3, 3, 5)
    mu_normal = 0
    sigma_normal = 1
    x_normal = np.linspace(mu_normal - 4*sigma_normal, mu_normal + 4*sigma_normal, 100)
    y_normal = norm.pdf(x_normal, mu_normal, sigma_normal)
    ax5.plot(x_normal, y_normal, 'r-', linewidth=2)
    ax5.fill_between(x_normal, y_normal, alpha=0.3, color='red')
    ax5.set_xlabel('x')
    ax5.set_ylabel('PDF')
    ax5.set_title(f'Normal Distribution (μ={mu_normal}, σ={sigma_normal})')
    ax5.grid(True, alpha=0.3)
    
    # Exponential
    ax6 = plt.subplot(3, 3, 6)
    lambda_exp = 0.5
    x_exp = np.linspace(0, 10, 100)
    y_exp = expon.pdf(x_exp, scale=1/lambda_exp)
    ax6.plot(x_exp, y_exp, 'g-', linewidth=2)
    ax6.fill_between(x_exp, y_exp, alpha=0.3, color='green')
    ax6.set_xlabel('x')
    ax6.set_ylabel('PDF')
    ax6.set_title(f'Exponential Distribution (λ={lambda_exp})')
    ax6.grid(True, alpha=0.3)
    
    # CDFs
    # Binomial CDF
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(x_binom, binom.cdf(x_binom, n_binom, p_binom), 'o-', linewidth=2, markersize=4)
    ax7.set_xlabel('k')
    ax7.set_ylabel('CDF')
    ax7.set_title('Binomial CDF')
    ax7.grid(True, alpha=0.3)
    
    # Normal CDF
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(x_normal, norm.cdf(x_normal, mu_normal, sigma_normal), 'r-', linewidth=2)
    ax8.set_xlabel('x')
    ax8.set_ylabel('CDF')
    ax8.set_title('Normal CDF')
    ax8.grid(True, alpha=0.3)
    
    # Exponential CDF
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(x_exp, expon.cdf(x_exp, scale=1/lambda_exp), 'g-', linewidth=2)
    ax9.set_xlabel('x')
    ax9.set_ylabel('CDF')
    ax9.set_title('Exponential CDF')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_bayes_probability_tree(prior, sensitivity, specificity, save_path=None):
    """
    Visualize Bayes' theorem using a probability tree.
    
    Parameters:
    -----------
    prior : float
        Prior probability
    sensitivity : float
        Sensitivity
    specificity : float
        Specificity
    save_path : str, optional
        Path to save the plot
    """
    results = apply_bayes_theorem(prior, sensitivity, specificity)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Calculate probabilities
    p_disease = prior
    p_no_disease = 1 - prior
    p_test_pos_given_disease = sensitivity
    p_test_neg_given_disease = 1 - sensitivity
    p_test_neg_given_no_disease = specificity
    p_test_pos_given_no_disease = 1 - specificity
    
    # Joint probabilities
    p_disease_and_pos = p_disease * p_test_pos_given_disease
    p_disease_and_neg = p_disease * p_test_neg_given_disease
    p_no_disease_and_pos = p_no_disease * p_test_pos_given_no_disease
    p_no_disease_and_neg = p_no_disease * p_test_neg_given_no_disease
    
    # Text representation of probability tree
    tree_text = f"""
    Probability Tree for Diagnostic Test
    
    Initial State:
    ├─ Disease Present: {p_disease:.3f} ({p_disease*100:.1f}%)
    │  ├─ Test Positive: {p_test_pos_given_disease:.3f} → Joint: {p_disease_and_pos:.4f}
    │  └─ Test Negative: {p_test_neg_given_disease:.3f} → Joint: {p_disease_and_neg:.4f}
    │
    └─ No Disease: {p_no_disease:.3f} ({p_no_disease*100:.1f}%)
       ├─ Test Positive: {p_test_pos_given_no_disease:.3f} → Joint: {p_no_disease_and_pos:.4f}
       └─ Test Negative: {p_test_neg_given_no_disease:.3f} → Joint: {p_no_disease_and_neg:.4f}
    
    Marginal Probabilities:
    - P(Test Positive) = {results['p_test_positive']:.4f}
    - P(Test Negative) = {1 - results['p_test_positive']:.4f}
    
    Posterior Probability (Bayes' Theorem):
    - P(Disease | Test Positive) = {results['p_disease_given_test_positive']:.4f} ({results['p_disease_given_test_positive']*100:.2f}%)
    - P(No Disease | Test Positive) = {results['p_no_disease_given_test_positive']:.4f} ({results['p_no_disease_given_test_positive']*100:.2f}%)
    """
    
    ax.text(0.1, 0.5, tree_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title('Bayes\' Theorem: Probability Tree', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_statistical_report(data_dict, output_file='lab4_statistical_report.txt'):
    """
    Create a statistical report summarizing findings.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data and analysis results
    output_file : str
        Output file name
    """
    report_path = SCRIPT_DIR / output_file
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LAB 4: STATISTICAL ANALYSIS REPORT\n")
        f.write("Descriptive Statistics and Probability Distributions\n")
        f.write("="*80 + "\n\n")
        
        # Concrete Strength Analysis
        if 'concrete_stats' in data_dict:
            f.write("1. CONCRETE STRENGTH ANALYSIS\n")
            f.write("-"*80 + "\n")
            stats = data_dict['concrete_stats']
            f.write(f"Sample Size: {stats['count']}\n")
            f.write(f"Mean: {stats['mean']:.2f} MPa\n")
            f.write(f"Median: {stats['median']:.2f} MPa\n")
            f.write(f"Mode: {stats['mode']:.2f} MPa\n")
            f.write(f"Standard Deviation: {stats['std']:.2f} MPa\n")
            f.write(f"Variance: {stats['variance']:.2f} MPa²\n")
            f.write(f"Range: {stats['range']:.2f} MPa\n")
            f.write(f"IQR: {stats['iqr']:.2f} MPa\n")
            f.write(f"Skewness: {stats['skewness']:.3f}\n")
            f.write(f"Kurtosis: {stats['kurtosis']:.3f}\n")
            f.write(f"\nFive-Number Summary:\n")
            f.write(f"  Min: {stats['min']:.2f} MPa\n")
            f.write(f"  Q1: {stats['q1']:.2f} MPa\n")
            f.write(f"  Median (Q2): {stats['q2']:.2f} MPa\n")
            f.write(f"  Q3: {stats['q3']:.2f} MPa\n")
            f.write(f"  Max: {stats['max']:.2f} MPa\n")
            
            # Interpretation
            f.write(f"\nInterpretation:\n")
            if abs(stats['skewness']) < 0.5:
                f.write("- Distribution is approximately symmetric.\n")
            elif stats['skewness'] > 0:
                f.write("- Distribution is right-skewed (positive skewness).\n")
            else:
                f.write("- Distribution is left-skewed (negative skewness).\n")
            
            if stats['kurtosis'] < 3:
                f.write("- Distribution has lighter tails than normal (platykurtic).\n")
            elif stats['kurtosis'] > 3:
                f.write("- Distribution has heavier tails than normal (leptokurtic).\n")
            else:
                f.write("- Distribution has normal tail behavior (mesokurtic).\n")
            
            f.write(f"\nEngineering Implications:\n")
            f.write("- The mean strength provides an estimate of expected performance.\n")
            f.write("- Standard deviation indicates variability in concrete strength.\n")
            f.write("- For quality control, values outside ±2σ may require attention.\n")
            f.write(f"- 95th percentile strength: {stats['percentile_95']:.2f} MPa\n")
            f.write(f"- 99th percentile strength: {stats['percentile_99']:.2f} MPa\n")
            f.write("\n")
        
        # Material Comparison
        if 'material_stats' in data_dict:
            f.write("2. MATERIAL COMPARISON ANALYSIS\n")
            f.write("-"*80 + "\n")
            material_stats = data_dict['material_stats']
            for material, stats in material_stats.items():
                f.write(f"\n{material}:\n")
                f.write(f"  Mean: {stats['mean']:.2f} MPa\n")
                f.write(f"  Std: {stats['std']:.2f} MPa\n")
                f.write(f"  CV (Coefficient of Variation): {(stats['std']/stats['mean']*100):.2f}%\n")
            f.write("\n")
        
        # Probability Distributions
        if 'probability_results' in data_dict:
            f.write("3. PROBABILITY DISTRIBUTION APPLICATIONS\n")
            f.write("-"*80 + "\n")
            prob_results = data_dict['probability_results']
            
            # Binomial
            if 'binomial' in prob_results:
                f.write("\nBinomial Distribution (Quality Control):\n")
                f.write(f"  Scenario: {prob_results['binomial']['scenario']}\n")
                f.write(f"  P(X = 3): {prob_results['binomial']['p_exact_3']:.4f}\n")
                f.write(f"  P(X ≤ 5): {prob_results['binomial']['p_le_5']:.4f}\n")
            
            # Poisson
            if 'poisson' in prob_results:
                f.write("\nPoisson Distribution (Bridge Load Events):\n")
                f.write(f"  Scenario: {prob_results['poisson']['scenario']}\n")
                f.write(f"  P(X = 8): {prob_results['poisson']['p_exact_8']:.4f}\n")
                f.write(f"  P(X > 15): {prob_results['poisson']['p_gt_15']:.4f}\n")
            
            # Normal
            if 'normal' in prob_results:
                f.write("\nNormal Distribution (Steel Yield Strength):\n")
                f.write(f"  Scenario: {prob_results['normal']['scenario']}\n")
                f.write(f"  P(X > 280): {prob_results['normal']['p_gt_280']:.4f} ({prob_results['normal']['p_gt_280']*100:.2f}%)\n")
                f.write(f"  95th Percentile: {prob_results['normal']['percentile_95']:.2f} MPa\n")
            
            # Exponential
            if 'exponential' in prob_results:
                f.write("\nExponential Distribution (Component Lifetime):\n")
                f.write(f"  Scenario: {prob_results['exponential']['scenario']}\n")
                f.write(f"  P(X < 500): {prob_results['exponential']['p_lt_500']:.4f}\n")
                f.write(f"  P(X > 1500): {prob_results['exponential']['p_gt_1500']:.4f}\n")
            f.write("\n")
        
        # Bayes' Theorem
        if 'bayes_results' in data_dict:
            f.write("4. BAYES' THEOREM APPLICATION\n")
            f.write("-"*80 + "\n")
            bayes = data_dict['bayes_results']
            f.write(f"Scenario: Structural Damage Detection\n")
            f.write(f"Prior Probability (Base Rate): {bayes['prior']:.3f} ({bayes['prior']*100:.1f}%)\n")
            f.write(f"Test Sensitivity: {bayes['sensitivity']:.3f} ({bayes['sensitivity']*100:.1f}%)\n")
            f.write(f"Test Specificity: {bayes['specificity']:.3f} ({bayes['specificity']*100:.1f}%)\n")
            f.write(f"\nPosterior Probability:\n")
            f.write(f"  P(Damage | Test Positive) = {bayes['p_disease_given_test_positive']:.4f} ({bayes['p_disease_given_test_positive']*100:.2f}%)\n")
            f.write(f"\nInterpretation:\n")
            f.write(f"- Even with a positive test result, the probability of actual damage is {bayes['p_disease_given_test_positive']*100:.2f}%.\n")
            f.write(f"- This is due to the low base rate ({bayes['prior']*100:.1f}%) and false positive rate.\n")
            f.write(f"- Engineering decision-making should consider both test results and prior knowledge.\n")
            f.write("\n")
        
        # Distribution Fitting
        if 'fitting_results' in data_dict:
            f.write("5. DISTRIBUTION FITTING\n")
            f.write("-"*80 + "\n")
            fitting = data_dict['fitting_results']
            f.write(f"Fitted Normal Distribution Parameters:\n")
            f.write(f"  Mean (μ): {fitting['params']['mean']:.2f} MPa\n")
            f.write(f"  Standard Deviation (σ): {fitting['params']['std']:.2f} MPa\n")
            f.write(f"\nSample Statistics:\n")
            f.write(f"  Sample Mean: {fitting['sample_mean']:.2f} MPa\n")
            f.write(f"  Sample Std: {fitting['sample_std']:.2f} MPa\n")
            f.write(f"\nComparison:\n")
            f.write(f"  The fitted parameters closely match the sample statistics, ")
            f.write(f"indicating a good fit to the normal distribution.\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"Statistical report saved to {report_path}")


def create_statistical_dashboard(data, save_path=None):
    """
    Create a comprehensive statistical summary dashboard.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    save_path : str, optional
        Path to save the plot
    """
    fig = plt.figure(figsize=(16, 12))
    
    values = data['strength_mpa'].dropna()
    stats_dict = calculate_descriptive_stats(data, 'strength_mpa')
    
    # Summary statistics table
    ax1 = plt.subplot(2, 3, 1)
    ax1.axis('off')
    stats_text = f"""
    Descriptive Statistics
    
    Count: {stats_dict['count']}
    Mean: {stats_dict['mean']:.2f} MPa
    Median: {stats_dict['median']:.2f} MPa
    Std Dev: {stats_dict['std']:.2f} MPa
    Variance: {stats_dict['variance']:.2f} MPa²
    Min: {stats_dict['min']:.2f} MPa
    Max: {stats_dict['max']:.2f} MPa
    Range: {stats_dict['range']:.2f} MPa
    IQR: {stats_dict['iqr']:.2f} MPa
    Skewness: {stats_dict['skewness']:.3f}
    Kurtosis: {stats_dict['kurtosis']:.3f}
    """
    ax1.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Histogram
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(stats_dict['mean'], color='red', linestyle='--', label=f"Mean: {stats_dict['mean']:.2f}")
    ax2.axvline(stats_dict['median'], color='green', linestyle='--', label=f"Median: {stats_dict['median']:.2f}")
    ax2.set_xlabel('Strength (MPa)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution Histogram')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Boxplot
    ax3 = plt.subplot(2, 3, 3)
    ax3.boxplot(values, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightgreen', alpha=0.7))
    ax3.set_ylabel('Strength (MPa)')
    ax3.set_title('Boxplot')
    ax3.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax4 = plt.subplot(2, 3, 4)
    stats.probplot(values, dist='norm', plot=ax4)
    ax4.set_title('Q-Q Plot (Normal)')
    ax4.grid(True, alpha=0.3)
    
    # Density plot with normal overlay
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(values, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    x = np.linspace(values.min(), values.max(), 100)
    y = norm.pdf(x, stats_dict['mean'], stats_dict['std'])
    ax5.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')
    ax5.set_xlabel('Strength (MPa)')
    ax5.set_ylabel('Density')
    ax5.set_title('Density Plot with Normal Overlay')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Cumulative distribution
    ax6 = plt.subplot(2, 3, 6)
    sorted_values = np.sort(values)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax6.plot(sorted_values, cumulative, 'b-', linewidth=2, label='Empirical CDF')
    x_norm = np.linspace(values.min(), values.max(), 100)
    y_norm = norm.cdf(x_norm, stats_dict['mean'], stats_dict['std'])
    ax6.plot(x_norm, y_norm, 'r--', linewidth=2, label='Normal CDF')
    ax6.set_xlabel('Strength (MPa)')
    ax6.set_ylabel('Cumulative Probability')
    ax6.set_title('Cumulative Distribution Function')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Statistical Summary Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main execution function."""
    print("="*80)
    print("LAB 4: STATISTICAL ANALYSIS")
    print("Descriptive Statistics and Probability Distributions")
    print("="*80)
    print()
    
    # Dictionary to store results for reporting
    results = {}
    
    # Part 1: Descriptive Statistics
    print("PART 1: DESCRIPTIVE STATISTICS")
    print("-"*80)
    
    # Load concrete strength data
    print("\n1. Loading Concrete Strength Data...")
    concrete_data = load_data('concrete_strength.csv')
    print(f"   Shape: {concrete_data.shape}")
    print(f"   Columns: {list(concrete_data.columns)}")
    print(f"   Data types:\n{concrete_data.dtypes}")
    print(f"\n   First few rows:")
    print(concrete_data.head())
    print(f"\n   Missing values:\n{concrete_data.isnull().sum()}")
    print(f"\n   Summary statistics:")
    print(concrete_data.describe())
    
    # Calculate descriptive statistics
    print("\n2. Calculating Descriptive Statistics...")
    concrete_stats = calculate_descriptive_stats(concrete_data, 'strength_mpa')
    results['concrete_stats'] = concrete_stats
    
    print(f"   Mean: {concrete_stats['mean']:.2f} MPa")
    print(f"   Median: {concrete_stats['median']:.2f} MPa")
    print(f"   Mode: {concrete_stats['mode']:.2f} MPa")
    print(f"   Standard Deviation: {concrete_stats['std']:.2f} MPa")
    print(f"   Variance: {concrete_stats['variance']:.2f} MPa²")
    print(f"   Range: {concrete_stats['range']:.2f} MPa")
    print(f"   IQR: {concrete_stats['iqr']:.2f} MPa")
    print(f"   Skewness: {concrete_stats['skewness']:.3f}")
    print(f"   Kurtosis: {concrete_stats['kurtosis']:.3f}")
    print(f"\n   Five-Number Summary:")
    print(f"     Min: {concrete_stats['min']:.2f} MPa")
    print(f"     Q1: {concrete_stats['q1']:.2f} MPa")
    print(f"     Q2 (Median): {concrete_stats['q2']:.2f} MPa")
    print(f"     Q3: {concrete_stats['q3']:.2f} MPa")
    print(f"     Max: {concrete_stats['max']:.2f} MPa")
    
    # Plot distribution
    print("\n3. Creating Distribution Plots...")
    plot_distribution(concrete_data, 'strength_mpa', 'Concrete Strength',
                     save_path=SCRIPT_DIR / 'concrete_strength_distribution.png')
    
    # Material comparison
    print("\n4. Loading Material Properties Data...")
    material_data = load_data('material_properties.csv')
    print(f"   Material types: {material_data['material_type'].unique()}")
    
    print("\n5. Calculating Material Comparison Statistics...")
    material_stats = {}
    for material in material_data['material_type'].unique():
        material_subset = material_data[material_data['material_type'] == material]
        stats = calculate_descriptive_stats(material_subset, 'yield_strength_mpa')
        material_stats[material] = stats
        print(f"\n   {material}:")
        print(f"     Mean: {stats['mean']:.2f} MPa")
        print(f"     Std: {stats['std']:.2f} MPa")
        print(f"     CV: {(stats['std']/stats['mean']*100):.2f}%")
    
    results['material_stats'] = material_stats
    
    # Plot material comparison
    print("\n6. Creating Material Comparison Plot...")
    plot_material_comparison(material_data, 'yield_strength_mpa', 'material_type',
                            save_path=SCRIPT_DIR / 'material_comparison_boxplot.png')
    
    # Part 2: Probability Distributions
    print("\n\nPART 2: PROBABILITY DISTRIBUTIONS")
    print("-"*80)
    
    # Discrete distributions
    print("\n7. Working with Discrete Distributions...")
    
    # Binomial
    print("\n   Binomial Distribution:")
    n, p = 100, 0.05
    prob_exact_3 = calculate_probability_binomial(n, p, 3)
    prob_le_5 = sum([calculate_probability_binomial(n, p, k) for k in range(6)])
    print(f"     Scenario: Quality control - 100 components, 5% defect rate")
    print(f"     P(X = 3): {prob_exact_3:.4f}")
    print(f"     P(X ≤ 5): {prob_le_5:.4f}")
    
    # Poisson
    print("\n   Poisson Distribution:")
    lambda_poisson = 10
    prob_exact_8 = calculate_probability_poisson(lambda_poisson, 8)
    prob_gt_15 = 1 - sum([calculate_probability_poisson(lambda_poisson, k) for k in range(16)])
    print(f"     Scenario: Bridge load events - Average 10 heavy trucks per hour")
    print(f"     P(X = 8): {prob_exact_8:.4f}")
    print(f"     P(X > 15): {prob_gt_15:.4f}")
    
    # Store probability results
    results['probability_results'] = {
        'binomial': {
            'scenario': 'Quality control - 100 components, 5% defect rate',
            'p_exact_3': prob_exact_3,
            'p_le_5': prob_le_5
        },
        'poisson': {
            'scenario': 'Bridge load events - Average 10 heavy trucks per hour',
            'p_exact_8': prob_exact_8,
            'p_gt_15': prob_gt_15
        }
    }
    
    # Continuous distributions
    print("\n8. Working with Continuous Distributions...")
    
    # Normal
    print("\n   Normal Distribution:")
    mu_normal = 250
    sigma_normal = 15
    prob_gt_280 = calculate_probability_normal(mu_normal, sigma_normal, x_lower=280)
    percentile_95 = norm.ppf(0.95, mu_normal, sigma_normal)
    print(f"     Scenario: Steel yield strength - Mean=250 MPa, Std=15 MPa")
    print(f"     P(X > 280): {prob_gt_280:.4f} ({prob_gt_280*100:.2f}%)")
    print(f"     95th Percentile: {percentile_95:.2f} MPa")
    
    results['probability_results']['normal'] = {
        'scenario': 'Steel yield strength - Mean=250 MPa, Std=15 MPa',
        'p_gt_280': prob_gt_280,
        'percentile_95': percentile_95
    }
    
    # Exponential
    print("\n   Exponential Distribution:")
    mean_exp = 1000
    prob_lt_500 = calculate_probability_exponential(mean_exp, 500)
    prob_gt_1500 = 1 - calculate_probability_exponential(mean_exp, 1500)
    print(f"     Scenario: Component lifetime - Mean=1000 hours")
    print(f"     P(X < 500): {prob_lt_500:.4f}")
    print(f"     P(X > 1500): {prob_gt_1500:.4f}")
    
    results['probability_results']['exponential'] = {
        'scenario': 'Component lifetime - Mean=1000 hours',
        'p_lt_500': prob_lt_500,
        'p_gt_1500': prob_gt_1500
    }
    
    # Plot probability distributions
    print("\n9. Creating Probability Distribution Plots...")
    plot_probability_distributions(save_path=SCRIPT_DIR / 'probability_distributions.png')
    
    # Distribution fitting
    print("\n10. Fitting Normal Distribution to Concrete Strength Data...")
    params, fitted_dist = fit_distribution(concrete_data, 'strength_mpa', 'normal')
    print(f"     Fitted parameters:")
    print(f"       Mean (μ): {params['mean']:.2f} MPa")
    print(f"       Std (σ): {params['std']:.2f} MPa")
    print(f"     Sample statistics:")
    print(f"       Sample Mean: {concrete_stats['mean']:.2f} MPa")
    print(f"       Sample Std: {concrete_stats['std']:.2f} MPa")
    
    results['fitting_results'] = {
        'params': params,
        'sample_mean': concrete_stats['mean'],
        'sample_std': concrete_stats['std']
    }
    
    # Plot distribution fitting
    print("\n11. Creating Distribution Fitting Plot...")
    plot_distribution_fitting(concrete_data, 'strength_mpa', fitted_dist,
                             save_path=SCRIPT_DIR / 'distribution_fitting.png')
    
    # Part 3: Probability Applications
    print("\n\nPART 3: PROBABILITY APPLICATIONS")
    print("-"*80)
    
    # Bayes' theorem
    print("\n12. Applying Bayes' Theorem...")
    prior = 0.05
    sensitivity = 0.95
    specificity = 0.90
    bayes_results = apply_bayes_theorem(prior, sensitivity, specificity)
    results['bayes_results'] = bayes_results
    
    print(f"     Scenario: Structural damage detection")
    print(f"     Prior Probability (Base Rate): {prior:.3f} ({prior*100:.1f}%)")
    print(f"     Test Sensitivity: {sensitivity:.3f} ({sensitivity*100:.1f}%)")
    print(f"     Test Specificity: {specificity:.3f} ({specificity*100:.1f}%)")
    print(f"     P(Test Positive): {bayes_results['p_test_positive']:.4f}")
    print(f"     P(Damage | Test Positive): {bayes_results['p_disease_given_test_positive']:.4f} ({bayes_results['p_disease_given_test_positive']*100:.2f}%)")
    
    # Plot Bayes' probability tree
    print("\n13. Creating Bayes' Theorem Probability Tree...")
    plot_bayes_probability_tree(prior, sensitivity, specificity,
                               save_path=SCRIPT_DIR / 'bayes_probability_tree.png')
    
    # Part 4: Visualization and Reporting
    print("\n\nPART 4: VISUALIZATION AND REPORTING")
    print("-"*80)
    
    # Create statistical dashboard
    print("\n14. Creating Statistical Summary Dashboard...")
    create_statistical_dashboard(concrete_data,
                                save_path=SCRIPT_DIR / 'statistical_summary_dashboard.png')
    
    # Create statistical report
    print("\n15. Creating Statistical Report...")
    create_statistical_report(results, 'lab4_statistical_report.txt')
    
    print("\n" + "="*80)
    print("LAB 4 COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  1. concrete_strength_distribution.png")
    print("  2. material_comparison_boxplot.png")
    print("  3. probability_distributions.png")
    print("  4. distribution_fitting.png")
    print("  5. statistical_summary_dashboard.png")
    print("  6. bayes_probability_tree.png")
    print("  7. lab4_statistical_report.txt")
    print("="*80)


if __name__ == "__main__":
    main()

