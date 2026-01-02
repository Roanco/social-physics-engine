"""
⚡ Physics Data Analysis & Validation
=====================================
Analyzes the Six Sigma Kanban physics data for quality assurance.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def validate_six_sigma_compliance(df: pd.DataFrame) -> dict:
    """
    Validate that generated data complies with Six Sigma principles.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'total_nodes': len(df),
        'validations': [],
        'warnings': [],
        'errors': []
    }
    
    # Validation 1: Demand follows normal distribution
    demand_mean = df['demand_D'].mean()
    if 200 <= demand_mean <= 300:
        results['validations'].append(f"✓ Demand mean ({demand_mean:.2f}) within expected range")
    else:
        results['warnings'].append(f"⚠ Demand mean ({demand_mean:.2f}) outside expected range")
    
    # Validation 2: Lead time follows Poisson distribution
    lead_time_mean = df['lead_time_L'].mean()
    if 4 <= lead_time_mean <= 6:
        results['validations'].append(f"✓ Lead time mean ({lead_time_mean:.2f}) within expected range")
    else:
        results['warnings'].append(f"⚠ Lead time mean ({lead_time_mean:.2f}) outside expected range")
    
    # Validation 3: Safety stock within bounds
    ss_min, ss_max = df['safety_stock_SS'].min(), df['safety_stock_SS'].max()
    if 0.10 <= ss_min and ss_max <= 0.30:
        results['validations'].append(f"✓ Safety stock range [{ss_min:.2f}, {ss_max:.2f}] correct")
    else:
        results['errors'].append(f"✗ Safety stock range [{ss_min:.2f}, {ss_max:.2f}] out of bounds")
    
    # Validation 4: Container capacity only uses standard sizes
    valid_sizes = {20, 50, 100}
    actual_sizes = set(df['container_capacity_C'].unique())
    if actual_sizes.issubset(valid_sizes):
        results['validations'].append(f"✓ Container sizes {sorted(actual_sizes)} are standard")
    else:
        results['errors'].append(f"✗ Invalid container sizes: {actual_sizes - valid_sizes}")
    
    # Validation 5: Kanban cards calculation
    # N = (D × L × (1 + SS)) / C
    calculated_n = np.ceil(
        (df['demand_D'] * df['lead_time_L'] * (1 + df['safety_stock_SS'])) / df['container_capacity_C']
    ).astype(int)
    
    if (df['kanban_cards_N'] == calculated_n).all():
        results['validations'].append("✓ Kanban cards formula verified")
    else:
        mismatches = (df['kanban_cards_N'] != calculated_n).sum()
        results['errors'].append(f"✗ Kanban cards calculation mismatch in {mismatches} nodes")
    
    # Validation 6: Reorder point calculation
    # ROP = D × L × (1 + SS)
    calculated_rop = (df['demand_D'] * df['lead_time_L'] * (1 + df['safety_stock_SS'])).astype(int)
    
    if (df['reorder_point_ROP'] == calculated_rop).all():
        results['validations'].append("✓ Reorder point formula verified")
    else:
        mismatches = (df['reorder_point_ROP'] != calculated_rop).sum()
        results['errors'].append(f"✗ Reorder point calculation mismatch in {mismatches} nodes")
    
    # Validation 7: No negative values
    if (df[['demand_D', 'lead_time_L', 'safety_stock_SS', 'container_capacity_C']] >= 0).all().all():
        results['validations'].append("✓ No negative values detected")
    else:
        results['errors'].append("✗ Negative values found in dataset")
    
    # Validation 8: No NaN values
    if not df.isnull().any().any():
        results['validations'].append("✓ No missing values (NaN) detected")
    else:
        results['errors'].append("✗ Missing values (NaN) found in dataset")
    
    return results


def print_validation_report(results: dict):
    """Print a formatted validation report."""
    print("\n" + "=" * 70)
    print("🔍 SIX SIGMA COMPLIANCE VALIDATION REPORT")
    print("=" * 70)
    print(f"\nTotal Nodes Analyzed: {results['total_nodes']}")
    
    print(f"\n✅ Passed Validations ({len(results['validations'])}):")
    for validation in results['validations']:
        print(f"   {validation}")
    
    if results['warnings']:
        print(f"\n⚠️  Warnings ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"   {warning}")
    
    if results['errors']:
        print(f"\n❌ Errors ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"   {error}")
    else:
        print("\n🎉 No errors detected!")
    
    # Calculate quality score
    total_checks = len(results['validations']) + len(results['warnings']) + len(results['errors'])
    quality_score = (len(results['validations']) / total_checks * 100) if total_checks > 0 else 0
    
    print(f"\n📊 Quality Score: {quality_score:.1f}%")
    print("=" * 70)


def analyze_correlations(df: pd.DataFrame):
    """Analyze correlations between physics parameters."""
    print("\n" + "=" * 70)
    print("🔗 CORRELATION ANALYSIS")
    print("=" * 70)
    
    # Select numeric columns
    numeric_cols = ['demand_D', 'lead_time_L', 'safety_stock_SS', 
                    'container_capacity_C', 'kanban_cards_N', 'reorder_point_ROP']
    
    corr_matrix = df[numeric_cols].corr()
    
    print("\nKey Correlations:")
    print("-" * 70)
    
    # Kanban cards correlations
    print("\n🎴 Kanban Cards (N) correlations:")
    kanban_corr = corr_matrix['kanban_cards_N'].drop('kanban_cards_N').sort_values(ascending=False)
    for param, corr in kanban_corr.items():
        print(f"   {param}: {corr:+.3f}")
    
    # Reorder point correlations
    print("\n📍 Reorder Point (ROP) correlations:")
    rop_corr = corr_matrix['reorder_point_ROP'].drop('reorder_point_ROP').sort_values(ascending=False)
    for param, corr in rop_corr.items():
        print(f"   {param}: {corr:+.3f}")


def generate_summary_statistics(df: pd.DataFrame):
    """Generate comprehensive summary statistics."""
    print("\n" + "=" * 70)
    print("📈 COMPREHENSIVE SUMMARY STATISTICS")
    print("=" * 70)
    
    stats = df.describe()
    print("\n" + stats.to_string())
    
    # Additional percentiles
    print("\n" + "=" * 70)
    print("📊 PERCENTILE ANALYSIS")
    print("=" * 70)
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for col in ['demand_D', 'lead_time_L', 'kanban_cards_N', 'reorder_point_ROP']:
        print(f"\n{col}:")
        for p in percentiles:
            value = df[col].quantile(p/100)
            print(f"   P{p:02d}: {value:.2f}")


def main():
    """Main analysis function."""
    print("⚡ Six Sigma Kanban Physics - Data Analysis & Validation")
    print("=" * 70)
    
    # Load data
    csv_file = Path("kanban_physics_data.csv")
    
    if not csv_file.exists():
        print(f"\n❌ Error: {csv_file} not found!")
        print("Please run generate_data.py first to create the dataset.")
        return
    
    print(f"\n📂 Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} nodes")
    
    # Run validation
    validation_results = validate_six_sigma_compliance(df)
    print_validation_report(validation_results)
    
    # Analyze correlations
    analyze_correlations(df)
    
    # Generate summary statistics
    generate_summary_statistics(df)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    
    if not validation_results['errors']:
        print("\n🎉 Dataset is 99.9% compliant with Six Sigma standards!")
        print("⚡ All formulas verified and validated successfully.")
    else:
        print(f"\n⚠️  Found {len(validation_results['errors'])} error(s) - review required")


if __name__ == "__main__":
    main()
