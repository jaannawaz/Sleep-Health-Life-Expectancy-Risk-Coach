"""
WHO Life Expectancy Data Integration Module
Provides population-level health context for individual sleep health predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../config'))
try:
    from config import *
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

class WHOHealthContext:
    """
    WHO Health Context Provider for Sleep Health Risk Calibration
    
    Integrates WHO Life Expectancy data to provide population-level health context
    for individual sleep health predictions and risk benchmarking.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize WHO Health Context with data loading"""
        
        self.data_path = data_path or PROCESSED_DATA_DIR / 'who_life_expectancy_key.csv'
        self.who_data = None
        self.country_profiles = {}
        self.health_indicators = []  # Will be set in _load_data
        
        # Load and prepare data
        self._load_data()
        self._prepare_country_profiles()
        
    def _load_data(self):
        """Load and validate WHO dataset"""
        
        try:
            self.who_data = pd.read_csv(self.data_path)
            print(f"✓ Loaded WHO data: {self.who_data.shape}")
            
            # Update health indicators to match actual column names
            self.health_indicators = [
                'Life expectancy ', 'Adult Mortality', ' BMI ', 
                'Income composition of resources', 'Schooling', 'GDP', 'Population'
            ]
            
            # Handle missing values in key indicators
            for indicator in self.health_indicators:
                if indicator in self.who_data.columns:
                    # Fill missing values with country median
                    country_medians = self.who_data.groupby('Country')[indicator].median()
                    self.who_data[indicator] = self.who_data.groupby('Country')[indicator].transform(
                        lambda x: x.fillna(country_medians[x.name])
                    )
            
            print(f"✓ Data cleaning completed")
            
        except Exception as e:
            print(f"❌ Error loading WHO data: {e}")
            raise
    
    def _prepare_country_profiles(self):
        """Create comprehensive country health profiles"""
        
        if self.who_data is None:
            return
        
        latest_year = self.who_data['Year'].max()
        latest_data = self.who_data[self.who_data['Year'] == latest_year].copy()
        
        print(f"✓ Creating country profiles for {latest_year} (latest year)")
        
        for _, row in latest_data.iterrows():
            country = row['Country']
            
            # Create comprehensive country profile
            profile = {
                'country': country,
                'year': int(row['Year']),
                'status': row['Status'],
                'health_indicators': {},
                'risk_factors': {},
                'benchmarks': {}
            }
            
            # Health indicators
            for indicator in self.health_indicators:
                if indicator in row and pd.notna(row[indicator]):
                    clean_name = indicator.strip()
                    profile['health_indicators'][clean_name] = float(row[indicator])
            
            # Calculate risk factors and benchmarks
            profile['risk_factors'] = self._calculate_risk_factors(row)
            profile['benchmarks'] = self._calculate_benchmarks(row, latest_data)
            
            self.country_profiles[country] = profile
        
        print(f"✓ Created {len(self.country_profiles)} country profiles")
    
    def _calculate_risk_factors(self, country_row: pd.Series) -> Dict:
        """Calculate country-specific risk factors"""
        
        risk_factors = {}
        
        # BMI risk assessment
        bmi_col = ' BMI ' if ' BMI ' in country_row else 'BMI'
        if bmi_col in country_row and pd.notna(country_row[bmi_col]):
            bmi = float(country_row[bmi_col])
            risk_factors['bmi_level'] = self._categorize_population_bmi(bmi)
            risk_factors['obesity_risk'] = 'High' if bmi > 25 else 'Medium' if bmi > 20 else 'Low'
        
        # Mortality risk
        if 'Adult Mortality' in country_row and pd.notna(country_row['Adult Mortality']):
            mortality = float(country_row['Adult Mortality'])
            risk_factors['mortality_level'] = 'High' if mortality > 200 else 'Medium' if mortality > 100 else 'Low'
        
        # Socioeconomic factors
        if 'Income composition of resources' in country_row and pd.notna(country_row['Income composition of resources']):
            income_comp = float(country_row['Income composition of resources'])
            risk_factors['socioeconomic_level'] = 'High' if income_comp > 0.7 else 'Medium' if income_comp > 0.5 else 'Low'
        
        return risk_factors
    
    def _calculate_benchmarks(self, country_row: pd.Series, all_countries: pd.DataFrame) -> Dict:
        """Calculate country benchmarks against global averages"""
        
        benchmarks = {}
        
        for indicator in self.health_indicators:
            if indicator in country_row and pd.notna(country_row[indicator]):
                country_value = float(country_row[indicator])
                global_mean = all_countries[indicator].mean()
                global_std = all_countries[indicator].std()
                
                # Calculate percentile rank
                percentile = (all_countries[indicator] <= country_value).mean() * 100
                
                # Calculate z-score
                z_score = (country_value - global_mean) / global_std if global_std > 0 else 0
                
                benchmarks[indicator.strip()] = {
                    'value': country_value,
                    'global_mean': global_mean,
                    'percentile': percentile,
                    'z_score': z_score,
                    'relative_position': 'Above' if country_value > global_mean else 'Below'
                }
        
        return benchmarks
    
    def _categorize_population_bmi(self, bmi: float) -> str:
        """Categorize population-level BMI"""
        if bmi < 18.5:
            return 'Underweight Population'
        elif bmi < 25:
            return 'Normal Weight Population'
        elif bmi < 30:
            return 'Overweight Population'
        else:
            return 'Obese Population'
    
    def get_country_context(self, country: str, year: Optional[int] = None) -> Optional[Dict]:
        """
        Get comprehensive health context for a specific country
        
        Args:
            country: Country name (must match WHO dataset naming)
            year: Optional year (defaults to latest available)
            
        Returns:
            Dictionary with country health context or None if not found
        """
        
        if country in self.country_profiles:
            return self.country_profiles[country].copy()
        
        # If not in profiles, try to find in raw data
        if year:
            country_data = self.who_data[
                (self.who_data['Country'] == country) & 
                (self.who_data['Year'] == year)
            ]
        else:
            country_data = self.who_data[self.who_data['Country'] == country]
            if not country_data.empty:
                country_data = country_data[country_data['Year'] == country_data['Year'].max()]
        
        if country_data.empty:
            return None
        
        row = country_data.iloc[0]
        latest_data = self.who_data[self.who_data['Year'] == row['Year']]
        
        # Create temporary profile
        profile = {
            'country': country,
            'year': int(row['Year']),
            'status': row['Status'],
            'health_indicators': {},
            'risk_factors': self._calculate_risk_factors(row),
            'benchmarks': self._calculate_benchmarks(row, latest_data)
        }
        
        for indicator in self.health_indicators:
            if indicator in row and pd.notna(row[indicator]):
                profile['health_indicators'][indicator.strip()] = float(row[indicator])
        
        return profile
    
    def get_available_countries(self) -> List[str]:
        """Get list of all available countries"""
        return list(self.country_profiles.keys())
    
    def get_major_countries(self) -> List[str]:
        """Get list of major countries with complete data"""
        major_countries = [
            'United States of America', 'United Kingdom', 'Germany', 'Japan',
            'Australia', 'Canada', 'France', 'Italy', 'Spain', 'Netherlands',
            'Sweden', 'Norway', 'Denmark', 'Finland', 'Switzerland'
        ]
        
        return [country for country in major_countries if country in self.country_profiles]
    
    def compare_countries(self, countries: List[str], indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare health indicators across multiple countries
        
        Args:
            countries: List of country names
            indicators: Optional list of indicators to compare (defaults to all)
            
        Returns:
            DataFrame with comparison data
        """
        
        if indicators is None:
            indicators = self.health_indicators
        
        comparison_data = []
        
        for country in countries:
            context = self.get_country_context(country)
            if context:
                row = {'Country': country, 'Status': context['status']}
                
                for indicator in indicators:
                    clean_indicator = indicator.strip()
                    if clean_indicator in context['health_indicators']:
                        row[clean_indicator] = context['health_indicators'][clean_indicator]
                    else:
                        row[clean_indicator] = np.nan
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def benchmark_individual_against_population(self, 
                                               individual_data: Dict, 
                                               country: str) -> Dict:
        """
        Benchmark individual health data against country population
        
        Args:
            individual_data: Dictionary with individual health metrics
            country: Country for population benchmarking
            
        Returns:
            Dictionary with benchmarking results
        """
        
        context = self.get_country_context(country)
        if not context:
            return {'error': f'Country {country} not found'}
        
        benchmark_results = {
            'country': country,
            'individual_vs_population': {},
            'risk_assessment': {},
            'recommendations': []
        }
        
        # BMI benchmarking
        bmi_key = 'BMI' if 'BMI' in context['health_indicators'] else ' BMI '
        if 'bmi_numeric' in individual_data and bmi_key in context['health_indicators']:
            individual_bmi = individual_data['bmi_numeric']
            population_bmi = context['health_indicators'][bmi_key]
            
            benchmark_results['individual_vs_population']['BMI'] = {
                'individual': individual_bmi,
                'population_average': population_bmi,
                'difference': individual_bmi - population_bmi,
                'relative_risk': 'Higher' if individual_bmi > population_bmi else 'Lower'
            }
            
            # Risk assessment based on population context
            if individual_bmi > population_bmi + 5:
                benchmark_results['risk_assessment']['BMI'] = 'Significantly above population average'
                benchmark_results['recommendations'].append('Consider weight management given country context')
            elif individual_bmi > population_bmi:
                benchmark_results['risk_assessment']['BMI'] = 'Above population average'
        
        # Age-related benchmarking using life expectancy
        life_exp_key = 'Life expectancy' if 'Life expectancy' in context['health_indicators'] else 'Life expectancy '
        if 'age' in individual_data and life_exp_key in context['health_indicators']:
            individual_age = individual_data['age']
            life_expectancy = context['health_indicators'][life_exp_key]
            
            age_ratio = individual_age / life_expectancy
            benchmark_results['individual_vs_population']['Age_Risk'] = {
                'age': individual_age,
                'life_expectancy': life_expectancy,
                'age_ratio': age_ratio,
                'life_stage': 'Early' if age_ratio < 0.3 else 'Middle' if age_ratio < 0.6 else 'Later'
            }
        
        # Socioeconomic context
        if 'Income composition of resources' in context['health_indicators']:
            income_score = context['health_indicators']['Income composition of resources']
            if income_score < 0.5:
                benchmark_results['recommendations'].append('Consider socioeconomic health factors in country context')
        
        return benchmark_results
    
    def get_health_trends(self, country: str, years: Optional[int] = 10) -> Dict:
        """
        Get health trends for a country over specified years
        
        Args:
            country: Country name
            years: Number of years to analyze (from latest available)
            
        Returns:
            Dictionary with trend analysis
        """
        
        country_data = self.who_data[self.who_data['Country'] == country].copy()
        if country_data.empty:
            return {'error': f'No data available for {country}'}
        
        # Get recent years
        latest_year = country_data['Year'].max()
        start_year = max(latest_year - years + 1, country_data['Year'].min())
        trend_data = country_data[country_data['Year'] >= start_year].sort_values('Year')
        
        trends = {
            'country': country,
            'period': f'{start_year}-{latest_year}',
            'indicator_trends': {},
            'overall_health_direction': None
        }
        
        positive_trends = 0
        total_trends = 0
        
        for indicator in self.health_indicators:
            if indicator in trend_data.columns and trend_data[indicator].notna().sum() >= 2:
                values = trend_data[indicator].dropna()
                if len(values) >= 2:
                    # Calculate trend (simple linear)
                    years_list = trend_data[trend_data[indicator].notna()]['Year'].tolist()
                    correlation = np.corrcoef(years_list, values)[0, 1] if len(values) > 1 else 0
                    
                    change = values.iloc[-1] - values.iloc[0]
                    percent_change = (change / values.iloc[0]) * 100 if values.iloc[0] != 0 else 0
                    
                    # Determine if trend is positive for health
                    if indicator in ['Life expectancy ', 'Income composition of resources', 'Schooling']:
                        is_positive = change > 0
                    elif indicator in ['Adult Mortality']:
                        is_positive = change < 0  # Lower mortality is better
                    else:
                        is_positive = abs(correlation) < 0.5  # Stable is good for BMI, etc.
                    
                    trends['indicator_trends'][indicator.strip()] = {
                        'start_value': float(values.iloc[0]),
                        'end_value': float(values.iloc[-1]),
                        'change': float(change),
                        'percent_change': float(percent_change),
                        'direction': 'Improving' if is_positive else 'Declining',
                        'correlation': float(correlation)
                    }
                    
                    if is_positive:
                        positive_trends += 1
                    total_trends += 1
        
        # Overall health direction
        if total_trends > 0:
            positive_ratio = positive_trends / total_trends
            if positive_ratio >= 0.7:
                trends['overall_health_direction'] = 'Improving'
            elif positive_ratio >= 0.3:
                trends['overall_health_direction'] = 'Mixed'
            else:
                trends['overall_health_direction'] = 'Declining'
        
        return trends
    
    def harmonize_bmi_categories(self, sleep_bmi_category: str) -> Dict:
        """
        Harmonize BMI categories between sleep dataset and WHO data
        
        Args:
            sleep_bmi_category: BMI category from sleep dataset ('Normal', 'Overweight', 'Obese')
            
        Returns:
            Dictionary with BMI harmonization and WHO comparison data
        """
        
        # BMI category mapping to numeric ranges
        bmi_ranges = {
            'Normal': (18.5, 24.9),
            'Overweight': (25.0, 29.9), 
            'Obese': (30.0, 40.0)
        }
        
        # Get representative BMI value for WHO comparison
        if sleep_bmi_category in bmi_ranges:
            bmi_range = bmi_ranges[sleep_bmi_category]
            representative_bmi = (bmi_range[0] + bmi_range[1]) / 2
        else:
            representative_bmi = 22.0  # Default normal BMI
        
        # Global BMI statistics from WHO data
        latest_who = self.who_data[self.who_data['Year'] == self.who_data['Year'].max()]
        bmi_column = ' BMI '  # Use the correct column name with spaces
        global_bmi_stats = {
            'mean': float(latest_who[bmi_column].mean()),
            'std': float(latest_who[bmi_column].std()),
            'percentiles': {
                '25th': float(latest_who[bmi_column].quantile(0.25)),
                '50th': float(latest_who[bmi_column].quantile(0.50)),
                '75th': float(latest_who[bmi_column].quantile(0.75)),
                '90th': float(latest_who[bmi_column].quantile(0.90))
            }
        }
        
        # Calculate where individual falls in global distribution
        percentile_rank = (latest_who[bmi_column] <= representative_bmi).mean() * 100
        
        harmonization = {
            'sleep_category': sleep_bmi_category,
            'representative_bmi': representative_bmi,
            'who_global_stats': global_bmi_stats,
            'global_percentile': float(percentile_rank),
            'risk_level': self._assess_global_bmi_risk(representative_bmi, global_bmi_stats),
            'countries_with_similar_bmi': self._find_countries_with_similar_bmi(representative_bmi)
        }
        
        return harmonization
    
    def _assess_global_bmi_risk(self, individual_bmi: float, global_stats: Dict) -> str:
        """Assess BMI risk in global context"""
        
        global_mean = global_stats['mean']
        global_std = global_stats['std']
        
        z_score = (individual_bmi - global_mean) / global_std
        
        if z_score > 2:
            return 'Very High (significantly above global average)'
        elif z_score > 1:
            return 'High (above global average)'
        elif z_score > -1:
            return 'Moderate (near global average)'
        else:
            return 'Low (below global average)'
    
    def _find_countries_with_similar_bmi(self, target_bmi: float, tolerance: float = 2.0) -> List[str]:
        """Find countries with similar population BMI"""
        
        latest_who = self.who_data[self.who_data['Year'] == self.who_data['Year'].max()]
        bmi_column = ' BMI '  # Use the correct column name with spaces
        similar_countries = latest_who[
            abs(latest_who[bmi_column] - target_bmi) <= tolerance
        ]['Country'].tolist()
        
        return similar_countries[:10]  # Return top 10 similar countries


def main():
    """Test WHO integration functionality"""
    
    print("🌍 Testing WHO Health Context Integration")
    print("=" * 60)
    
    # Initialize WHO context
    who_context = WHOHealthContext()
    
    # Test major countries
    major_countries = who_context.get_major_countries()
    print(f"✓ Major countries available: {len(major_countries)}")
    print(f"  Countries: {major_countries[:5]}...")
    
    # Test country context
    test_country = 'United States of America'
    context = who_context.get_country_context(test_country)
    
    if context:
        print(f"\\n📊 {test_country} Health Context:")
        print(f"  Status: {context['status']}")
        print(f"  Year: {context['year']}")
        
        health_indicators = context['health_indicators']
        life_exp = health_indicators.get('Life expectancy', 'N/A')
        mortality = health_indicators.get('Adult Mortality', 'N/A')
        bmi = health_indicators.get('BMI', 'N/A')
        
        print(f"  Life Expectancy: {life_exp:.1f}" if isinstance(life_exp, (int, float)) else f"  Life Expectancy: {life_exp}")
        print(f"  Adult Mortality: {mortality:.0f}" if isinstance(mortality, (int, float)) else f"  Adult Mortality: {mortality}")
        print(f"  BMI: {bmi:.1f}" if isinstance(bmi, (int, float)) else f"  BMI: {bmi}")
    
    # Test BMI harmonization
    print(f"\\n🔄 BMI Harmonization Test:")
    bmi_harmony = who_context.harmonize_bmi_categories('Overweight')
    print(f"  Category: {bmi_harmony['sleep_category']}")
    print(f"  Representative BMI: {bmi_harmony['representative_bmi']:.1f}")
    print(f"  Global Percentile: {bmi_harmony['global_percentile']:.1f}%")
    print(f"  Risk Level: {bmi_harmony['risk_level']}")
    
    # Test individual benchmarking
    print(f"\\n⚖️ Individual Benchmarking Test:")
    individual_data = {
        'bmi_numeric': 28.5,
        'age': 35
    }
    
    benchmark = who_context.benchmark_individual_against_population(
        individual_data, test_country
    )
    
    if 'individual_vs_population' in benchmark:
        bmi_comparison = benchmark['individual_vs_population'].get('BMI', {})
        print(f"  Individual BMI: {bmi_comparison.get('individual', 'N/A')}")
        print(f"  Population BMI: {bmi_comparison.get('population_average', 'N/A'):.1f}")
        print(f"  Relative Risk: {bmi_comparison.get('relative_risk', 'N/A')}")
    
    # Test health trends
    print(f"\\n📈 Health Trends Test:")
    trends = who_context.get_health_trends(test_country, years=15)
    
    if 'overall_health_direction' in trends:
        print(f"  Period: {trends['period']}")
        print(f"  Overall Direction: {trends['overall_health_direction']}")
        
        if 'Life expectancy' in trends['indicator_trends']:
            life_trend = trends['indicator_trends']['Life expectancy']
            print(f"  Life Expectancy Change: {life_trend['percent_change']:+.1f}%")
    
    print(f"\\n✅ WHO Integration Testing Complete!")
    return who_context


if __name__ == "__main__":
    who_context = main()
