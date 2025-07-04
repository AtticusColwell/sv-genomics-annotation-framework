#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import stats
import argparse
from pathlib import Path
import json

class PopulationFilter:
    def __init__(self, frequency_threshold=0.01, hardy_weinberg_p=0.001):
        self.frequency_threshold = frequency_threshold
        self.hardy_weinberg_p = hardy_weinberg_p
        self.population_data = {}
        
    def load_population_frequencies(self, freq_file):
        with open(freq_file, 'r') as f:
            self.population_data = json.load(f)
    
    def calculate_allele_frequencies(self, variants_df):
        frequencies = {}
        
        for _, variant in variants_df.iterrows():
            key = f"{variant['chrom']}:{variant['pos']}:{variant['svtype']}"
            
            if key not in frequencies:
                frequencies[key] = {
                    'total_samples': 0,
                    'alt_alleles': 0,
                    'homozygous_alt': 0,
                    'heterozygous': 0,
                    'homozygous_ref': 0
                }
            
            frequencies[key]['total_samples'] += 1
            
            if variant['support'] >= 10:
                frequencies[key]['homozygous_alt'] += 1
                frequencies[key]['alt_alleles'] += 2
            elif variant['support'] >= 3:
                frequencies[key]['heterozygous'] += 1
                frequencies[key]['alt_alleles'] += 1
            else:
                frequencies[key]['homozygous_ref'] += 1
        
        for key in frequencies:
            total_alleles = frequencies[key]['total_samples'] * 2
            if total_alleles > 0:
                frequencies[key]['af'] = frequencies[key]['alt_alleles'] / total_alleles
            else:
                frequencies[key]['af'] = 0.0
        
        return frequencies
    
    def hardy_weinberg_test(self, obs_het, obs_hom_alt, obs_hom_ref):
        total = obs_het + obs_hom_alt + obs_hom_ref
        if total == 0:
            return 1.0
        
        obs_alt_freq = (obs_hom_alt * 2 + obs_het) / (total * 2)
        obs_ref_freq = 1 - obs_alt_freq
        
        exp_hom_alt = obs_alt_freq ** 2 * total
        exp_het = 2 * obs_alt_freq * obs_ref_freq * total
        exp_hom_ref = obs_ref_freq ** 2 * total
        
        observed = [obs_hom_alt, obs_het, obs_hom_ref]
        expected = [exp_hom_alt, exp_het, exp_hom_ref]
        
        if any(e < 5 for e in expected):
            return 1.0
        
        chi2, p_value = stats.chisquare(observed, expected)
        return p_value
    
    def filter_by_frequency(self, variants_df, frequencies):
        filtered_variants = []
        
        for _, variant in variants_df.iterrows():
            key = f"{variant['chrom']}:{variant['pos']}:{variant['svtype']}"
            
            if key in frequencies:
                freq_data = frequencies[key]
                
                if freq_data['af'] < self.frequency_threshold:
                    hw_p = self.hardy_weinberg_test(
                        freq_data['heterozygous'],
                        freq_data['homozygous_alt'],
                        freq_data['homozygous_ref']
                    )
                    
                    if hw_p >= self.hardy_weinberg_p:
                        variant_dict = variant.to_dict()
                        variant_dict['population_af'] = freq_data['af']
                        variant_dict['hw_p_value'] = hw_p
                        filtered_variants.append(variant_dict)
        
        return pd.DataFrame(filtered_variants)
    
    def calculate_linkage_disequilibrium(self, variants_df, max_distance=100000):
        ld_matrix = {}
        variants_list = variants_df.to_dict('records')
        
        for i, var1 in enumerate(variants_list):
            for j, var2 in enumerate(variants_list[i+1:], i+1):
                if var1['chrom'] != var2['chrom']:
                    continue
                
                distance = abs(var1['pos'] - var2['pos'])
                if distance > max_distance:
                    continue
                
                r_squared = self.calculate_r_squared(var1, var2)
                ld_matrix[(i, j)] = {
                    'distance': distance,
                    'r_squared': r_squared
                }
        
        return ld_matrix
    
    def calculate_r_squared(self, var1, var2):
        support1 = var1['support']
        support2 = var2['support']
        
        n11 = min(support1, support2)
        n10 = support1 - n11
        n01 = support2 - n11
        n00 = max(0, 100 - support1 - support2 + n11)
        
        total = n11 + n10 + n01 + n00
        if total == 0:
            return 0.0
        
        p1 = (n11 + n10) / total
        p2 = (n11 + n01) / total
        
        if p1 == 0 or p1 == 1 or p2 == 0 or p2 == 1:
            return 0.0
        
        d = (n11 / total) - (p1 * p2)
        d_max = min(p1 * (1 - p2), (1 - p1) * p2) if d > 0 else min(p1 * p2, (1 - p1) * (1 - p2))
        
        if d_max == 0:
            return 0.0
        
        d_prime = d / d_max
        r_squared = (d ** 2) / (p1 * (1 - p1) * p2 * (1 - p2))
        
        return min(1.0, r_squared)
    
    def filter_by_linkage(self, variants_df, ld_threshold=0.8):
        ld_matrix = self.calculate_linkage_disequilibrium(variants_df)
        
        variants_to_remove = set()
        
        for (i, j), ld_data in ld_matrix.items():
            if ld_data['r_squared'] > ld_threshold:
                var1 = variants_df.iloc[i]
                var2 = variants_df.iloc[j]
                
                if var1['pathogenicity_score'] > var2['pathogenicity_score']:
                    variants_to_remove.add(j)
                else:
                    variants_to_remove.add(i)
        
        filtered_indices = [i for i in range(len(variants_df)) if i not in variants_to_remove]
        return variants_df.iloc[filtered_indices].reset_index(drop=True)
    
    def apply_quality_filters(self, variants_df):
        quality_filtered = variants_df[
            (variants_df['support'] >= 3) &
            (variants_df['svlen'] >= 50) &
            (variants_df['pathogenicity_score'] >= 0.1)
        ].copy()
        
        return quality_filtered
    
    def generate_summary_stats(self, original_df, filtered_df):
        stats = {
            'original_count': len(original_df),
            'filtered_count': len(filtered_df),
            'reduction_percentage': (1 - len(filtered_df) / len(original_df)) * 100,
            'sv_types': filtered_df['svtype'].value_counts().to_dict(),
            'mean_pathogenicity': filtered_df['pathogenicity_score'].mean(),
            'mean_support': filtered_df['support'].mean()
        }
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='Filter structural variants using population genetics')
    parser.add_argument('--input', required=True, help='Input annotated variants file')
    parser.add_argument('--output', required=True, help='Output filtered variants file')
    parser.add_argument('--freq-threshold', type=float, default=0.01, help='Frequency threshold')
    parser.add_argument('--population-freq', help='Population frequency database (JSON)')
    parser.add_argument('--stats', help='Output statistics file')
    
    args = parser.parse_args()
    
    filter_obj = PopulationFilter(frequency_threshold=args.freq_threshold)
    
    if args.population_freq:
        filter_obj.load_population_frequencies(args.population_freq)
    
    variants_df = pd.read_csv(args.input, sep='\t')
    original_count = len(variants_df)
    
    frequencies = filter_obj.calculate_allele_frequencies(variants_df)
    variants_df = filter_obj.filter_by_frequency(variants_df, frequencies)
    
    variants_df = filter_obj.apply_quality_filters(variants_df)
    variants_df = filter_obj.filter_by_linkage(variants_df)
    
    variants_df.to_csv(args.output, sep='\t', index=False)
    
    if args.stats:
        original_df = pd.read_csv(args.input, sep='\t')
        stats = filter_obj.generate_summary_stats(original_df, variants_df)
        with open(args.stats, 'w') as f:
            json.dump(stats, f, indent=2)
    
    print(f"Filtered {original_count} variants down to {len(variants_df)} high-quality variants")

if __name__ == "__main__":
    main()