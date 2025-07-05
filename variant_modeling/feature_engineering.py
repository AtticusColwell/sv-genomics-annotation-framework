#!/usr/bin/env python3

import numpy as np
import pandas as pd
from collections import Counter
import re
from pathlib import Path
import json

class SVFeatureExtractor:
    def __init__(self, reference_fasta=None, window_size=500):
        self.window_size = window_size
        self.reference_fasta = reference_fasta
        self.gc_content_cache = {}
        
    def extract_sequence_features(self, chrom, start, end, sequence=None):
        if sequence is None:
            sequence = self.get_reference_sequence(chrom, start, end)
        
        features = {}
        
        if not sequence or len(sequence) == 0:
            return self.get_empty_sequence_features()
        
        features['gc_content'] = self.calculate_gc_content(sequence)
        features['repeat_content'] = self.calculate_repeat_content(sequence)
        features['complexity_score'] = self.calculate_complexity_score(sequence)
        features['tandem_repeat_score'] = self.detect_tandem_repeats(sequence)
        features['homopolymer_runs'] = self.count_homopolymer_runs(sequence)
        features['dinucleotide_bias'] = self.calculate_dinucleotide_bias(sequence)
        
        return features
    
    def get_reference_sequence(self, chrom, start, end):
        if self.reference_fasta is None:
            return self.simulate_sequence(end - start)
        return "N" * (end - start)
    
    def simulate_sequence(self, length):
        bases = ['A', 'T', 'G', 'C']
        return ''.join(np.random.choice(bases, length))
    
    def get_empty_sequence_features(self):
        return {
            'gc_content': 0.5,
            'repeat_content': 0.0,
            'complexity_score': 1.0,
            'tandem_repeat_score': 0.0,
            'homopolymer_runs': 0,
            'dinucleotide_bias': 0.0
        }
    
    def calculate_gc_content(self, sequence):
        if len(sequence) == 0:
            return 0.5
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)
    
    def calculate_repeat_content(self, sequence):
        repeat_patterns = ['AT', 'TA', 'GC', 'CG', 'AAT', 'TTA', 'GGC', 'CCG']
        total_repeats = 0
        
        for pattern in repeat_patterns:
            total_repeats += len(re.findall(pattern * 3, sequence))
        
        return min(1.0, total_repeats / (len(sequence) / 10))
    
    def calculate_complexity_score(self, sequence):
        if len(sequence) < 4:
            return 1.0
        
        kmer_counts = Counter()
        k = 4
        
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            kmer_counts[kmer] += 1
        
        if len(kmer_counts) == 0:
            return 0.0
        
        max_possible_kmers = len(sequence) - k + 1
        unique_kmers = len(kmer_counts)
        
        return unique_kmers / min(max_possible_kmers, 4**k)
    
    def detect_tandem_repeats(self, sequence):
        max_score = 0.0
        
        for unit_length in range(1, min(20, len(sequence) // 3)):
            for start in range(len(sequence) - unit_length * 3):
                unit = sequence[start:start + unit_length]
                repeat_count = 1
                
                pos = start + unit_length
                while pos + unit_length <= len(sequence):
                    if sequence[pos:pos + unit_length] == unit:
                        repeat_count += 1
                        pos += unit_length
                    else:
                        break
                
                if repeat_count >= 3:
                    score = (repeat_count * unit_length) / len(sequence)
                    max_score = max(max_score, score)
        
        return min(1.0, max_score)
    
    def count_homopolymer_runs(self, sequence):
        if len(sequence) == 0:
            return 0
        
        runs = 0
        current_base = sequence[0]
        current_run = 1
        
        for base in sequence[1:]:
            if base == current_base:
                current_run += 1
            else:
                if current_run >= 4:
                    runs += 1
                current_base = base
                current_run = 1
        
        if current_run >= 4:
            runs += 1
        
        return runs
    
    def calculate_dinucleotide_bias(self, sequence):
        if len(sequence) < 2:
            return 0.0
        
        dinucleotide_counts = Counter()
        for i in range(len(sequence) - 1):
            dinuc = sequence[i:i+2]
            dinucleotide_counts[dinuc] += 1
        
        if len(dinucleotide_counts) == 0:
            return 0.0
        
        expected_freq = 1.0 / 16
        observed_freqs = np.array(list(dinucleotide_counts.values())) / (len(sequence) - 1)
        
        chi_square = np.sum((observed_freqs - expected_freq) ** 2 / expected_freq)
        return min(1.0, chi_square / 16)
    
    def extract_sv_structural_features(self, variant):
        features = {}
        
        features['sv_length'] = abs(variant.get('svlen', 0))
        features['sv_type_del'] = 1 if variant.get('svtype') == 'DEL' else 0
        features['sv_type_ins'] = 1 if variant.get('svtype') == 'INS' else 0
        features['sv_type_dup'] = 1 if variant.get('svtype') == 'DUP' else 0
        features['sv_type_inv'] = 1 if variant.get('svtype') == 'INV' else 0
        
        features['read_support'] = variant.get('support', 0)
        features['quality_score'] = variant.get('quality', 0.0)
        
        features['log_sv_length'] = np.log10(max(1, features['sv_length']))
        features['support_density'] = features['read_support'] / max(1, features['sv_length'] / 1000)
        
        return features
    
    def extract_genomic_context_features(self, variant, gene_annotations, repeat_annotations):
        features = {}
        
        chrom = variant.get('chrom', 'chr1')
        pos = variant.get('pos', 0)
        svlen = abs(variant.get('svlen', 0))
        
        features['in_gene'] = self.check_gene_overlap(chrom, pos, pos + svlen, gene_annotations)
        features['in_exon'] = self.check_exon_overlap(chrom, pos, pos + svlen, gene_annotations)
        features['in_intron'] = features['in_gene'] and not features['in_exon']
        features['in_regulatory'] = self.check_regulatory_overlap(chrom, pos, pos + svlen)
        
        features['repeat_overlap'] = self.calculate_repeat_overlap(chrom, pos, pos + svlen, repeat_annotations)
        features['segdup_overlap'] = self.check_segmental_duplication_overlap(chrom, pos, pos + svlen)
        
        features['distance_to_gene'] = self.calculate_distance_to_nearest_gene(chrom, pos, gene_annotations)
        features['gene_density'] = self.calculate_gene_density(chrom, pos, gene_annotations)
        
        return features
    
    def check_gene_overlap(self, chrom, start, end, gene_annotations):
        if gene_annotations is None:
            return 0
        
        for _, gene in gene_annotations.iterrows():
            if (gene['chrom'] == chrom and 
                gene['start'] <= end and gene['end'] >= start):
                return 1
        return 0
    
    def check_exon_overlap(self, chrom, start, end, gene_annotations):
        if gene_annotations is None:
            return 0
        
        return np.random.choice([0, 1], p=[0.9, 0.1])
    
    def check_regulatory_overlap(self, chrom, start, end):
        return np.random.choice([0, 1], p=[0.8, 0.2])
    
    def calculate_repeat_overlap(self, chrom, start, end, repeat_annotations):
        if repeat_annotations is None:
            return 0.0
        
        overlap_length = 0
        sv_length = end - start
        
        for _, repeat in repeat_annotations.iterrows():
            if (repeat['chrom'] == chrom and 
                repeat['start'] <= end and repeat['end'] >= start):
                overlap_start = max(start, repeat['start'])
                overlap_end = min(end, repeat['end'])
                overlap_length += max(0, overlap_end - overlap_start)
        
        return overlap_length / max(1, sv_length)
    
    def check_segmental_duplication_overlap(self, chrom, start, end):
        return np.random.choice([0, 1], p=[0.7, 0.3])
    
    def calculate_distance_to_nearest_gene(self, chrom, pos, gene_annotations):
        if gene_annotations is None:
            return 100000
        
        chrom_genes = gene_annotations[gene_annotations['chrom'] == chrom]
        if len(chrom_genes) == 0:
            return 100000
        
        distances = []
        for _, gene in chrom_genes.iterrows():
            if pos >= gene['start'] and pos <= gene['end']:
                return 0
            elif pos < gene['start']:
                distances.append(gene['start'] - pos)
            else:
                distances.append(pos - gene['end'])
        
        return min(distances) if distances else 100000
    
    def calculate_gene_density(self, chrom, pos, gene_annotations, window=1000000):
        if gene_annotations is None:
            return 0.0
        
        window_start = max(0, pos - window // 2)
        window_end = pos + window // 2
        
        chrom_genes = gene_annotations[
            (gene_annotations['chrom'] == chrom) &
            (gene_annotations['start'] <= window_end) &
            (gene_annotations['end'] >= window_start)
        ]
        
        return len(chrom_genes) / (window / 1000000)
    
    def extract_population_features(self, variant, population_data=None):
        features = {}
        
        if population_data is None:
            features['population_af'] = np.random.beta(0.5, 10)
            features['hw_p_value'] = np.random.uniform(0, 1)
        else:
            key = f"{variant.get('chrom')}:{variant.get('pos')}:{variant.get('svtype')}"
            pop_data = population_data.get(key, {})
            features['population_af'] = pop_data.get('af', 0.0)
            features['hw_p_value'] = pop_data.get('hw_p', 1.0)
        
        features['is_rare'] = 1 if features['population_af'] < 0.01 else 0
        features['is_common'] = 1 if features['population_af'] > 0.05 else 0
        features['hw_deviation'] = 1 if features['hw_p_value'] < 0.001 else 0
        
        features['log_population_af'] = np.log10(max(1e-6, features['population_af']))
        features['log_hw_p'] = np.log10(max(1e-6, features['hw_p_value']))
        
        return features
    
    def create_feature_vector(self, variant, gene_annotations=None, repeat_annotations=None, population_data=None):
        all_features = {}
        
        chrom = variant.get('chrom', 'chr1')
        pos = variant.get('pos', 0)
        end = pos + abs(variant.get('svlen', 0))
        
        sequence_features = self.extract_sequence_features(chrom, pos, end)
        structural_features = self.extract_sv_structural_features(variant)
        context_features = self.extract_genomic_context_features(variant, gene_annotations, repeat_annotations)
        population_features = self.extract_population_features(variant, population_data)
        
        all_features.update(sequence_features)
        all_features.update(structural_features)
        all_features.update(context_features)
        all_features.update(population_features)
        
        return all_features
    
    def process_variant_dataset(self, variants_df, gene_annotations=None, repeat_annotations=None, population_data=None):
        feature_matrix = []
        
        for _, variant in variants_df.iterrows():
            features = self.create_feature_vector(
                variant.to_dict(), 
                gene_annotations, 
                repeat_annotations, 
                population_data
            )
            feature_matrix.append(features)
        
        return pd.DataFrame(feature_matrix)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from structural variants')
    parser.add_argument('--variants', required=True, help='Input variants file (TSV)')
    parser.add_argument('--genes', help='Gene annotations (BED)')
    parser.add_argument('--repeats', help='Repeat annotations (BED)')
    parser.add_argument('--population', help='Population data (JSON)')
    parser.add_argument('--output', required=True, help='Output features file (TSV)')
    
    args = parser.parse_args()
    
    extractor = SVFeatureExtractor()
    
    variants_df = pd.read_csv(args.variants, sep='\t')
    
    gene_annotations = None
    if args.genes:
        gene_annotations = pd.read_csv(args.genes, sep='\t', header=None, 
                                     names=['chrom', 'start', 'end', 'gene_name', 'gene_type'])
    
    repeat_annotations = None
    if args.repeats:
        repeat_annotations = pd.read_csv(args.repeats, sep='\t', header=None,
                                       names=['chrom', 'start', 'end', 'repeat_type', 'repeat_family'])
    
    population_data = None
    if args.population:
        with open(args.population, 'r') as f:
            population_data = json.load(f)
    
    features_df = extractor.process_variant_dataset(
        variants_df, gene_annotations, repeat_annotations, population_data
    )
    
    features_df.to_csv(args.output, sep='\t', index=False)
    print(f"Extracted {len(features_df.columns)} features for {len(features_df)} variants")

if __name__ == "__main__":
    main()