#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json

class GenomicAnnotator:
    def __init__(self, gene_bed_file, repeat_bed_file):
        self.genes = self.load_bed_file(gene_bed_file, ['gene_name', 'gene_type'])
        self.repeats = self.load_bed_file(repeat_bed_file, ['repeat_type', 'repeat_family'])
        
    def load_bed_file(self, bed_file, extra_columns):
        columns = ['chrom', 'start', 'end'] + extra_columns
        return pd.read_csv(bed_file, sep='\t', header=None, names=columns)
    
    def parse_vcf_line(self, line):
        if line.startswith('#'):
            return None
        
        fields = line.strip().split('\t')
        chrom = fields[0]
        pos = int(fields[1])
        info = fields[7]
        
        info_dict = {}
        for item in info.split(';'):
            if '=' in item:
                key, value = item.split('=', 1)
                info_dict[key] = value
        
        return {
            'chrom': chrom,
            'pos': pos,
            'svtype': info_dict.get('SVTYPE', 'UNK'),
            'svlen': int(info_dict.get('SVLEN', 0)),
            'support': int(info_dict.get('SUPPORT', 0)),
            'info': info_dict
        }
    
    def find_overlaps(self, variant, annotation_df):
        chrom_matches = annotation_df[annotation_df['chrom'] == variant['chrom']]
        
        var_start = variant['pos']
        var_end = variant['pos'] + abs(variant['svlen'])
        
        overlaps = chrom_matches[
            (chrom_matches['start'] <= var_end) & 
            (chrom_matches['end'] >= var_start)
        ]
        
        return overlaps
    
    def calculate_overlap_fraction(self, var_start, var_end, ann_start, ann_end):
        overlap_start = max(var_start, ann_start)
        overlap_end = min(var_end, ann_end)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap_length = overlap_end - overlap_start
        var_length = var_end - var_start
        
        return overlap_length / var_length if var_length > 0 else 0.0
    
    def annotate_variant(self, variant):
        annotations = {
            'genes': [],
            'repeats': [],
            'regulatory_impact': 'LOW',
            'pathogenicity_score': 0.0
        }
        
        gene_overlaps = self.find_overlaps(variant, self.genes)
        repeat_overlaps = self.find_overlaps(variant, self.repeats)
        
        var_start = variant['pos']
        var_end = variant['pos'] + abs(variant['svlen'])
        
        max_gene_overlap = 0.0
        for _, gene in gene_overlaps.iterrows():
            overlap_frac = self.calculate_overlap_fraction(
                var_start, var_end, gene['start'], gene['end']
            )
            
            annotations['genes'].append({
                'name': gene['gene_name'],
                'type': gene['gene_type'],
                'overlap_fraction': overlap_frac
            })
            
            max_gene_overlap = max(max_gene_overlap, overlap_frac)
        
        for _, repeat in repeat_overlaps.iterrows():
            overlap_frac = self.calculate_overlap_fraction(
                var_start, var_end, repeat['start'], repeat['end']
            )
            
            annotations['repeats'].append({
                'type': repeat['repeat_type'],
                'family': repeat['repeat_family'],
                'overlap_fraction': overlap_frac
            })
        
        if max_gene_overlap > 0.5:
            annotations['regulatory_impact'] = 'HIGH'
        elif max_gene_overlap > 0.1:
            annotations['regulatory_impact'] = 'MODERATE'
        
        pathogenicity = self.calculate_pathogenicity(variant, annotations)
        annotations['pathogenicity_score'] = pathogenicity
        
        return annotations
    
    def calculate_pathogenicity(self, variant, annotations):
        score = 0.0
        
        if variant['svtype'] == 'DEL':
            score += 0.3
        elif variant['svtype'] == 'INS':
            score += 0.2
        
        if variant['svlen'] > 1000:
            score += 0.2
        elif variant['svlen'] > 100:
            score += 0.1
        
        for gene in annotations['genes']:
            if gene['type'] == 'protein_coding':
                score += gene['overlap_fraction'] * 0.4
            elif gene['type'] == 'lncRNA':
                score += gene['overlap_fraction'] * 0.2
        
        for repeat in annotations['repeats']:
            if repeat['type'] == 'LINE':
                score += repeat['overlap_fraction'] * 0.1
        
        if variant['support'] < 5:
            score *= 0.7
        
        return min(1.0, score)
    
    def process_vcf(self, vcf_file):
        annotated_variants = []
        
        with open(vcf_file, 'r') as f:
            for line in f:
                variant = self.parse_vcf_line(line)
                if variant is None:
                    continue
                
                annotations = self.annotate_variant(variant)
                variant.update(annotations)
                annotated_variants.append(variant)
        
        return annotated_variants
    
    def export_annotations(self, annotated_variants, output_file):
        results = []
        
        for variant in annotated_variants:
            gene_names = [g['name'] for g in variant['genes']]
            repeat_types = [r['type'] for r in variant['repeats']]
            
            results.append({
                'chrom': variant['chrom'],
                'pos': variant['pos'],
                'svtype': variant['svtype'],
                'svlen': variant['svlen'],
                'support': variant['support'],
                'genes': ','.join(gene_names) if gene_names else 'NONE',
                'repeats': ','.join(repeat_types) if repeat_types else 'NONE',
                'regulatory_impact': variant['regulatory_impact'],
                'pathogenicity_score': round(variant['pathogenicity_score'], 3)
            })
        
        df = pd.DataFrame(results)
        df.to_csv(output_file, sep='\t', index=False)

def main():
    parser = argparse.ArgumentParser(description='Annotate structural variants with genomic features')
    parser.add_argument('--vcf', required=True, help='Input VCF file')
    parser.add_argument('--genes', required=True, help='Gene annotation BED file')
    parser.add_argument('--repeats', required=True, help='Repeat annotation BED file')
    parser.add_argument('--output', required=True, help='Output annotation file')
    
    args = parser.parse_args()
    
    annotator = GenomicAnnotator(args.genes, args.repeats)
    annotated_variants = annotator.process_vcf(args.vcf)
    annotator.export_annotations(annotated_variants, args.output)
    
    print(f"Annotated {len(annotated_variants)} variants")

if __name__ == "__main__":
    main()