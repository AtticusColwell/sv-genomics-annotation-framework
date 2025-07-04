#!/usr/bin/env python3

import numpy as np
import pandas as pd
from collections import defaultdict
import argparse
from pathlib import Path
import json

class SVHunter:
    def __init__(self, min_support=3, min_length=50, max_distance=500):
        self.min_support = min_support
        self.min_length = min_length
        self.max_distance = max_distance
        self.breakpoints = defaultdict(list)
        self.variants = []
        
    def parse_cigar(self, cigar_string):
        operations = []
        current_num = ""
        
        for char in cigar_string:
            if char.isdigit():
                current_num += char
            else:
                if current_num:
                    operations.append((int(current_num), char))
                    current_num = ""
        return operations
    
    def detect_insertions(self, cigar_ops, pos, read_id):
        ref_pos = pos
        for length, op in cigar_ops:
            if op == 'I' and length >= self.min_length:
                self.breakpoints[f"{ref_pos}_INS"].append({
                    'read_id': read_id,
                    'pos': ref_pos,
                    'length': length,
                    'type': 'INS'
                })
            elif op in ['M', 'D', 'N']:
                ref_pos += length
                
    def detect_deletions(self, cigar_ops, pos, read_id):
        ref_pos = pos
        for length, op in cigar_ops:
            if op == 'D' and length >= self.min_length:
                self.breakpoints[f"{ref_pos}_DEL"].append({
                    'read_id': read_id,
                    'pos': ref_pos,
                    'length': length,
                    'type': 'DEL'
                })
            elif op in ['M', 'D', 'N']:
                ref_pos += length
    
    def process_alignments(self, alignments):
        for alignment in alignments:
            read_id = alignment['read_id']
            pos = alignment['pos']
            cigar = alignment['cigar']
            
            cigar_ops = self.parse_cigar(cigar)
            self.detect_insertions(cigar_ops, pos, read_id)
            self.detect_deletions(cigar_ops, pos, read_id)
    
    def cluster_breakpoints(self):
        for bp_key, supports in self.breakpoints.items():
            if len(supports) >= self.min_support:
                chrom, sv_type = bp_key.split('_')
                positions = [s['pos'] for s in supports]
                lengths = [s['length'] for s in supports]
                
                self.variants.append({
                    'chrom': chrom,
                    'pos': int(np.median(positions)),
                    'type': sv_type,
                    'length': int(np.median(lengths)),
                    'support': len(supports),
                    'supporting_reads': [s['read_id'] for s in supports]
                })
    
    def filter_variants(self, quality_threshold=0.8):
        filtered = []
        for variant in self.variants:
            if variant['support'] >= self.min_support:
                quality = min(1.0, variant['support'] / 10.0)
                if quality >= quality_threshold:
                    variant['quality'] = quality
                    filtered.append(variant)
        return filtered
    
    def export_vcf(self, variants, output_path):
        with open(output_path, 'w') as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write("##INFO=<ID=SVTYPE,Number=1,Type=String,Description=\"Type of structural variant\">\n")
            f.write("##INFO=<ID=SVLEN,Number=1,Type=Integer,Description=\"Length of structural variant\">\n")
            f.write("##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description=\"Number of supporting reads\">\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            
            for i, variant in enumerate(variants):
                svtype = variant['type']
                svlen = variant['length']
                support = variant['support']
                quality = int(variant['quality'] * 100)
                
                if svtype == 'DEL':
                    alt = '<DEL>'
                elif svtype == 'INS':
                    alt = '<INS>'
                else:
                    alt = f'<{svtype}>'
                
                info = f"SVTYPE={svtype};SVLEN={svlen};SUPPORT={support}"
                f.write(f"{variant['chrom']}\t{variant['pos']}\tsv_{i}\tN\t{alt}\t{quality}\tPASS\t{info}\n")

def main():
    parser = argparse.ArgumentParser(description='Hunt for structural variants in long-read data')
    parser.add_argument('--input', required=True, help='Input alignment file (JSON format)')
    parser.add_argument('--output', required=True, help='Output VCF file')
    parser.add_argument('--min-support', type=int, default=3, help='Minimum read support')
    parser.add_argument('--min-length', type=int, default=50, help='Minimum SV length')
    
    args = parser.parse_args()
    
    hunter = SVHunter(
        min_support=args.min_support,
        min_length=args.min_length
    )
    
    with open(args.input) as f:
        alignments = json.load(f)
    
    hunter.process_alignments(alignments)
    hunter.cluster_breakpoints()
    variants = hunter.filter_variants()
    
    hunter.export_vcf(variants, args.output)
    print(f"Detected {len(variants)} high-quality structural variants")

if __name__ == "__main__":
    main()