# SV Genomics Annotation Framework

Structural variant detection and annotation pipeline for long-read sequencing data.

## Features

- CIGAR-based SV detection with clustering
- Population genetics filtering (Hardy-Weinberg, LD analysis)
- Gene/repeat annotation with pathogenicity scoring
- Automated pipeline with quality control

## Installation

```bash
git clone https://github.com/atticuscolwell/sv-genomics-annotation-framework.git
cd sv-genomics-annotation-framework
pip install -r requirements.txt
```

## Usage

1. Edit `config.json` with file paths
2. Run: `python pipeline_runner.py --config config.json`

## Input/Output

**Input:** JSON alignments, BED annotations, JSON config  
**Output:** VCF variants, TSV annotations, HTML report

## Components

- `sv_hunter.py` - SV detection from CIGAR strings
- `variant_annotator.py` - Gene/repeat annotation and scoring  
- `population_filter.py` - Population genetics filtering
- `pipeline_runner.py` - Workflow orchestration

## Key Parameters

- `min_support`: Read support threshold (default: 3)
- `min_length`: SV length cutoff (default: 50bp)  
- `frequency_threshold`: Population frequency filter (default: 0.01)