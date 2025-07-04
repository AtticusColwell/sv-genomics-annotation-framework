# SV Genomics Annotation Framework

A precision structural variant detection and annotation pipeline for long-read genomic sequencing data.

## Overview

ChromaHunter implements state-of-the-art algorithms for detecting structural variants (SVs) from long-read sequencing alignments, with sophisticated population genetics filtering and comprehensive genomic annotation. The pipeline is designed for high-throughput analysis of large-scale genomic datasets.

## Features

- **Advanced SV Detection**: CIGAR-based breakpoint detection with clustering algorithms
- **Population Genetics Filtering**: Hardy-Weinberg equilibrium testing and linkage disequilibrium analysis
- **Comprehensive Annotation**: Gene overlap analysis, repeat element characterization, and pathogenicity scoring
- **Quality Control**: Multi-level filtering with configurable thresholds
- **Scalable Pipeline**: Automated workflow with intermediate file management

## Installation

```bash
git clone https://github.com/yourusername/ChromaHunter.git
cd ChromaHunter
pip install -r requirements.txt
```

## Quick Start

1. Prepare your input files:
   - Alignment data in JSON format
   - Gene annotations (BED format)
   - Repeat annotations (BED format)

2. Configure the pipeline:
   - Edit `config.json` with your file paths and parameters

3. Run the complete pipeline:
   ```bash
   python pipeline_runner.py --config config.json
   ```

## Pipeline Components

### SV Detection (`sv_hunter.py`)
Detects insertions and deletions from CIGAR strings with configurable support thresholds.

### Variant Annotation (`variant_annotator.py`)
Annotates detected variants with:
- Gene overlaps and impact assessment
- Repeat element characterization
- Regulatory impact scoring
- Pathogenicity prediction

### Population Filtering (`population_filter.py`)
Applies population genetics principles:
- Allele frequency calculations
- Hardy-Weinberg equilibrium testing
- Linkage disequilibrium analysis
- Quality-based filtering

### Pipeline Management (`pipeline_runner.py`)
Orchestrates the complete analysis workflow with logging and error handling.

## Input Formats

- **Alignments**: JSON with read_id, pos, cigar fields
- **Annotations**: BED format with standard genomic intervals
- **Configuration**: JSON with all pipeline parameters

## Output

- VCF file with detected variants
- TSV file with comprehensive annotations
- HTML report with summary statistics
- JSON file with filtering statistics

## Configuration

Key parameters in `config.json`:
- `min_support`: Minimum read support for SV calling
- `min_length`: Minimum SV length threshold
- `frequency_threshold`: Population frequency cutoff
- `quality_threshold`: Variant quality score minimum

## Performance

Optimized for:
- Large-scale datasets (>100GB)
- High-coverage long-read data
- Population-scale analysis
- Clinical genomics workflows

## Citation

If you use ChromaHunter in your research, please cite:
*ChromaHunter: A comprehensive pipeline for structural variant detection and population genetics analysis in long-read sequencing data*