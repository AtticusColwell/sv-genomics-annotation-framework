#!/usr/bin/env python3

import subprocess
import argparse
import json
import os
from pathlib import Path
import time
import logging

class SVPipeline:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.setup_logging()
        self.validate_inputs()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sv_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_inputs(self):
        required_files = [
            'input_alignments',
            'gene_annotations',
            'repeat_annotations'
        ]
        
        for file_key in required_files:
            if file_key not in self.config:
                raise ValueError(f"Missing required input: {file_key}")
            
            file_path = Path(self.config[file_key])
            if not file_path.exists():
                raise FileNotFoundError(f"Input file not found: {file_path}")
    
    def run_command(self, command, description):
        self.logger.info(f"Starting: {description}")
        self.logger.info(f"Command: {' '.join(command)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            
            elapsed = time.time() - start_time
            self.logger.info(f"Completed: {description} ({elapsed:.2f}s)")
            
            if result.stdout:
                self.logger.info(f"Output: {result.stdout.strip()}")
            
            return result
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed: {description}")
            self.logger.error(f"Return code: {e.returncode}")
            self.logger.error(f"Stderr: {e.stderr}")
            raise
    
    def step_1_detect_variants(self):
        output_vcf = self.config['output_dir'] + '/detected_variants.vcf'
        
        command = [
            'python3', 'sv_hunter.py',
            '--input', self.config['input_alignments'],
            '--output', output_vcf,
            '--min-support', str(self.config.get('min_support', 3)),
            '--min-length', str(self.config.get('min_length', 50))
        ]
        
        self.run_command(command, "Structural variant detection")
        return output_vcf
    
    def step_2_annotate_variants(self, vcf_file):
        output_annotations = self.config['output_dir'] + '/annotated_variants.tsv'
        
        command = [
            'python3', 'variant_annotator.py',
            '--vcf', vcf_file,
            '--genes', self.config['gene_annotations'],
            '--repeats', self.config['repeat_annotations'],
            '--output', output_annotations
        ]
        
        self.run_command(command, "Variant annotation")
        return output_annotations
    
    def step_3_population_filtering(self, annotations_file):
        output_filtered = self.config['output_dir'] + '/filtered_variants.tsv'
        stats_file = self.config['output_dir'] + '/filtering_stats.json'
        
        command = [
            'python3', 'population_filter.py',
            '--input', annotations_file,
            '--output', output_filtered,
            '--freq-threshold', str(self.config.get('frequency_threshold', 0.01)),
            '--stats', stats_file
        ]
        
        if 'population_frequencies' in self.config:
            command.extend(['--population-freq', self.config['population_frequencies']])
        
        self.run_command(command, "Population-based filtering")
        return output_filtered, stats_file
    
    def generate_report(self, filtered_variants, stats_file):
        report_file = self.config['output_dir'] + '/pipeline_report.html'
        
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SV Pipeline Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .stats {{ margin: 20px 0; }}
                .stat-item {{ margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Structural Variant Analysis Report</h1>
                <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="stats">
                <h2>Summary Statistics</h2>
                <div class="stat-item"><strong>Original variants:</strong> {stats['original_count']}</div>
                <div class="stat-item"><strong>Filtered variants:</strong> {stats['filtered_count']}</div>
                <div class="stat-item"><strong>Reduction:</strong> {stats['reduction_percentage']:.1f}%</div>
                <div class="stat-item"><strong>Mean pathogenicity score:</strong> {stats['mean_pathogenicity']:.3f}</div>
                <div class="stat-item"><strong>Mean read support:</strong> {stats['mean_support']:.1f}</div>
            </div>
            
            <div class="stats">
                <h2>Variant Types</h2>
                <table>
                    <tr><th>Type</th><th>Count</th></tr>
        """
        
        for sv_type, count in stats['sv_types'].items():
            html_content += f"<tr><td>{sv_type}</td><td>{count}</td></tr>"
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Report generated: {report_file}")
        return report_file
    
    def run_pipeline(self):
        self.logger.info("Starting SV analysis pipeline")
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        try:
            vcf_file = self.step_1_detect_variants()
            annotations_file = self.step_2_annotate_variants(vcf_file)
            filtered_variants, stats_file = self.step_3_population_filtering(annotations_file)
            report_file = self.generate_report(filtered_variants, stats_file)
            
            self.logger.info("Pipeline completed successfully")
            self.logger.info(f"Final results: {filtered_variants}")
            self.logger.info(f"Report: {report_file}")
            
            return {
                'vcf': vcf_file,
                'annotations': annotations_file,
                'filtered_variants': filtered_variants,
                'stats': stats_file,
                'report': report_file
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def cleanup_intermediate_files(self):
        intermediate_files = [
            self.config['output_dir'] + '/detected_variants.vcf',
            self.config['output_dir'] + '/annotated_variants.tsv'
        ]
        
        for file_path in intermediate_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info(f"Cleaned up: {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Run complete SV analysis pipeline')
    parser.add_argument('--config', required=True, help='Pipeline configuration file (JSON)')
    parser.add_argument('--cleanup', action='store_true', help='Remove intermediate files')
    
    args = parser.parse_args()
    
    pipeline = SVPipeline(args.config)
    results = pipeline.run_pipeline()
    
    if args.cleanup:
        pipeline.cleanup_intermediate_files()
    
    print("Pipeline completed successfully!")
    print(f"Results directory: {pipeline.config['output_dir']}")

if __name__ == "__main__":
    main()