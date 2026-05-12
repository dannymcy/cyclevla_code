#!/usr/bin/env python3
"""
MBR Analysis Script for LIBERO Task Suite

This script aggregates trajectory analysis results across multiple checkpoint folders
and LIBERO tasks. It computes statistics matrices for Random, MBR, and r-NN methods
with number of hypotheses (N) and distance metrics as axes.

Usage:
    python run_mbr_analysis.py [--rollouts_dir PATH] [--output_dir PATH] [--metric METRIC]
    python run_mbr_analysis.py --first_n 3  # Only use first 3 timesteps (e.g., 0, 8, 16)

Example:
    python run_mbr_analysis.py --rollouts_dir /hdd2/kai/openvla-oft/rollouts
"""

# conda activate /hdd2/kai/openvla-oft/env
# python experiments/robot/libero/run_mbr_analysis.py
# python experiments/robot/libero/run_mbr_analysis.py --first_n 1

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# Constants
N_VALUES = [4, 8, 16, 32, 64]
DISTANCE_METRICS = ['CHEBYSHEV', 'CORRELATION', 'COSINE', 'L1', 'L2']
METHODS = ['MBR', 'RNN']  # Random is handled separately
TARGET_FEATURES = ['combined', 'positions', 'rotations']
TARGET_TYPE = 'REP'  # Only REP, not AWAY


def find_checkpoint_folders(rollouts_dir: str) -> List[str]:
    """Find all checkpoint folders matching the pattern."""
    pattern = os.path.join(rollouts_dir, 'rollouts_sub_decomposed_progress_transit_seed_*_chkpt')
    folders = sorted(glob.glob(pattern))
    return folders


def extract_seed_from_folder(folder: str) -> int:
    """Extract the seed number from folder name."""
    match = re.search(r'seed_(\d+)_chkpt', folder)
    if match:
        return int(match.group(1))
    return -1


def find_xlsx_files(checkpoint_folder: str) -> List[str]:
    """Find all trajectory_analysis xlsx files in libero task subdirectories."""
    xlsx_files = []
    # Look in libero_* subdirectories
    libero_dirs = glob.glob(os.path.join(checkpoint_folder, 'libero_*'))
    for libero_dir in libero_dirs:
        pattern = os.path.join(libero_dir, 'trajectory_analysis_*.xlsx')
        # pattern = os.path.join(libero_dir, '*.xlsx')
        xlsx_files.extend(glob.glob(pattern))
    return xlsx_files


def parse_metric_name(metric: str) -> Tuple[str, str, int]:
    """
    Parse metric name into (method, distance, N).
    
    Examples:
        'MBR_CHEBYSHEV(REP)_N16' -> ('MBR', 'CHEBYSHEV', 16)
        'RNN_L2(REP)_N64' -> ('RNN', 'L2', 64)
        'RANDOM_N16' -> ('RANDOM', None, 16)
    """
    # Match RANDOM_N{x}
    random_match = re.match(r'RANDOM_N(\d+)', metric)
    if random_match:
        return ('RANDOM', None, int(random_match.group(1)))
    
    # Match METHOD_DISTANCE(TYPE)_N{x}
    method_match = re.match(r'(MBR|RNN)_([A-Z0-9]+)\(([A-Z]+)\)_N(\d+)', metric)
    if method_match:
        method, distance, metric_type, n = method_match.groups()
        if metric_type == TARGET_TYPE:
            return (method, distance, int(n))
    
    return (None, None, None)


def read_xlsx_feature_averages(xlsx_path: str) -> pd.DataFrame:
    """Read the Feature_Metric_Averages sheet from an xlsx file."""
    try:
        df = pd.read_excel(xlsx_path, sheet_name='Feature_Metric_Averages')
        # Forward fill the Feature column
        df['Feature'] = df['Feature'].ffill()
        return df
    except Exception as e:
        print(f"Warning: Could not read {xlsx_path}: {e}")
        return None


def read_xlsx_raw_data_first_n(xlsx_path: str, first_n: int) -> pd.DataFrame:
    """
    Read the Raw_Data sheet and filter to first N timesteps.
    Returns DataFrame with same structure as Feature_Metric_Averages.
    
    Args:
        xlsx_path: Path to xlsx file
        first_n: Number of timesteps to use (e.g., 3 means timesteps 0, 8, 16)
    """
    try:
        df = pd.read_excel(xlsx_path, sheet_name='Raw_Data')
        
        # Get unique timesteps sorted
        unique_timesteps = sorted(df['Timestep'].unique())
        
        if first_n > len(unique_timesteps):
            print(f"  Warning: first_n={first_n} > available timesteps ({len(unique_timesteps)})")
            valid_timesteps = unique_timesteps
        else:
            valid_timesteps = unique_timesteps[:first_n]
        
        # Filter to first N timesteps
        df_filtered = df[df['Timestep'].isin(valid_timesteps)].copy()
        
        # Convert percentage strings to floats
        for col in ['Top-1_Prob', 'Top-3_Prob']:
            if df_filtered[col].dtype == object:
                df_filtered[col] = df_filtered[col].str.rstrip('%').astype(float)
        
        # Group by Feature and Metric, compute mean
        aggregated = df_filtered.groupby(['Feature', 'Metric']).agg({
            'Top-1_Prob': 'mean',
            'Top-3_Prob': 'mean'
        }).reset_index()
        
        return aggregated
            
    except Exception as e:
        print(f"Warning: Could not read Raw_Data from {xlsx_path}: {e}")
        return None


def extract_rep_data(df: pd.DataFrame) -> Dict:
    """Extract data for all target features with REP metrics."""
    data = {}
    
    for feature in TARGET_FEATURES:
        feature_df = df[df['Feature'] == feature].copy()
        
        feature_data = {
            'random': {},  # {N: {'Top-1_Prob': [], 'Top-3_Prob': []}}
            'mbr': {},     # {(distance, N): {'Top-1_Prob': [], 'Top-3_Prob': []}}
            'rnn': {}      # {(distance, N): {'Top-1_Prob': [], 'Top-3_Prob': []}}
        }
        
        for _, row in feature_df.iterrows():
            metric = row['Metric']
            method, distance, n = parse_metric_name(metric)
            
            if method is None:
                continue
            
            # Handle both percentage strings and float values
            top1 = row['Top-1_Prob']
            top3 = row['Top-3_Prob']
            
            if isinstance(top1, str):
                top1 = float(top1.rstrip('%'))
            if isinstance(top3, str):
                top3 = float(top3.rstrip('%'))
            
            if method == 'RANDOM':
                if n not in feature_data['random']:
                    feature_data['random'][n] = {'Top-1_Prob': [], 'Top-3_Prob': []}
                feature_data['random'][n]['Top-1_Prob'].append(top1)
                feature_data['random'][n]['Top-3_Prob'].append(top3)
            elif method == 'MBR':
                key = (distance, n)
                if key not in feature_data['mbr']:
                    feature_data['mbr'][key] = {'Top-1_Prob': [], 'Top-3_Prob': []}
                feature_data['mbr'][key]['Top-1_Prob'].append(top1)
                feature_data['mbr'][key]['Top-3_Prob'].append(top3)
            elif method == 'RNN':
                key = (distance, n)
                if key not in feature_data['rnn']:
                    feature_data['rnn'][key] = {'Top-1_Prob': [], 'Top-3_Prob': []}
                feature_data['rnn'][key]['Top-1_Prob'].append(top1)
                feature_data['rnn'][key]['Top-3_Prob'].append(top3)
        
        data[feature] = feature_data
    
    return data


def aggregate_data(all_data: List[Dict]) -> Dict:
    """Aggregate data from multiple xlsx files for all features."""
    aggregated = {}
    
    for feature in TARGET_FEATURES:
        aggregated[feature] = {
            'random': {},
            'mbr': {},
            'rnn': {}
        }
    
    for data in all_data:
        for feature in TARGET_FEATURES:
            if feature not in data:
                continue
            for method in ['random', 'mbr', 'rnn']:
                for key, values in data[feature][method].items():
                    if key not in aggregated[feature][method]:
                        aggregated[feature][method][key] = {'Top-1_Prob': [], 'Top-3_Prob': []}
                    aggregated[feature][method][key]['Top-1_Prob'].extend(values['Top-1_Prob'])
                    aggregated[feature][method][key]['Top-3_Prob'].extend(values['Top-3_Prob'])
    
    return aggregated


def compute_statistics(values: List[float]) -> Dict:
    """Compute mean and std from a list of values."""
    if not values:
        return {'mean': np.nan, 'std': np.nan, 'count': 0}
    arr = np.array(values)
    return {
        'mean': np.nanmean(arr),
        'std': np.nanstd(arr),
        'count': len(arr)
    }


def create_random_matrix(aggregated_data: Dict, prob_type: str = 'Top-1_Prob') -> pd.DataFrame:
    """Create a simple row for random results."""
    random_data = aggregated_data['random']
    
    results = {}
    for n in N_VALUES:
        if n in random_data:
            stats = compute_statistics(random_data[n][prob_type])
            results[f'N{n}'] = f"{stats['mean']:.1f}±{stats['std']:.1f}"
        else:
            results[f'N{n}'] = 'N/A'
    
    df = pd.DataFrame([results], index=['RANDOM'])
    return df


def create_method_matrix(aggregated_data: Dict, method: str, prob_type: str = 'Top-1_Prob') -> pd.DataFrame:
    """Create a matrix for MBR or RNN with distance as rows and N as columns."""
    method_key = method.lower()
    method_data = aggregated_data[method_key]
    
    matrix = []
    for distance in DISTANCE_METRICS:
        row = {'Distance': distance}
        for n in N_VALUES:
            key = (distance, n)
            if key in method_data:
                stats = compute_statistics(method_data[key][prob_type])
                row[f'N{n}'] = f"{stats['mean']:.1f}±{stats['std']:.1f}"
            else:
                row[f'N{n}'] = 'N/A'
        matrix.append(row)
    
    df = pd.DataFrame(matrix)
    df = df.set_index('Distance')
    return df


def create_method_matrix_raw(aggregated_data: Dict, method: str, prob_type: str = 'Top-1_Prob') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create raw mean and std matrices for MBR or RNN."""
    method_key = method.lower()
    method_data = aggregated_data[method_key]
    
    mean_matrix = []
    std_matrix = []
    for distance in DISTANCE_METRICS:
        mean_row = {'Distance': distance}
        std_row = {'Distance': distance}
        for n in N_VALUES:
            key = (distance, n)
            if key in method_data:
                stats = compute_statistics(method_data[key][prob_type])
                mean_row[f'N{n}'] = stats['mean']
                std_row[f'N{n}'] = stats['std']
            else:
                mean_row[f'N{n}'] = np.nan
                std_row[f'N{n}'] = np.nan
        mean_matrix.append(mean_row)
        std_matrix.append(std_row)
    
    mean_df = pd.DataFrame(mean_matrix).set_index('Distance')
    std_df = pd.DataFrame(std_matrix).set_index('Distance')
    return mean_df, std_df


def process_checkpoint(checkpoint_folder: str, first_n: Optional[int] = None) -> Dict:
    """Process all xlsx files in a checkpoint folder."""
    xlsx_files = find_xlsx_files(checkpoint_folder)
    
    if not xlsx_files:
        print(f"  No xlsx files found in {checkpoint_folder}")
        return None
    
    print(f"  Found {len(xlsx_files)} xlsx files")
    
    all_data = []
    for xlsx_file in xlsx_files:
        if first_n is not None:
            # Read from Raw_Data and filter to first N timesteps
            df = read_xlsx_raw_data_first_n(xlsx_file, first_n)
        else:
            # Original behavior: read from Feature_Metric_Averages
            df = read_xlsx_feature_averages(xlsx_file)
        
        if df is not None:
            data = extract_rep_data(df)
            all_data.append(data)
    
    if not all_data:
        return None
    
    aggregated = aggregate_data(all_data)
    return aggregated


def print_results(seed: int, aggregated: Dict, prob_type: str = 'Top-1_Prob', first_n: Optional[int] = None):
    """Print formatted results for a checkpoint."""
    print(f"\n{'='*80}")
    print(f"Checkpoint Seed: {seed}")
    print(f"Metric: {prob_type}")
    if first_n is not None:
        print(f"Timesteps: First {first_n} only")
    print(f"Features: {', '.join(TARGET_FEATURES)} | Type: {TARGET_TYPE}")
    print(f"{'='*80}")
    
    for feature in TARGET_FEATURES:
        print(f"\n>>> Feature: {feature.upper()} <<<")
        feature_data = aggregated[feature]
        
        # Random
        print("\n--- RANDOM ---")
        random_df = create_random_matrix(feature_data, prob_type)
        print(random_df.to_string())
        
        # MBR
        print("\n--- MBR (REP) ---")
        mbr_df = create_method_matrix(feature_data, 'MBR', prob_type)
        print(mbr_df.to_string())
        
        # RNN
        print("\n--- r-NN (REP) ---")
        rnn_df = create_method_matrix(feature_data, 'RNN', prob_type)
        print(rnn_df.to_string())


def create_random_matrix_df(aggregated_data: Dict, prob_type: str = 'Top-1_Prob') -> pd.DataFrame:
    """Create a DataFrame for random results (single row, N as columns)."""
    random_data = aggregated_data['random']
    row = {}
    for n in N_VALUES:
        if n in random_data:
            stats = compute_statistics(random_data[n][prob_type])
            row[f'N{n}'] = stats['mean']
        else:
            row[f'N{n}'] = np.nan
    df = pd.DataFrame([row], index=['RANDOM'])
    return df


def create_method_matrix_df(aggregated_data: Dict, method: str, prob_type: str = 'Top-1_Prob') -> pd.DataFrame:
    """Create a matrix DataFrame: rows=Distance, columns=N."""
    method_key = method.lower()
    method_data = aggregated_data[method_key]
    
    matrix = []
    for distance in DISTANCE_METRICS:
        row = {'Distance': distance}
        for n in N_VALUES:
            key = (distance, n)
            if key in method_data:
                stats = compute_statistics(method_data[key][prob_type])
                row[f'N{n}'] = stats['mean']
            else:
                row[f'N{n}'] = np.nan
        matrix.append(row)
    
    df = pd.DataFrame(matrix).set_index('Distance')
    return df


def save_all_results_to_xlsx(all_results: Dict[int, Dict], output_dir: str, first_n: Optional[int] = None):
    """Save all checkpoint results to a single xlsx file with separate sheets."""
    os.makedirs(output_dir, exist_ok=True)
    
    if first_n is not None:
        output_path = os.path.join(output_dir, f'mbr_analysis_results_first{first_n}.xlsx')
    else:
        output_path = os.path.join(output_dir, 'mbr_analysis_results.xlsx')
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for seed in sorted(all_results.keys()):
            aggregated = all_results[seed]
            
            for prob_type in ['Top-1_Prob', 'Top-3_Prob']:
                prob_suffix = 'Top1' if prob_type == 'Top-1_Prob' else 'Top3'
                
                # RANDOM sheet
                random_dfs = []
                for i, feature in enumerate(TARGET_FEATURES):
                    df = create_random_matrix_df(aggregated[feature], prob_type)
                    df.columns = [f'{feature}_{col}' for col in df.columns]
                    random_dfs.append(df)
                    if i < len(TARGET_FEATURES) - 1:
                        spacer = pd.DataFrame({'': [np.nan]}, index=df.index)
                        random_dfs.append(spacer)
                combined_random = pd.concat(random_dfs, axis=1)
                sheet_name = f'{seed}_RANDOM_{prob_suffix}'
                combined_random.to_excel(writer, sheet_name=sheet_name)
                
                # MBR sheet
                mbr_dfs = []
                for i, feature in enumerate(TARGET_FEATURES):
                    df = create_method_matrix_df(aggregated[feature], 'MBR', prob_type)
                    df.columns = [f'{feature}_{col}' for col in df.columns]
                    mbr_dfs.append(df)
                    if i < len(TARGET_FEATURES) - 1:
                        spacer = pd.DataFrame({' ': [np.nan]*len(df)}, index=df.index)
                        mbr_dfs.append(spacer)
                combined_mbr = pd.concat(mbr_dfs, axis=1)
                sheet_name = f'{seed}_MBR_{prob_suffix}'
                combined_mbr.to_excel(writer, sheet_name=sheet_name)
                
                # RNN sheet
                rnn_dfs = []
                for i, feature in enumerate(TARGET_FEATURES):
                    df = create_method_matrix_df(aggregated[feature], 'RNN', prob_type)
                    df.columns = [f'{feature}_{col}' for col in df.columns]
                    rnn_dfs.append(df)
                    if i < len(TARGET_FEATURES) - 1:
                        spacer = pd.DataFrame({'  ': [np.nan]*len(df)}, index=df.index)
                        rnn_dfs.append(spacer)
                combined_rnn = pd.concat(rnn_dfs, axis=1)
                sheet_name = f'{seed}_RNN_{prob_suffix}'
                combined_rnn.to_excel(writer, sheet_name=sheet_name)
    
    print(f"\nSaved all results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='MBR Analysis for LIBERO Task Suite')
    parser.add_argument('--rollouts_dir', type=str, 
                        default='/hdd2/kai/openvla-oft/rollouts',
                        help='Path to rollouts directory')
    parser.add_argument('--output_dir', type=str,
                        default='/hdd2/kai/openvla-oft/experiments/logs/mbr_analysis_results',
                        help='Path to output directory for xlsx file')
    parser.add_argument('--metric', type=str, choices=['Top-1_Prob', 'Top-3_Prob', 'both'],
                        default='Top-1_Prob',
                        help='Which probability metric to display')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save results to xlsx file')
    parser.add_argument('--first_n', type=int, default=None,
                        help='Only use first N timesteps (reads from Raw_Data sheet). '
                             'E.g., --first_n 3 uses timesteps 0,8,16')
    
    args = parser.parse_args()
    
    print(f"Looking for checkpoint folders in: {args.rollouts_dir}")
    if args.first_n is not None:
        print(f"Using first {args.first_n} timesteps only")
    
    checkpoint_folders = find_checkpoint_folders(args.rollouts_dir)
    
    if not checkpoint_folders:
        print("No checkpoint folders found!")
        return
    
    print(f"Found {len(checkpoint_folders)} checkpoint folders:")
    for folder in checkpoint_folders:
        seed = extract_seed_from_folder(folder)
        print(f"  - seed_{seed}: {folder}")
    
    all_results = {}
    
    for folder in checkpoint_folders:
        seed = extract_seed_from_folder(folder)
        print(f"\nProcessing checkpoint seed_{seed}...")
        
        aggregated = process_checkpoint(folder, first_n=args.first_n)
        
        if aggregated is None:
            print(f"  Skipping - no valid data")
            continue
        
        all_results[seed] = aggregated
        
        if args.metric == 'both':
            print_results(seed, aggregated, 'Top-1_Prob', first_n=args.first_n)
            print_results(seed, aggregated, 'Top-3_Prob', first_n=args.first_n)
        else:
            print_results(seed, aggregated, args.metric, first_n=args.first_n)
    
    if not args.no_save and all_results:
        save_all_results_to_xlsx(all_results, args.output_dir, first_n=args.first_n)


if __name__ == '__main__':
    main()