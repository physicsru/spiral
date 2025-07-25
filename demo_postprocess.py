#!/usr/bin/env python3
"""
Demonstration script for using the postprocess_analysis.py functions.
This shows how to analyze Qwen vs CFR+ episode logs and create plots.
"""

import os
from pathlib import Path
from postprocess_analysis import postprocess_qwen_analysis, parse_episode_log
import glob


def demo_with_example_logs():
    """
    Demonstrate the postprocess function with example log files.
    """
    print("=== Qwen vs CFR+ Postprocess Analysis Demo ===\n")
    
    # Look for log files in the path directory
    log_patterns = [
        "path/eval_qwen_vs_cfr+.*.out",
        "path/*.out",
        "*.out"
    ]
    
    log_files = []
    for pattern in log_patterns:
        found_files = glob.glob(pattern)
        log_files.extend(found_files)
    
    if not log_files:
        print("No log files found. Creating a demo with simulated data...")
        create_demo_log_file()
        log_files = ["demo_log.txt"]
    else:
        print(f"Found {len(log_files)} log files:")
        for log_file in log_files[:5]:  # Show first 5
            print(f"  - {log_file}")
        if len(log_files) > 5:
            print(f"  ... and {len(log_files) - 5} more")
    
    print(f"\nAnalyzing {len(log_files)} log files...")
    
    # Run the postprocess analysis
    analysis = postprocess_qwen_analysis(
        log_files=log_files,
        output_dir="analysis_results",
        save_plot=True,
        show_plot=False,  # Set to True if you want to display the plot
        save_summary=True
    )
    
    if analysis:
        print("\n=== Analysis Complete ===")
        print("Generated files:")
        print("  - analysis_results/qwen_vs_cfrplus_analysis.png")
        print("  - analysis_results/qwen_analysis_summary.json")
        
        # Show key insights
        print(f"\n=== Key Insights ===")
        print(f"Overall win rate: {analysis['win_rate']:.1%}")
        
        # Check if there's a position bias
        pos0_rate = analysis['position_0_win_rate']
        pos1_rate = analysis['position_1_win_rate']
        if abs(pos0_rate - pos1_rate) > 0.1:  # 10% difference
            better_pos = "Position 0 (First)" if pos0_rate > pos1_rate else "Position 1 (Second)"
            print(f"Position bias detected: {better_pos} performs better")
            print(f"  Position 0: {pos0_rate:.1%} win rate")
            print(f"  Position 1: {pos1_rate:.1%} win rate")
        else:
            print("No significant position bias detected")
    
    return analysis


def create_demo_log_file():
    """
    Create a demo log file with simulated episode results for testing.
    """
    demo_content = """
INFO 07-04 01:13:14 [core_client.py:435] Core engine process 0 ready.

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  4.93it/s, est. speed input: 6110.37 toks/s, output: 157.68 toks/s]
Episode 1/100: Qwen Position: 0 Qwen W (0.0:-1.0)

Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  4.92it/s, est. speed input: 6511.53 toks/s, output: 157.49 toks/s]
Episode 2/100: Qwen Position: 1 Qwen L (-1.0:0.0)

Episode 3/100: Qwen Position: 0 Qwen W (0.0:-1.0)
Episode 4/100: Qwen Position: 1 Qwen L (-1.0:0.0)
Episode 5/100: Qwen Position: 0 Qwen L (-1.0:0.0)
Episode 6/100: Qwen Position: 1 Qwen W (0.0:-1.0)
Episode 7/100: Qwen Position: 0 Qwen W (0.0:-1.0)
Episode 8/100: Qwen Position: 1 Qwen W (0.0:-1.0)
Episode 9/100: Qwen Position: 0 Qwen L (-1.0:0.0)
Episode 10/100: Qwen Position: 1 Qwen L (-1.0:0.0)
Episode 11/100: Qwen Position: 0 Qwen W (0.0:-1.0)
Episode 12/100: Qwen Position: 1 Qwen W (0.0:-1.0)
Episode 13/100: Qwen Position: 0 Qwen L (-1.0:0.0)
Episode 14/100: Qwen Position: 1 Qwen L (-1.0:0.0)
Episode 15/100: Qwen Position: 0 Qwen W (0.0:-1.0)
Episode 16/100: Qwen Position: 1 Qwen L (-1.0:0.0)
Episode 17/100: Qwen Position: 0 Qwen W (0.0:-1.0)
Episode 18/100: Qwen Position: 1 Qwen W (0.0:-1.0)
Episode 19/100: Qwen Position: 0 Qwen L (-1.0:0.0)
Episode 20/100: Qwen Position: 1 Qwen W (0.0:-1.0)
""".strip()
    
    with open("demo_log.txt", "w") as f:
        f.write(demo_content)
    
    print("Created demo log file: demo_log.txt")


def analyze_single_log_file(log_file_path: str):
    """
    Demonstrate analyzing a single log file.
    """
    print(f"\n=== Analyzing Single Log File: {log_file_path} ===")
    
    episodes = parse_episode_log(log_file_path)
    
    if not episodes:
        print("No episodes found in the log file")
        return
    
    print(f"Found {len(episodes)} episodes")
    print(f"Episode range: {episodes[0]['episode']} to {episodes[-1]['episode']}")
    
    # Quick stats
    wins = sum(1 for ep in episodes if ep['outcome'] == 'W')
    losses = sum(1 for ep in episodes if ep['outcome'] == 'L')
    draws = sum(1 for ep in episodes if ep['outcome'] == 'D')
    
    print(f"Quick stats: {wins} wins, {losses} losses, {draws} draws")
    print(f"Win rate: {wins/len(episodes):.1%}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Analyze specific log file(s) provided as arguments
        log_files = sys.argv[1:]
        print(f"Analyzing {len(log_files)} log file(s)...")
        
        analysis = postprocess_qwen_analysis(
            log_files=log_files,
            output_dir="analysis_results",
            save_plot=True,
            show_plot=False,
            save_summary=True
        )
    else:
        # Run demo
        analysis = demo_with_example_logs() 