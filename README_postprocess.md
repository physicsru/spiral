# Qwen vs CFR+ Postprocess Analysis

This document describes how to use the postprocess analysis tools to extract episode win/loss information and create plots showing Qwen's performance over episodes.

## Files

- `postprocess_analysis.py` - Main analysis functions and command-line interface
- `demo_postprocess.py` - Demonstration script with examples
- `README_postprocess.md` - This documentation file

## Quick Start

### Basic Usage

```bash
# Analyze log files and create plots
python postprocess_analysis.py path/eval_qwen_vs_cfr+.*.out

# Analyze specific log files
python postprocess_analysis.py logfile1.out logfile2.out

# Run demo with example data
python demo_postprocess.py
```

### Expected Log Format

The analysis expects log files with episode results in this format:
```
Episode {episode_number}/{total}: Qwen Position: {0_or_1} Qwen {W|L|D} ({qwen_score}:{opponent_score})
```

Example:
```
Episode 4/10000: Qwen Position: 1 Qwen L (-1.0:0.0)
Episode 5/10000: Qwen Position: 0 Qwen W (0.0:-1.0)
```

## Generated Outputs

The analysis creates several outputs:

### 1. Comprehensive Plot (`qwen_vs_cfrplus_analysis.png`)

A 6-panel visualization showing:
- **Win/Loss Distribution** (pie chart)
- **Win/Loss Counts** (bar chart)  
- **Running Win Rate Over Episodes** (line plot)
- **Cumulative Wins vs Losses** (line plot)
- **Win Rate by Starting Position** (bar chart)
- **Episode Outcomes Over Time** (scatter plot)

### 2. Analysis Summary (`qwen_analysis_summary.json`)

JSON file containing:
- Overall statistics (win rate, total episodes, etc.)
- Position-specific analysis
- Final running win rate

## Functions

### Core Functions

#### `parse_episode_log(log_file_path: str) -> List[Dict]`
Parses a single log file and extracts episode data.

#### `parse_multiple_log_files(log_file_paths: List[str]) -> List[Dict]`
Parses multiple log files and combines results.

#### `analyze_performance(episodes: List[Dict]) -> Dict`
Analyzes episode data and computes statistics.

#### `plot_qwen_performance(analysis: Dict, save_path: str = None, show_plot: bool = False)`
Creates comprehensive performance plots.

#### `postprocess_qwen_analysis(log_files: List[str], ...) -> Dict`
Main function that orchestrates the complete analysis pipeline.

### Example Usage in Code

```python
from postprocess_analysis import postprocess_qwen_analysis

# Analyze log files
analysis = postprocess_qwen_analysis(
    log_files=["eval_log1.out", "eval_log2.out"],
    output_dir="results",
    save_plot=True,
    show_plot=False,
    save_summary=True
)

# Access results
print(f"Win rate: {analysis['win_rate']:.1%}")
print(f"Total episodes: {analysis['total_episodes']}")
```

## Command Line Options

```bash
python postprocess_analysis.py [options] log_file1 [log_file2 ...]

Options:
  -o, --output-dir DIR    Output directory for results (default: current directory)
  --no-plot              Don't save plot
  --show                 Display plot interactively
  --no-summary           Don't save JSON summary
```

## Analysis Features

### Performance Metrics
- Overall win/loss/draw rates
- Running win rate over episodes
- Cumulative wins vs losses
- Position-specific performance (first player vs second player)

### Visualizations
- Time series of episode outcomes
- Position bias analysis
- Statistical summaries
- Trend analysis

### Position Analysis
The tool analyzes whether Qwen performs differently as the first player (Position 0) versus the second player (Position 1), which can reveal:
- First-player advantage/disadvantage
- Model biases
- Game-specific effects

## Example Output

```
==================================================
QWEN VS CFR+ PERFORMANCE SUMMARY
==================================================
Total Episodes: 453
Wins: 183 (40.4%)
Losses: 270 (59.6%)
Draws: 0 (0.0%)

By Position:
Position 0 (First): 89/227 (39.2%)
Position 1 (Second): 94/226 (41.6%)
```

## Dependencies

Required Python packages:
- `matplotlib` - For plotting
- `numpy` - For numerical operations  
- `pathlib` - For path handling
- `json` - For JSON output
- `re` - For regex parsing
- `argparse` - For command line interface

Install dependencies:
```bash
pip install matplotlib numpy
```

## Troubleshooting

### No Episodes Found
- Check that log files contain lines matching the expected format
- Verify file paths are correct
- Ensure log files are readable

### Empty Plots
- Verify that episode data was successfully parsed
- Check that the analysis dictionary contains valid data
- Ensure matplotlib is properly installed

### Permission Errors
- Check write permissions for output directory
- Ensure output directory exists or can be created

## Integration with Existing Code

The postprocess functions can be integrated into existing training/evaluation pipelines:

```python
# At the end of your training script
from postprocess_analysis import postprocess_qwen_analysis

# Analyze the generated log files
analysis = postprocess_qwen_analysis(
    log_files=glob.glob("training_logs/*.out"),
    output_dir="final_analysis"
)

# Use analysis results for decision making
if analysis['win_rate'] > 0.6:
    print("Model performance is satisfactory")
else:
    print("Model needs further training")
``` 