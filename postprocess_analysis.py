#!/usr/bin/env python3
"""
Postprocess function to extract episode win/loss information from Qwen vs CFR+ log files
and create plots showing Qwen's performance over episodes.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import List, Dict, Tuple
from pathlib import Path
import json


def parse_episode_log(log_file_path: str) -> List[Dict]:
    """
    Parse episode results from log files.
    
    Expected log format:
    Episode {episode_number}/{total}: Qwen Position: {0_or_1} Qwen {W|L|D} ({qwen_score}:{opponent_score})
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        List of episode data dictionaries
    """
    episodes = []
    
    # Regex pattern to match episode results
    pattern = r"Episode (\d+)/(\d+): Qwen Position: ([01]) Qwen ([WLD]) \(([^:]+):([^)]+)\)"
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                match = re.search(pattern, line)
                if match:
                    episode_num = int(match.group(1))
                    total_episodes = int(match.group(2))
                    qwen_position = int(match.group(3))
                    outcome = match.group(4)  # W, L, or D
                    qwen_score = float(match.group(5))
                    opponent_score = float(match.group(6))
                    
                    episodes.append({
                        'episode': episode_num,
                        'total_episodes': total_episodes,
                        'qwen_position': qwen_position,
                        'outcome': outcome,
                        'qwen_score': qwen_score,
                        'opponent_score': opponent_score,
                        'line_number': line_num
                    })
                    
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file_path}")
        return []
    except Exception as e:
        print(f"Error reading log file {log_file_path}: {e}")
        return []
    
    return episodes


def parse_multiple_log_files(log_file_paths: List[str]) -> List[Dict]:
    """
    Parse multiple log files and combine results.
    
    Args:
        log_file_paths: List of paths to log files
        
    Returns:
        Combined list of episode data
    """
    all_episodes = []
    
    for log_path in log_file_paths:
        print(f"Parsing {log_path}...")
        episodes = parse_episode_log(log_path)
        all_episodes.extend(episodes)
    
    # Sort by episode number
    all_episodes.sort(key=lambda x: x['episode'])
    
    return all_episodes


def analyze_performance(episodes: List[Dict]) -> Dict:
    """
    Analyze Qwen's performance from episode data.
    
    Args:
        episodes: List of episode dictionaries
        
    Returns:
        Analysis results dictionary
    """
    if not episodes:
        return {}
    
    total_episodes = len(episodes)
    wins = sum(1 for ep in episodes if ep['outcome'] == 'W')
    losses = sum(1 for ep in episodes if ep['outcome'] == 'L')
    draws = sum(1 for ep in episodes if ep['outcome'] == 'D')
    
    win_rate = wins / total_episodes if total_episodes > 0 else 0
    loss_rate = losses / total_episodes if total_episodes > 0 else 0
    draw_rate = draws / total_episodes if total_episodes > 0 else 0
    
    # Calculate running statistics
    running_wins = []
    running_losses = []
    running_win_minus_loss = []
    
    win_count = 0
    loss_count = 0
    
    for i, episode in enumerate(episodes):
        if episode['outcome'] == 'W':
            win_count += 1
        elif episode['outcome'] == 'L':
            loss_count += 1
            
        running_wins.append(win_count)
        running_losses.append(loss_count)
        current_win_minus_loss = win_count - loss_count
        running_win_minus_loss.append(current_win_minus_loss)
    
    # Analyze by position
    position_0_episodes = [ep for ep in episodes if ep['qwen_position'] == 0]
    position_1_episodes = [ep for ep in episodes if ep['qwen_position'] == 1]
    
    position_0_wins = sum(1 for ep in position_0_episodes if ep['outcome'] == 'W')
    position_1_wins = sum(1 for ep in position_1_episodes if ep['outcome'] == 'W')
    
    position_0_win_rate = position_0_wins / len(position_0_episodes) if position_0_episodes else 0
    position_1_win_rate = position_1_wins / len(position_1_episodes) if position_1_episodes else 0
    
    return {
        'total_episodes': total_episodes,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'draw_rate': draw_rate,
        'running_wins': running_wins,
        'running_losses': running_losses,
        'running_win_minus_loss': running_win_minus_loss,
        'position_0_episodes': len(position_0_episodes),
        'position_1_episodes': len(position_1_episodes),
        'position_0_wins': position_0_wins,
        'position_1_wins': position_1_wins,
        'position_0_win_rate': position_0_win_rate,
        'position_1_win_rate': position_1_win_rate,
        'episodes': episodes
    }


def plot_qwen_performance(analysis: Dict, save_path: str | None = None, show_plot: bool = False):
    """
    Create comprehensive plots of Qwen's performance.
    
    Args:
        analysis: Analysis results from analyze_performance()
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    if not analysis or not analysis['episodes']:
        print("No episode data to plot")
        return
    
    episodes = analysis['episodes']
    episode_numbers = [ep['episode'] for ep in episodes]
    
    # Create a comprehensive plot with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Win/Loss distribution (pie chart)
    ax1 = axes[0, 0]
    labels = ['Wins', 'Losses', 'Draws']
    sizes = [analysis['wins'], analysis['losses'], analysis['draws']]
    colors = ['green', 'red', 'gray']
    
    # Only include non-zero categories
    filtered_data = [(label, size, color) for label, size, color in zip(labels, sizes, colors) if size > 0]
    if filtered_data:
        labels_filtered, sizes_filtered, colors_filtered = zip(*filtered_data)
        ax1.pie(sizes_filtered, labels=labels_filtered, colors=colors_filtered, 
                autopct='%1.1f%%', startangle=90)
    ax1.set_title('Overall Win/Loss Distribution')
    
    # 2. Win/Loss counts (bar chart)
    ax2 = axes[0, 1]
    bars = ax2.bar(labels, sizes, color=colors)
    ax2.set_title('Win/Loss Counts')
    ax2.set_ylabel('Number of Episodes')
    
    # Add count labels on bars
    for bar, count in zip(bars, sizes):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
    
    # 3. Running wins minus losses over episodes
    ax3 = axes[0, 2]
    ax3.plot(episode_numbers, analysis['running_win_minus_loss'], 'b-', linewidth=2, alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Neutral (0)')
    ax3.set_xlabel('Episode Number')
    ax3.set_ylabel('Wins - Losses')
    ax3.set_title('Running Wins Minus Losses Over Episodes')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Cumulative wins vs losses
    ax4 = axes[1, 0]
    ax4.plot(episode_numbers, analysis['running_wins'], 'g-', linewidth=2, label='Cumulative Wins')
    ax4.plot(episode_numbers, analysis['running_losses'], 'r-', linewidth=2, label='Cumulative Losses')
    ax4.set_xlabel('Episode Number')
    ax4.set_ylabel('Cumulative Count')
    ax4.set_title('Cumulative Wins vs Losses')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Win rate by position
    ax5 = axes[1, 1]
    positions = ['Position 0\n(First Player)', 'Position 1\n(Second Player)']
    position_win_rates = [analysis['position_0_win_rate'], analysis['position_1_win_rate']]
    position_counts = [analysis['position_0_episodes'], analysis['position_1_episodes']]
    
    bars = ax5.bar(positions, position_win_rates, color=['lightblue', 'lightcoral'])
    ax5.set_ylabel('Win Rate')
    ax5.set_title('Win Rate by Starting Position')
    ax5.set_ylim(0, 1)
    
    # Add win rate and episode count labels
    for i, (bar, rate, count) in enumerate(zip(bars, position_win_rates, position_counts)):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}\n({count} episodes)', ha='center', va='bottom')
    
    # 6. Wins minus losses over time by position (curve plot)
    ax6 = axes[1, 2]
    
    # Calculate running wins minus losses for each position
    position_0_win_minus_loss = []
    position_1_win_minus_loss = []
    pos0_wins, pos0_losses = 0, 0
    pos1_wins, pos1_losses = 0, 0
    
    position_0_episodes_num = []
    position_1_episodes_num = []
    
    for ep in episodes:
        if ep['qwen_position'] == 0:
            if ep['outcome'] == 'W':
                pos0_wins += 1
            elif ep['outcome'] == 'L':
                pos0_losses += 1
            position_0_win_minus_loss.append(pos0_wins - pos0_losses)
            position_0_episodes_num.append(ep['episode'])
        else:  # position 1
            if ep['outcome'] == 'W':
                pos1_wins += 1
            elif ep['outcome'] == 'L':
                pos1_losses += 1
            position_1_win_minus_loss.append(pos1_wins - pos1_losses)
            position_1_episodes_num.append(ep['episode'])
    
    # Plot curves for each position
    if position_0_episodes_num:
        ax6.plot(position_0_episodes_num, position_0_win_minus_loss, 'b-', linewidth=2, 
                label='Position 0 (First Player)', alpha=0.8)
    if position_1_episodes_num:
        ax6.plot(position_1_episodes_num, position_1_win_minus_loss, 'orange', linewidth=2, 
                label='Position 1 (Second Player)', alpha=0.8)
    
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Neutral (0)')
    ax6.set_xlabel('Episode Number')
    ax6.set_ylabel('Wins - Losses')
    ax6.set_title('Running Wins Minus Losses by Position')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    
    # Add summary text
    summary_text = (
        f"Total Episodes: {analysis['total_episodes']}\n"
        f"Overall Win Rate: {analysis['win_rate']:.1%}\n"
        f"Wins: {analysis['wins']}, Losses: {analysis['losses']}, Draws: {analysis['draws']}"
    )
    
    fig.suptitle(f'Qwen vs CFR+ Performance Analysis\n{summary_text}', fontsize=14, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def save_analysis_summary(analysis: Dict, save_path: str):
    """
    Save analysis summary to a JSON file.
    
    Args:
        analysis: Analysis results
        save_path: Path to save the JSON file
    """
    # Create a summary without the full episodes list for cleaner JSON
    summary = {
        'total_episodes': analysis['total_episodes'],
        'wins': analysis['wins'],
        'losses': analysis['losses'],
        'draws': analysis['draws'],
        'win_rate': analysis['win_rate'],
        'loss_rate': analysis['loss_rate'],
        'draw_rate': analysis['draw_rate'],
        'position_analysis': {
            'position_0_episodes': analysis['position_0_episodes'],
            'position_1_episodes': analysis['position_1_episodes'],
            'position_0_wins': analysis['position_0_wins'],
            'position_1_wins': analysis['position_1_wins'],
            'position_0_win_rate': analysis['position_0_win_rate'],
            'position_1_win_rate': analysis['position_1_win_rate']
        },
        'final_running_win_minus_loss': analysis['running_win_minus_loss'][-1] if analysis['running_win_minus_loss'] else 0
    }
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis summary saved to: {save_path}")


def postprocess_qwen_analysis(log_files: List[str], output_dir: str = ".", 
                            save_plot: bool = True, show_plot: bool = False,
                            save_summary: bool = True) -> Dict:
    """
    Main postprocess function to analyze Qwen vs CFR+ episodes and create plots.
    
    Args:
        log_files: List of log file paths to analyze
        output_dir: Directory to save outputs
        save_plot: Whether to save the plot
        show_plot: Whether to display the plot
        save_summary: Whether to save analysis summary
        
    Returns:
        Analysis results dictionary
    """
    print(f"Starting postprocess analysis of {len(log_files)} log files...")
    
    # Parse all log files
    episodes = parse_multiple_log_files(log_files)
    
    if not episodes:
        print("No episode data found in log files")
        return {}
    
    print(f"Found {len(episodes)} episodes to analyze")
    
    # Analyze performance
    analysis = analyze_performance(episodes)
    
    # Print summary
    print("\n" + "="*50)
    print("QWEN VS CFR+ PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Total Episodes: {analysis['total_episodes']}")
    print(f"Wins: {analysis['wins']} ({analysis['win_rate']:.1%})")
    print(f"Losses: {analysis['losses']} ({analysis['loss_rate']:.1%})")
    print(f"Draws: {analysis['draws']} ({analysis['draw_rate']:.1%})")
    print(f"\nBy Position:")
    print(f"Position 0 (First): {analysis['position_0_wins']}/{analysis['position_0_episodes']} ({analysis['position_0_win_rate']:.1%})")
    print(f"Position 1 (Second): {analysis['position_1_wins']}/{analysis['position_1_episodes']} ({analysis['position_1_win_rate']:.1%})")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create and save plot
    if save_plot or show_plot:
        plot_path = str(output_path / "qwen_vs_cfrplus_analysis.png") if save_plot else None
        plot_qwen_performance(analysis, save_path=plot_path, show_plot=show_plot)
    
    # Save analysis summary
    if save_summary:
        summary_path = str(output_path / "qwen_analysis_summary.json")
        save_analysis_summary(analysis, summary_path)
    
    return analysis


def main():
    """Command line interface for the postprocess function."""
    parser = argparse.ArgumentParser(description="Postprocess Qwen vs CFR+ episode logs")
    parser.add_argument("log_files", nargs="+", help="Log files to analyze")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory for results")
    parser.add_argument("--no-plot", action="store_true", help="Don't save plot")
    parser.add_argument("--show", action="store_true", help="Display plot")
    parser.add_argument("--no-summary", action="store_true", help="Don't save summary JSON")
    
    args = parser.parse_args()
    
    # Run analysis
    postprocess_qwen_analysis(
        log_files=args.log_files,
        output_dir=args.output_dir,
        save_plot=not args.no_plot,
        show_plot=args.show,
        save_summary=not args.no_summary
    )


if __name__ == "__main__":
    main() 