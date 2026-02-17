#!/usr/bin/env python3
"""
CS 325 - Runtime Analysis Experiment
Author: Abraham Smith

This module conducts empirical runtime experiments on the edit distance algorithm
to verify the theoretical time complexity of O(mn) where m and n are sequence lengths.
"""

import time
import random
import matplotlib.pyplot as plt
import numpy as np
from edit_distance import EditDistance, read_cost_matrix
from typing import List, Tuple
import csv


def generate_random_sequence(length: int, alphabet: str = "ATGC") -> str:
    """
    Generate a random DNA sequence of specified length.
    
    Args:
        length: Length of sequence to generate
        alphabet: Character set to use (default: DNA bases)
        
    Returns:
        Random sequence string
    """
    return ''.join(random.choice(alphabet) for _ in range(length))


def measure_runtime(calculator: EditDistance, seq1: str, seq2: str, trials: int = 3) -> float:
    """
    Measure average runtime for computing edit distance.
    
    Args:
        calculator: EditDistance instance
        seq1: First sequence
        seq2: Second sequence
        trials: Number of trials to average over
        
    Returns:
        Average runtime in seconds
    """
    times = []
    for _ in range(trials):
        start_time = time.perf_counter()
        calculator.compute_edit_distance(seq1, seq2)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return sum(times) / len(times)


def run_empirical_analysis(cost_file: str = 'imp2cost.txt') -> List[Tuple[int, float, float]]:
    """
    Run empirical analysis as required for the assignment.
    
    Tests sequence lengths: 500, 1000, 2000, 4000, 5000
    For each length, generates 10 random pairs using alphabet A, G, T, C
    Reports average runtime for each length.
    
    Args:
        cost_file: Path to cost matrix file
        
    Returns:
        List of (sequence_length, average_time, std_deviation) tuples
    """
    # Load cost matrix
    cost_matrix, x_chars, y_chars = read_cost_matrix(cost_file)
    calculator = EditDistance(cost_matrix, x_chars, y_chars)
    
    # Required test lengths and parameters
    test_lengths = [500, 1000, 2000, 4000, 5000]
    pairs_per_length = 10
    alphabet = ['A', 'G', 'T', 'C']  # DNA alphabet as specified
    
    results = []
    
    print("CS 325 - Empirical Runtime Analysis")
    print("=" * 50)
    print(f"Testing lengths: {test_lengths}")
    print(f"Pairs per length: {pairs_per_length}")
    print(f"Alphabet: {alphabet}")
    print()
    
    for length in test_lengths:
        print(f"Testing sequence length {length}...")
        times = []
        
        for pair_num in range(pairs_per_length):
            # Generate random sequence pair of the same length
            seq1 = generate_random_sequence(length, alphabet)
            seq2 = generate_random_sequence(length, alphabet)
            
            # Measure runtime
            start_time = time.perf_counter()
            calculator.compute_edit_distance(seq1, seq2)
            end_time = time.perf_counter()
            
            runtime = end_time - start_time
            times.append(runtime)
            
            print(f"  Pair {pair_num+1}/10: {runtime:.6f}s")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        std_dev = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
        
        results.append((length, avg_time, std_dev))
        print(f"  Average: {avg_time:.6f}s (+/-{std_dev:.6f}s)")
        print()
    
    return results


def create_empirical_plot(results: List[Tuple[int, float, float]], output_file: str = 'empirical_runtime.png') -> None:
    """
    Create plot specifically for empirical runtime analysis showing polynomial order.
    
    Args:
        results: List of (length, avg_time, std_dev) tuples
        output_file: Output file name
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        lengths = [r[0] for r in results]
        avg_times = [r[1] for r in results]
        std_devs = [r[2] for r in results]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Empirical Runtime Analysis: Edit Distance Algorithm', fontsize=16)
        
        # Plot 1: Linear scale - Runtime vs Sequence Length
        ax1.errorbar(lengths, avg_times, yerr=std_devs, fmt='o-', 
                    capsize=5, linewidth=2, markersize=8, 
                    color='blue', label='Measured Runtime')
        
        # Fit polynomial curves for comparison
        lengths_np = np.array(lengths)
        avg_times_np = np.array(avg_times)
        
        # Quadratic fit (O(n^2))
        quad_coeffs = np.polyfit(lengths_np, avg_times_np, 2)
        quad_fit = np.polyval(quad_coeffs, lengths_np)
        ax1.plot(lengths, quad_fit, '--', color='red', linewidth=2, 
                label=f'Quadratic fit O(n^2)')
        
        # Linear fit for comparison
        lin_coeffs = np.polyfit(lengths_np, avg_times_np, 1)
        lin_fit = np.polyval(lin_coeffs, lengths_np)
        ax1.plot(lengths, lin_fit, ':', color='green', linewidth=2, 
                label=f'Linear fit O(n)')
        
        ax1.set_xlabel('Sequence Length (n)', fontsize=12)
        ax1.set_ylabel('Average Runtime (seconds)', fontsize=12)
        ax1.set_title('Runtime vs Sequence Length', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Log-log scale - Clearly shows polynomial order
        ax2.loglog(lengths, avg_times, 'o-', linewidth=2, markersize=8, 
                  color='blue', label='Measured Runtime')
        
        # Theoretical lines for comparison
        lengths_extended = np.logspace(np.log10(min(lengths)), np.log10(max(lengths)), 100)
        
        # O(n) line (normalized to data)
        c1 = avg_times[0] / lengths[0]
        ax2.loglog(lengths_extended, c1 * lengths_extended, '--', 
                  color='green', alpha=0.7, label='Theoretical O(n)')
        
        # O(n^2) line (normalized to data)
        c2 = avg_times[0] / (lengths[0] ** 2)
        ax2.loglog(lengths_extended, c2 * (lengths_extended ** 2), '--', 
                  color='red', alpha=0.7, label='Theoretical O(n^2)')
        
        ax2.set_xlabel('Sequence Length (log scale)', fontsize=12)
        ax2.set_ylabel('Runtime (log scale)', fontsize=12)
        ax2.set_title('Log-Log Plot: Polynomial Order Verification', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Empirical analysis plot saved to: {output_file}")
        
        # Calculate and display complexity analysis
        print(f"\nComplexity Analysis:")
        print(f"Linear fit R^2: {np.corrcoef(lengths, lin_fit)[0,1]**2:.4f}")
        print(f"Quadratic fit R^2: {np.corrcoef(lengths, quad_fit)[0,1]**2:.4f}")
        
    except ImportError:
        print("Matplotlib not available - install with: pip install matplotlib")
    except Exception as e:
        print(f"Error creating plot: {e}")


def save_empirical_results(results: List[Tuple[int, float, float]], filename: str = 'empirical_results.csv') -> None:
    """
    Save empirical results in the format expected for the report.
    
    Args:
        results: List of (length, avg_time, std_dev) tuples
        filename: Output CSV filename
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sequence_Length', 'Average_Runtime_Seconds', 'Standard_Deviation', 'Pairs_Tested'])
        
        for length, avg_time, std_dev in results:
            writer.writerow([length, f"{avg_time:.8f}", f"{std_dev:.8f}", 10])
    
    print(f"Empirical results saved to: {filename}")
    
    # Also print formatted table for report
    print(f"\nFormatted Results Table:")
    print(f"{'Length':>8} | {'Avg Time (s)':>12} | {'Std Dev (s)':>12} | {'Pairs'}")
    print(f"-" * 50)
    for length, avg_time, std_dev in results:
        print(f"{length:>8} | {avg_time:>12.6f} | {std_dev:>12.6f} | {10:>5}")


def run_runtime_experiment(
    cost_file: str = 'imp2cost.txt',
    max_length: int = 200,
    step_size: int = 20,
    trials: int = 3
) -> List[Tuple[int, int, float]]:
    """
    Run comprehensive runtime experiments with varying sequence lengths.
    
    Args:
        cost_file: Path to cost matrix file
        max_length: Maximum sequence length to test
        step_size: Increment between test sizes
        trials: Number of trials per size combination
        
    Returns:
        List of (length1, length2, average_time) tuples
    """
    # Load cost matrix
    cost_matrix, x_chars, y_chars = read_cost_matrix(cost_file)
    calculator = EditDistance(cost_matrix, x_chars, y_chars)
    
    # Available characters for sequence generation
    alphabet = list(x_chars.keys())
    alphabet.remove('-')  # Remove gap character
    
    results = []
    lengths = list(range(step_size, max_length + 1, step_size))
    
    print(f"Running runtime experiments...")
    print(f"Testing lengths: {lengths}")
    print(f"Trials per combination: {trials}")
    print()
    
    total_combinations = len(lengths) * len(lengths)
    current_combination = 0
    
    for len1 in lengths:
        for len2 in lengths:
            current_combination += 1
            
            # Generate random sequences
            seq1 = generate_random_sequence(len1, alphabet)
            seq2 = generate_random_sequence(len2, alphabet)
            
            # Measure runtime
            avg_time = measure_runtime(calculator, seq1, seq2, trials)
            results.append((len1, len2, avg_time))
            
            print(f"Progress: {current_combination}/{total_combinations} "
                  f"- Lengths ({len1}, {len2}): {avg_time:.6f}s")
    
    return results


def analyze_complexity(results: List[Tuple[int, int, float]], create_plot: bool = True) -> None:
    """
    Analyze the empirical complexity of the algorithm.
    
    Args:
        results: Runtime experiment results
        create_plot: Whether to generate visualization plots
    """
    print("\nComplexity Analysis:")
    print("=" * 50)
    
    # Extract data for analysis
    data = np.array(results)
    len1_vals = data[:, 0]
    len2_vals = data[:, 1]
    times = data[:, 2]
    
    # Compute mn products
    mn_products = len1_vals * len2_vals
    
    # Fit linear model: time = a * (m*n) + b
    # This should give us a good fit if complexity is O(mn)
    coeffs = np.polyfit(mn_products, times, 1)
    slope, intercept = coeffs
    
    # R-squared calculation
    y_pred = slope * mn_products + intercept
    ss_res = np.sum((times - y_pred) ** 2)
    ss_tot = np.sum((times - np.mean(times)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"Linear fit to time vs (m*n):")
    print(f"  Slope: {slope:.2e} seconds per unit")
    print(f"  Intercept: {intercept:.6f} seconds")
    print(f"  R^2: {r_squared:.4f}")
    
    # Test quadratic fit for comparison: time = a * (m*n)^2 + b * (m*n) + c
    coeffs_quad = np.polyfit(mn_products, times, 2)
    a_quad, b_quad, c_quad = coeffs_quad
    y_pred_quad = a_quad * mn_products**2 + b_quad * mn_products + c_quad
    ss_res_quad = np.sum((times - y_pred_quad) ** 2)
    r_squared_quad = 1 - (ss_res_quad / ss_tot)
    
    print(f"\nQuadratic fit to time vs (m*n):")
    print(f"  Quadratic coefficient: {a_quad:.2e}")
    print(f"  Linear coefficient: {b_quad:.2e}")
    print(f"  Constant: {c_quad:.6f}")
    print(f"  R^2: {r_squared_quad:.4f}")
    
    # Compare fits
    print(f"\nModel Comparison:")
    print(f"  Linear model R^2: {r_squared:.4f}")
    print(f"  Quadratic model R^2: {r_squared_quad:.4f}")
    
    if r_squared > 0.85:
        print(f"  Strong linear relationship supports O(mn) complexity")
    else:
        print(f"  Weak linear relationship - may indicate measurement noise")
    
    # Generate plot if requested
    if create_plot:
        try:
            print(f"\nGenerating runtime visualization...")
            plot_runtime_results(results)
        except ImportError as e:
            print(f"\n⚠ Could not generate plots - matplotlib not available")
            print(f"Install with: pip install matplotlib")
        except Exception as e:
            print(f"\n⚠ Error generating plots: {e}")


def plot_runtime_results(results: List[Tuple[int, int, float]], output_file: str = 'runtime_plot.png') -> None:
    """
    Create visualization plots for runtime analysis.
    
    Args:
        results: Runtime experiment results
        output_file: Output file for the plot
    """
    data = np.array(results)
    len1_vals = data[:, 0]
    len2_vals = data[:, 1]
    times = data[:, 2]
    mn_products = len1_vals * len2_vals
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Runtime vs m*n (main complexity analysis)
    ax1.scatter(mn_products, times, alpha=0.6, s=30)
    
    # Fit and plot trend line
    coeffs = np.polyfit(mn_products, times, 1)
    slope, intercept = coeffs
    trend_line = slope * mn_products + intercept
    ax1.plot(mn_products, trend_line, 'r--', alpha=0.8, 
             label=f'Linear fit: y = {slope:.2e}x + {intercept:.6f}')
    
    ax1.set_xlabel('Product of Sequence Lengths (m * n)')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('Runtime vs Sequence Length Product\n(Testing O(mn) Complexity)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Runtime vs individual sequence lengths (heatmap style)
    unique_len1 = sorted(set(len1_vals))
    unique_len2 = sorted(set(len2_vals))
    
    # Create time matrix for heatmap
    time_matrix = np.zeros((len(unique_len1), len(unique_len2)))
    for i, l1 in enumerate(unique_len1):
        for j, l2 in enumerate(unique_len2):
            # Find corresponding time
            for len1, len2, t in results:
                if len1 == l1 and len2 == l2:
                    time_matrix[i][j] = t
                    break
    
    # Heatmap showing how runtime varies with both sequence lengths
    im = ax2.imshow(time_matrix, cmap='viridis', aspect='auto')
    ax2.set_xlabel('Length of Sequence 2 (n)')
    ax2.set_ylabel('Length of Sequence 1 (m)')
    ax2.set_title('Runtime Heatmap')
    ax2.set_xticks(range(len(unique_len2)))
    ax2.set_xticklabels(unique_len2)
    ax2.set_yticks(range(len(unique_len1)))
    ax2.set_yticklabels(unique_len1)
    plt.colorbar(im, ax=ax2, label='Runtime (seconds)')
    
    # Plot 3: Runtime vs max(m,n)
    max_lens = np.maximum(len1_vals, len2_vals)
    ax3.scatter(max_lens, times, alpha=0.6, s=30)
    ax3.set_xlabel('Max Sequence Length max(m,n)')
    ax3.set_ylabel('Runtime (seconds)')
    ax3.set_title('Runtime vs Maximum Sequence Length')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Log-log plot for complexity verification
    ax4.loglog(mn_products, times, 'o', alpha=0.6, markersize=4)
    ax4.set_xlabel('Product of Sequence Lengths (m * n) [log scale]')
    ax4.set_ylabel('Runtime (seconds) [log scale]')
    ax4.set_title('Log-Log Plot for Complexity Analysis')
    ax4.grid(True, alpha=0.3)
    
    # Add theoretical O(mn) line
    min_mn, max_mn = min(mn_products), max(mn_products)
    theoretical_x = np.logspace(np.log10(min_mn), np.log10(max_mn), 100)
    # Normalize to match data scale
    scaling_factor = np.median(times) / np.median(mn_products)
    theoretical_y = scaling_factor * theoretical_x
    ax4.loglog(theoretical_x, theoretical_y, 'r--', alpha=0.8, label='Theoretical O(mn)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nRuntime analysis plot saved to: {output_file}")


def create_simple_plot(results: List[Tuple[int, int, float]], title: str = "Runtime Analysis") -> None:
    """
    Create a clean 2x2 plot for runtime analysis.
    
    Args:
        results: Runtime experiment results
        title: Title for the plot
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        data = np.array(results)
        len1_vals = data[:, 0]
        len2_vals = data[:, 1] 
        times = data[:, 2]
        mn_products = len1_vals * len2_vals
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title, fontsize=14)
        
        # Plot 1: Main complexity plot - Runtime vs m*n
        ax1.scatter(mn_products, times, alpha=0.7, s=40, color='blue')
        coeffs = np.polyfit(mn_products, times, 1)
        trend_line = coeffs[0] * mn_products + coeffs[1]
        ax1.plot(mn_products, trend_line, 'r--', linewidth=2, 
                label=f'Linear fit (R^2 = {np.corrcoef(mn_products, times)[0,1]**2:.3f})')
        ax1.set_xlabel('Product of Sequence Lengths (m * n)')
        ax1.set_ylabel('Runtime (seconds)')
        ax1.set_title('Runtime vs m*n (Complexity Verification)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Runtime vs first sequence length
        ax2.scatter(len1_vals, times, alpha=0.7, s=40, color='green')
        ax2.set_xlabel('Sequence 1 Length (m)')
        ax2.set_ylabel('Runtime (seconds)')
        ax2.set_title('Runtime vs First Sequence Length')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Runtime vs second sequence length
        ax3.scatter(len2_vals, times, alpha=0.7, s=40, color='orange')
        ax3.set_xlabel('Sequence 2 Length (n)')
        ax3.set_ylabel('Runtime (seconds)')
        ax3.set_title('Runtime vs Second Sequence Length')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Log-log plot for complexity confirmation
        ax4.loglog(mn_products, times, 'o', alpha=0.7, markersize=6, color='purple')
        ax4.set_xlabel('m * n (log scale)')
        ax4.set_ylabel('Runtime (log scale)')
        ax4.set_title('Log-Log Plot (O(mn) Check)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = 'runtime_analysis.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Runtime analysis plot saved to: {filename}")
        
    except ImportError:
        print("Matplotlib not available - install with: pip install matplotlib")
    except Exception as e:
        print(f"Error creating plot: {e}")


def save_results_csv(results: List[Tuple[int, int, float]], filename: str = 'runtime_results.csv') -> None:
    """
    Save runtime results to CSV file.
    
    Args:
        results: Runtime experiment results
        filename: Output CSV filename
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sequence_1_Length', 'Sequence_2_Length', 'Runtime_Seconds', 'Product_mn'])
        
        for len1, len2, runtime in results:
            writer.writerow([len1, len2, runtime, len1 * len2])
    
    print(f"Runtime results saved to: {filename}")


def main():
    """Main function to run empirical runtime analysis as specified in assignment."""
    print("CS 325 - Edit Distance Empirical Runtime Analysis")
    print("=" * 60)
    print("Assignment Requirements:")
    print("- Sequence lengths: 500, 1000, 2000, 4000, 5000")
    print("- 10 random pairs per length") 
    print("- Alphabet: A, G, T, C")
    print("- Show polynomial order of runtime")
    print()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run empirical analysis
    print("Starting empirical analysis...")
    results = run_empirical_analysis('imp2cost.txt')
    
    # Save results
    save_empirical_results(results)
    
    # Create visualization
    create_empirical_plot(results)
    
    print("\nEmpirical Analysis Complete!")
    print("Files generated:")
    print("  - empirical_results.csv: Detailed runtime data")
    print("  - empirical_runtime.png: Polynomial order visualization")


def run_quick_test():
    """Run a quick test with smaller data for development/testing."""
    print("Quick Test - Smaller Scale Analysis")
    print("=" * 40)
    
    # Smaller test for quick verification
    results = run_runtime_experiment(
        cost_file='imp2cost.txt',
        max_length=100,
        step_size=50,
        trials=3
    )
    
    analyze_complexity(results, create_plot=True)


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--plot":
            # Try to load existing results and plot
            try:
                import pandas as pd
                df = pd.read_csv('empirical_results.csv')
                results = [(row['Sequence_Length'], row['Average_Runtime_Seconds'], row['Standard_Deviation']) 
                          for _, row in df.iterrows()]
                print("Loaded existing empirical results from empirical_results.csv")
                create_empirical_plot(results)
            except Exception as e:
                print(f"Could not load existing empirical data: {e}")
                print("Run full analysis first: python3 runtime_analysis.py")
        
        elif sys.argv[1] == "--test":
            # Quick test mode for development
            run_quick_test()
        
        elif sys.argv[1] == "--help":
            print("CS 325 Runtime Analysis")
            print("Usage:")
            print("  python3 runtime_analysis.py          # Run full empirical analysis")
            print("  python3 runtime_analysis.py --plot   # Generate plots from saved data")
            print("  python3 runtime_analysis.py --test   # Quick test with small data")
            print("  python3 runtime_analysis.py --help   # Show this help")
        
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Default: run full empirical analysis
        main()