#!/usr/bin/env python3
"""
CS 325 Assignment 2 - Empirical Runtime Analysis
Run this script to perform the exact empirical analysis required for the assignment.

This script will:
1. Test sequence lengths: 500, 1000, 2000, 4000, 5000
2. Generate 10 random pairs for each length (alphabet: A, G, T, C)
3. Compute average runtime and standard deviation
4. Generate polynomial order visualization plots
5. Save results in CSV format for the report

Note: This may take 10-30 minutes depending on your system performance.
"""

from runtime_analysis import run_empirical_analysis, create_empirical_plot, save_empirical_results
import time

def main():
    print("=" * 70)
    print("CS 325 Assignment 2 - Empirical Runtime Analysis")
    print("=" * 70)
    print()
    print("This analysis will test the following requirements:")
    print("✓ Sequence lengths: 500, 1000, 2000, 4000, 5000")
    print("✓ 10 random pairs per length")
    print("✓ Alphabet: A, G, T, C (DNA sequences)")
    print("✓ Polynomial order verification (O(n^2) expected)")
    print("✓ Statistical analysis with standard deviation")
    print()
    
    # Estimate runtime
    print("Estimated runtime: 10-30 minutes (depending on system performance)")
    response = input("Continue? (y/n): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Analysis cancelled.")
        return
    
    print()
    start_time = time.time()
    
    # Run the empirical analysis
    try:
        results = run_empirical_analysis('imp2cost.txt')
        
        # Save results
        save_empirical_results(results)
        
        # Create visualization
        create_empirical_plot(results)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print()
        print("Generated files:")
        print("  empirical_results.csv    - Detailed runtime data")
        print("  empirical_runtime.png    - Polynomial order plots")
        print()
        print("Results Summary:")
        for length, avg_time, std_dev in results:
            print(f"  Length {length:4d}: {avg_time:.6f}s +/- {std_dev:.6f}s")
        print()
        print("Use these files for your assignment report!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Make sure all required files are present and try again.")

if __name__ == "__main__":
    main()