#!/usr/bin/env python3
"""
CS 325 - Implementation Assignment 2: Edit Distance with Dynamic Programming
Author: Abraham Smith

This module implements the edit distance algorithm using dynamic programming
with an arbitrary cost matrix. It includes backtracking to generate the
optimal alignment between two sequences.
"""

import sys
import time
import os
from typing import List, Tuple, Dict


class EditDistance:
    """
    Edit Distance calculator with custom cost matrix support.
    
    This class implements the Wagner-Fischer algorithm for computing edit distance
    with arbitrary substitution/insertion/deletion costs, and provides backtracking
    to reconstruct the optimal alignment.
    """
    
    def __init__(self, cost_matrix: List[List[int]], x_chars: Dict[str, int], y_chars: Dict[str, int]):
        """
        Initialize the EditDistance calculator.
        
        Args:
            cost_matrix: 2D list where cost_matrix[i][j] is the cost of transforming 
                        character i to character j
            x_chars: Dictionary mapping characters to row indices in cost_matrix
            y_chars: Dictionary mapping characters to column indices in cost_matrix
        """
        self.cost_matrix = cost_matrix
        self.x_chars = x_chars
        self.y_chars = y_chars
    
    def get_cost(self, char_x: str, char_y: str) -> int:
        """
        Get the cost of transforming char_x to char_y.
        
        Args:
            char_x: Source character (or '-' for deletion)
            char_y: Target character (or '-' for insertion)
            
        Returns:
            Cost of the transformation
        """
        return self.cost_matrix[self.x_chars[char_x]][self.y_chars[char_y]]
    
    def compute_edit_distance(self, seq_x: str, seq_y: str) -> Tuple[int, List[List[int]]]:
        """
        Compute the minimum edit distance between two sequences.
        
        Args:
            seq_x: First sequence
            seq_y: Second sequence
            
        Returns:
            Tuple of (minimum_cost, DP_table)
        """
        m, n = len(seq_x), len(seq_y)
        
        # Initialize DP table with dimensions (m+1) x (n+1)
        # dp[i][j] represents minimum cost to align seq_x[0:i] with seq_y[0:j]
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Base cases: empty string transformations
        # Cost of aligning empty string with seq_y[0:j] (insertions)
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j-1] + self.get_cost('-', seq_y[j-1])
        
        # Cost of aligning seq_x[0:i] with empty string (deletions)
        for i in range(1, m + 1):
            dp[i][0] = dp[i-1][0] + self.get_cost(seq_x[i-1], '-')
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Three possible operations:
                # 1. Substitution (or match): transform seq_x[i-1] to seq_y[j-1]
                substitute_cost = dp[i-1][j-1] + self.get_cost(seq_x[i-1], seq_y[j-1])
                
                # 2. Deletion: delete seq_x[i-1]
                delete_cost = dp[i-1][j] + self.get_cost(seq_x[i-1], '-')
                
                # 3. Insertion: insert seq_y[j-1]
                insert_cost = dp[i][j-1] + self.get_cost('-', seq_y[j-1])
                
                # Choose the minimum cost operation
                dp[i][j] = min(substitute_cost, delete_cost, insert_cost)
        
        return dp[m][n], dp
    
    def backtrack_alignment(self, seq_x: str, seq_y: str, dp: List[List[int]]) -> Tuple[str, str]:
        """
        Backtrack through the DP table to construct the optimal alignment.
        
        Args:
            seq_x: First sequence
            seq_y: Second sequence
            dp: Completed DP table from compute_edit_distance
            
        Returns:
            Tuple of (aligned_seq_x, aligned_seq_y) with gaps marked as '-'
        """
        m, n = len(seq_x), len(seq_y)
        aligned_x, aligned_y = [], []
        
        i, j = m, n
        
        # Backtrack from dp[m][n] to dp[0][0]
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                # Check which operation was used to get to dp[i][j]
                substitute_cost = dp[i-1][j-1] + self.get_cost(seq_x[i-1], seq_y[j-1])
                delete_cost = dp[i-1][j] + self.get_cost(seq_x[i-1], '-')
                insert_cost = dp[i][j-1] + self.get_cost('-', seq_y[j-1])
                
                if dp[i][j] == substitute_cost:
                    # Substitution/match was used
                    aligned_x.append(seq_x[i-1])
                    aligned_y.append(seq_y[j-1])
                    i -= 1
                    j -= 1
                elif dp[i][j] == delete_cost:
                    # Deletion was used
                    aligned_x.append(seq_x[i-1])
                    aligned_y.append('-')
                    i -= 1
                else:
                    # Insertion was used
                    aligned_x.append('-')
                    aligned_y.append(seq_y[j-1])
                    j -= 1
            elif i > 0:
                # Only deletions left
                aligned_x.append(seq_x[i-1])
                aligned_y.append('-')
                i -= 1
            else:
                # Only insertions left
                aligned_x.append('-')
                aligned_y.append(seq_y[j-1])
                j -= 1
        
        # Reverse the alignments (we built them backwards)
        return ''.join(reversed(aligned_x)), ''.join(reversed(aligned_y))
    
    def align_sequences(self, seq_x: str, seq_y: str) -> Tuple[str, str, int]:
        """
        Compute optimal alignment and minimum edit distance.
        
        Args:
            seq_x: First sequence
            seq_y: Second sequence
            
        Returns:
            Tuple of (aligned_seq_x, aligned_seq_y, minimum_cost)
        """
        min_cost, dp_table = self.compute_edit_distance(seq_x, seq_y)
        aligned_x, aligned_y = self.backtrack_alignment(seq_x, seq_y, dp_table)
        return aligned_x, aligned_y, min_cost


def read_cost_matrix(filename: str) -> Tuple[List[List[int]], Dict[str, int], Dict[str, int]]:
    """
    Read cost matrix from CSV file.
    
    Format:
        First row: column headers (y-characters)
        Subsequent rows: row header (x-character) followed by costs
    
    Args:
        filename: Path to cost matrix file
        
    Returns:
        Tuple of (cost_matrix, x_char_to_index, y_char_to_index)
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Parse header row (y-characters)
    header = lines[0].split(',')
    y_chars = {char.strip(): i for i, char in enumerate(header[1:])}
    
    # Parse data rows
    cost_matrix = []
    x_chars = {}
    
    for i, line in enumerate(lines[1:]):
        parts = line.split(',')
        x_char = parts[0].strip()
        x_chars[x_char] = i
        
        row_costs = [int(cost.strip()) for cost in parts[1:]]
        cost_matrix.append(row_costs)
    
    return cost_matrix, x_chars, y_chars


def read_input_sequences(filename: str) -> List[Tuple[str, str]]:
    """
    Read sequence pairs from input file.
    
    Args:
        filename: Path to input file with sequence pairs
        
    Returns:
        List of (sequence1, sequence2) tuples
    """
    sequences = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                seq1, seq2 = line.split(',')
                sequences.append((seq1.strip(), seq2.strip()))
    return sequences


def main():
    """Main function to process input and generate alignments."""
    
    # Default file paths
    cost_file = 'imp2cost.txt'
    input_file = 'imp2input.txt' 
    output_file = 'imp2output.txt'
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        cost_file = sys.argv[1]
    if len(sys.argv) > 2:
        input_file = sys.argv[2]
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    try:
        # Read cost matrix
        print(f"Reading cost matrix from {cost_file}...")
        cost_matrix, x_chars, y_chars = read_cost_matrix(cost_file)
        
        # Initialize edit distance calculator
        calculator = EditDistance(cost_matrix, x_chars, y_chars)
        
        # Read input sequences
        print(f"Reading input sequences from {input_file}...")
        sequence_pairs = read_input_sequences(input_file)
        
        # Process each sequence pair
        print(f"Processing {len(sequence_pairs)} sequence pairs...")
        results = []
        
        for i, (seq1, seq2) in enumerate(sequence_pairs):
            print(f"Processing pair {i+1}/{len(sequence_pairs)}...")
            
            # Compute alignment
            aligned1, aligned2, cost = calculator.align_sequences(seq1, seq2)
            results.append((aligned1, aligned2, cost))
        
        # Write results to output file
        print(f"Writing results to {output_file}...")
        with open(output_file, 'w') as f:
            for aligned1, aligned2, cost in results:
                f.write(f"{aligned1},{aligned2}:{cost}\n")
        
        print(f"Successfully processed all sequences. Results written to {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()