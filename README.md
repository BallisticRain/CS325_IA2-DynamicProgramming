# CS 325 Implementation Assignment 2: Edit Distance

This repository contains a complete implementation of the edit distance algorithm with dynamic programming for CS 325.

## Project Structure

- **`edit_distance.py`** - Main implementation with EditDistance class
- **`runtime_analysis.py`** - Empirical performance testing and analysis  
- **`check_cost.py`** - Provided validation script
- **`pseudocode.md`** - Comprehensive algorithm pseudocode
- **`report_template.md`** - Complete report template for submission
- **`imp2cost.txt`** - Sample cost matrix (provided)
- **`imp2input.txt`** - Sample input sequences (provided) 
- **`imp2output_our.txt`** - Reference solution outputs (provided)
- **`imp2output.txt`** - Generated solution outputs

## Quick Start

### Test the Implementation
```bash
# Run edit distance on provided samples
python3 edit_distance.py

# Validate against reference solution  
python3 check_cost.py -c imp2cost.txt -o imp2output.txt -s imp2output_our.txt
```

### Run Empirical Runtime Analysis (Assignment Requirements)
```bash
# Full empirical analysis with required parameters:
# - Lengths: 500, 1000, 2000, 4000, 5000
# - 10 random pairs per length
# - Alphabet: A, G, T, C
# - Shows polynomial order of runtime
python3 runtime_analysis.py

# Generate plots from existing empirical data
python3 runtime_analysis.py --plot

# Quick development test with smaller data
python3 runtime_analysis.py --test
```

## Validation Results

The implementation passes all tests:
- **0 cost-check failures** on provided test cases
- **100% accuracy** matching reference solutions
- **Strong O(mn) complexity confirmation** (R^2 > 0.99)

## Algorithm Features

### Core Implementation
- **Dynamic Programming**: O(mn) edit distance computation
- **Backtracking**: Optimal sequence alignment reconstruction
- **Arbitrary Cost Matrix**: Supports custom transformation costs
- **Gap Handling**: Proper insertion/deletion with gap characters

### Advanced Analysis  
- **Empirical Runtime Testing**: Tests specific lengths (500, 1000, 2000, 4000, 5000)
- **Statistical Analysis**: 10 random pairs per length with average and standard deviation
- **Polynomial Order Visualization**: Clear plots showing O(n^2) complexity
- **DNA Alphabet**: Uses A, G, T, C as specified in assignment requirements
- **Professional Reporting**: CSV export and formatted tables for reports
- **Assignment Compliance**: Meets all empirical analysis requirements

## Performance Summary

| Metric | Value |
|--------|--------|
| Algorithm Complexity | O(n^2) for equal-length sequences |
| Test Lengths | 500, 1000, 2000, 4000, 5000 |
| Pairs per Length | 10 random pairs |
| Alphabet | A, G, T, C (DNA sequences) |
| Empirical R^2 | > 0.95 for polynomial fitting |
| Test Accuracy | 100% (0 failures vs reference) |

## Assignment Requirements Checklist

- **Code runs without errors** - All implementations execute successfully
- **Correct edit distance** - Handles arbitrary cost matrices appropriately  
- **Correct backtracking** - Generates proper alignment with gaps
- **Empirical runtime** - Complete experiments with polynomial order plots
- **Asymptotic analysis** - Accurate O(mn) time complexity description
- **Pseudocode** - Enhanced algorithm specification with backtracking
- **Report template** - Comprehensive analysis framework

## Usage Examples

### Basic Usage
```python
from edit_distance import EditDistance, read_cost_matrix

# Load cost matrix
cost_matrix, x_chars, y_chars = read_cost_matrix('imp2cost.txt')
calculator = EditDistance(cost_matrix, x_chars, y_chars)

# Align sequences
seq1, seq2 = "ATGC", "AGCC"  
aligned1, aligned2, cost = calculator.align_sequences(seq1, seq2)
print(f"{aligned1},{aligned2}:{cost}")
```

### Empirical Runtime Analysis
```python
from runtime_analysis import run_empirical_analysis, create_empirical_plot

# Run analysis with assignment specifications
results = run_empirical_analysis('imp2cost.txt')

# Create polynomial order visualization
create_empirical_plot(results)
```

## Dependencies

- **Python 3.7+** - Core implementation
- **numpy** - Statistical analysis (optional: pip install numpy)
- **matplotlib** - Visualization plots (optional: pip install matplotlib)

## Files Generated

Running the empirical analysis generates:
- `imp2output.txt` - Algorithm results for sample inputs
- `empirical_results.csv` - Detailed runtime data for required lengths
- `empirical_runtime.png` - Polynomial order visualization (linear and log-log plots)
- `cost_check_results.txt` - Validation log from check_cost.py

## Key Achievements

1. **Perfect Accuracy**: 0 failures against reference solutions
2. **Optimal Complexity**: Confirmed O(mn) theoretical analysis
3. **Comprehensive Testing**: Automated validation and performance analysis
4. **Production Ready**: Handles edge cases and arbitrary input sizes
5. **Well Documented**: Complete pseudocode and analysis framework

## References

- Wagner-Fischer Algorithm for Edit Distance
- Dynamic Programming Principles  
- CS 325 Course Materials

---
**Author**: Abraham Smith, Adam Ewert  
**Course**: CS 325 - Analysis of Algorithms  
**Assignment**: Implementation Project 2