# CS 325 - Edit Distance Algorithm Pseudocode
## Enhanced Wagner-Fischer Algorithm with Backtracking

### Author: Abraham Smith
### Date: 2/12/2026

---

## Algorithm Overview

The edit distance algorithm with backtracking consists of two main phases:
1. **Dynamic Programming Phase**: Compute the minimum edit distance using a 2D table
2. **Backtracking Phase**: Reconstruct the optimal alignment by tracing back through the DP table

---

## Phase 1: Dynamic Programming - Edit Distance Computation

```
ALGORITHM: ComputeEditDistance
INPUT: 
    - seq_x: string of length m (source sequence)  
    - seq_y: string of length n (target sequence)
    - cost_matrix: 2D array of transformation costs
    - x_chars: mapping from characters to row indices
    - y_chars: mapping from characters to column indices

OUTPUT:
    - dp: 2D table where dp[i][j] = minimum cost to align seq_x[0..i-1] with seq_y[0..j-1]
    - minimum_cost: dp[m][n] (final edit distance)

BEGIN
    // Initialize DP table of size (m+1) * (n+1)
    CREATE dp[0..m][0..n]
    
    // Base case: empty string to seq_y[0..j-1] (insertions only)
    dp[0][0] ← 0
    FOR j ← 1 TO n DO
        dp[0][j] ← dp[0][j-1] + cost(gap_char, seq_y[j-1])
    END FOR
    
    // Base case: seq_x[0..i-1] to empty string (deletions only) 
    FOR i ← 1 TO m DO
        dp[i][0] ← dp[i-1][0] + cost(seq_x[i-1], gap_char)
    END FOR
    
    // Fill DP table using recurrence relation
    FOR i ← 1 TO m DO
        FOR j ← 1 TO n DO
            // Option 1: Substitute/match seq_x[i-1] with seq_y[j-1]
            substitute_cost ← dp[i-1][j-1] + cost(seq_x[i-1], seq_y[j-1])
            
            // Option 2: Delete seq_x[i-1] (align with gap)
            delete_cost ← dp[i-1][j] + cost(seq_x[i-1], gap_char)
            
            // Option 3: Insert seq_y[j-1] (align gap with seq_y[j-1])
            insert_cost ← dp[i][j-1] + cost(gap_char, seq_y[j-1])
            
            // Take minimum cost operation
            dp[i][j] ← MIN(substitute_cost, delete_cost, insert_cost)
        END FOR
    END FOR
    
    RETURN dp, dp[m][n]
END
```

---

## Phase 2: Backtracking - Optimal Alignment Reconstruction

```
ALGORITHM: BacktrackAlignment
INPUT:
    - seq_x: string of length m (source sequence)
    - seq_y: string of length n (target sequence)  
    - dp: completed DP table from ComputeEditDistance
    - cost_matrix: 2D array of transformation costs
    - x_chars: mapping from characters to row indices
    - y_chars: mapping from characters to column indices

OUTPUT:
    - aligned_x: source sequence with gaps inserted for optimal alignment
    - aligned_y: target sequence with gaps inserted for optimal alignment

BEGIN
    // Initialize alignment arrays and position pointers
    CREATE aligned_x[] (empty array)
    CREATE aligned_y[] (empty array)
    i ← m
    j ← n
    
    // Trace back from dp[m][n] to dp[0][0]
    WHILE (i > 0 OR j > 0) DO
        IF (i > 0 AND j > 0) THEN
            // Check which operation achieves dp[i][j]
            substitute_cost ← dp[i-1][j-1] + cost(seq_x[i-1], seq_y[j-1])
            delete_cost ← dp[i-1][j] + cost(seq_x[i-1], gap_char)  
            insert_cost ← dp[i][j-1] + cost(gap_char, seq_y[j-1])
            
            IF (dp[i][j] == substitute_cost) THEN
                // Substitution/match was optimal
                PREPEND seq_x[i-1] TO aligned_x
                PREPEND seq_y[j-1] TO aligned_y
                i ← i - 1
                j ← j - 1
            ELSE IF (dp[i][j] == delete_cost) THEN  
                // Deletion was optimal
                PREPEND seq_x[i-1] TO aligned_x
                PREPEND gap_char TO aligned_y
                i ← i - 1
            ELSE
                // Insertion was optimal (dp[i][j] == insert_cost)
                PREPEND gap_char TO aligned_x
                PREPEND seq_y[j-1] TO aligned_y
                j ← j - 1
            END IF
            
        ELSE IF (i > 0) THEN
            // Only deletions remain (j = 0)
            PREPEND seq_x[i-1] TO aligned_x
            PREPEND gap_char TO aligned_y
            i ← i - 1
            
        ELSE
            // Only insertions remain (i = 0)
            PREPEND gap_char TO aligned_x  
            PREPEND seq_y[j-1] TO aligned_y
            j ← j - 1
        END IF
    END WHILE
    
    RETURN aligned_x, aligned_y
END
```

---

## Complete Algorithm: Edit Distance with Alignment

```
ALGORITHM: EditDistanceWithAlignment
INPUT:
    - seq_x: source sequence (string)
    - seq_y: target sequence (string)  
    - cost_matrix: transformation cost matrix
    - character mappings

OUTPUT:
    - aligned_x: optimally aligned source sequence
    - aligned_y: optimally aligned target sequence
    - minimum_cost: minimum edit distance

BEGIN
    // Phase 1: Compute minimum edit distance
    dp, minimum_cost ← ComputeEditDistance(seq_x, seq_y, cost_matrix, x_chars, y_chars)
    
    // Phase 2: Reconstruct optimal alignment
    aligned_x, aligned_y ← BacktrackAlignment(seq_x, seq_y, dp, cost_matrix, x_chars, y_chars)
    
    RETURN aligned_x, aligned_y, minimum_cost
END
```

---

## Complexity Analysis

### Time Complexity:
- **DP Table Construction**: O(mn) operations, each taking O(1) time
- **Backtracking**: O(m + n) operations in the worst case
- **Total**: O(mn) where m = |seq_x| and n = |seq_y|

### Space Complexity:  
- **DP Table**: O(mn) space for the 2D table
- **Alignment Arrays**: O(m + n) space for the reconstructed alignment
- **Total**: O(mn)

### Optimizations:
- **Space-Optimized DP**: If only the edit distance is needed (no alignment), 
  space can be reduced to O(min(m,n)) using two arrays
- **Alignment Space**: O(m + n) for storing the optimal alignment

---

## Key Algorithmic Insights

1. **Optimal Substructure**: The optimal alignment of sequences seq_x[0..i] and seq_y[0..j] 
   contains optimal alignments of smaller subsequences.

2. **Overlapping Subproblems**: Many alignment problems share common subproblems, 
   making dynamic programming efficient.

3. **Backtracking Strategy**: Multiple optimal solutions may exist. The backtracking 
   algorithm finds one optimal alignment by consistently choosing the first valid operation 
   that achieves the minimum cost.

4. **Gap Handling**: Gaps (represented as '-') allow for insertions and deletions, 
   enabling flexible sequence alignment with arbitrary cost penalties.

---

## Implementation Notes

- **Tie-Breaking**: When multiple operations achieve the minimum cost, the algorithm 
  prioritizes substitution > deletion > insertion for consistent results.

- **Cost Matrix**: The algorithm supports arbitrary cost matrices, enabling different 
  penalties for various character transformations.

- **Gap Character**: The special character '-' represents gaps in alignment and must 
  be included in the cost matrix for insertion/deletion operations.