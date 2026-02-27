# Vector Semantic Algebra: GloVe Analogy Solver

## Overview
This repository implements a high-performance **Semantic Analogy Solver** leveraging pre-trained **GloVe (Global Vectors for Word Representation)** embeddings. The project explores the geometric properties of word embeddings, specifically how semantic relationships are encoded as linear offsets in a high-dimensional continuous space.

## Mathematical Foundation
The core objective is to solve relational analogies in the form:  
**"A is to B as C is to D"** ($A:B :: C:D$)

According to the linear offset hypothesis, the relationship between two words can be represented by the vector difference of their embeddings. To find the unknown word $D$, we perform the following vector arithmetic:
$$v_{target} = v_B - v_A + v_C$$

The solver then searches for a word $w$ in the vocabulary $V$ that maximizes the cosine similarity with the target vector, while ensuring the output is not one of the input words:
$$d = \text{argmax}_{w \in V \setminus \{a, b, c\}} \left( \frac{v_w \cdot v_{target}}{\|v_w\| \|v_{target}\|} \right)$$

## Engineering Features
### 1. Vectorized Search (Performance)
Instead of iterating through the 400,000 words in the GloVe vocabulary—which would be computationally expensive ($O(N)$)—this implementation utilizes **matrix-vector multiplication**. 
- By pre-normalizing the entire embedding matrix to unit length ($L2$ norm = 1), the cosine similarity calculation is reduced to a simple **Dot Product**.
- This allows the solver to scan the entire vocabulary in milliseconds using NumPy’s optimized BLAS routines.

### 2. Identity Constraints
To ensure meaningful results, the algorithm implements a constraint-based search that filters out the input words ($A, B, C$) from the result set, preventing the identity mapping problem common in simple nearest-neighbor searches.

## Example Use Cases
The solver is capable of discovering various semantic and cultural relationships:
*   **Occupational Bias:** `man : doctor :: woman : nurse`
*   **Cultural Cuisine:** `japan : sushi :: germany : bratwurst`
*   **Famous Personalities:** `scientist : einstein :: painter : picasso`
*   **Grammatical Tense:** `tall : tallest :: short : shortest`

## Requirements
- `Python 3.x`
- `NumPy`
- `Gensim`
- `Scikit-learn` (for L2 normalization)

## Installation & Usage
1. Clone the repository.
2. Install dependencies: `pip install numpy gensim scikit-learn`.
3. Run the solver: The system will automatically download the `glove-wiki-gigaword-100` model on the first execution.
