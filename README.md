# Semantic Movie Search Engine

## Overview

This project implements a lightweight semantic search engine for movie plot summaries.

The system uses Sentence Transformers to convert movie plots and user queries into numerical embeddings. It then calculates cosine similarity between the query embedding and movie embeddings and returns the top three most relevant movies.

## Technologies Used

- Python
- Sentence Transformers
- NumPy
- Cosine Similarity

## Model

The project uses:

all-MiniLM-L6-v2

This pretrained Sentence Transformer model converts text into dense vector embeddings.

## How It Works

1. Store 20 movie plot summaries.
2. Load the Sentence Transformer model.
3. Convert the movie plots into embeddings.
4. Accept a natural-language user query.
5. Convert the query into an embedding.
6. Calculate cosine similarity between the query and movie embeddings.
7. Sort the results by similarity.
8. Return the top three movies.

## Example

Query:

"A movie about an astronaut surviving on another planet"

The system should return movies related to space travel and survival.

## Installation

```bash
uv sync
uv add -r requirements.txt

```for pip
pip install -r requirements.txt

###to run
get into the folder where the project is saved and open terminal from there
and 
run 
<uv run main.py>