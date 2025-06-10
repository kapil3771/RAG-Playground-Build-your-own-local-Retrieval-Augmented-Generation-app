#!/bin/bash

# Run RAG app inside virtual environment

echo "Activating virtual environment..."
source genai-env/bin/activate

echo "Running RAG app..."
python rag.py