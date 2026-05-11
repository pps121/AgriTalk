#!/usr/bin/env bash
# compile_proposal.sh — Compile the AgriTalk PhD proposal to Saha_Partha.pdf
# Usage: bash compile_proposal.sh
# Requires: pdflatex, bibtex (TeX Live / MacTeX / MiKTeX)
#
# If LaTeX is NOT installed locally, upload proposal/ folder to Overleaf:
#   1. Go to https://www.overleaf.com  → New Project → Upload Project
#   2. Zip the proposal/ folder: zip proposal.zip proposal/
#   3. Upload the zip; set main file to proposal_main.tex
#   4. Click Compile → Download PDF → rename to Saha_Partha.pdf

set -e

PROPOSAL_DIR="$(dirname "$0")/proposal"
OUTPUT_NAME="Saha_Partha.pdf"

echo "=== Compiling AgriTalk PhD Proposal ==="
echo "Working directory: $PROPOSAL_DIR"

cd "$PROPOSAL_DIR"

echo "Step 1/4: pdflatex (first pass)..."
pdflatex -interaction=nonstopmode proposal_main.tex

echo "Step 2/4: bibtex..."
bibtex proposal_main

echo "Step 3/4: pdflatex (second pass — resolve citations)..."
pdflatex -interaction=nonstopmode proposal_main.tex

echo "Step 4/4: pdflatex (third pass — finalise cross-refs)..."
pdflatex -interaction=nonstopmode proposal_main.tex

echo ""
echo "Copying output to submission file..."
cp proposal_main.pdf "../$OUTPUT_NAME"
echo "SUCCESS: $(dirname "$PROPOSAL_DIR")/$OUTPUT_NAME"
echo ""
echo "File size: $(ls -lh "../$OUTPUT_NAME" | awk '{print $5}')"
echo "Deadline: 11 May 2026, 12:00 CET — submit as Saha_Partha.pdf"
