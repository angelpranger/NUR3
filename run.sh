#!/bin/bash

echo "Running the script for question 1"
python3 Q1.py > output_Q1.txt

echo "Running the script for question 2"
python3 Q2.py > output_Q2.txt

echo "Generating the pdf"

pdflatex pranger.tex
bibtex pranger.aux
pdflatex pranger.tex
pdflatex pranger.tex