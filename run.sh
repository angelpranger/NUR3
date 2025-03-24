#!/bin/bash

echo "Downloading files"
if [ ! -e satgals_m11.txt ]; then
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m11.txt
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m12.txt
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m13.txt
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m14.txt
  wget home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m15.txt
fi

echo "Running the script for question 1"
python3 Q1.py > output_Q1.txt

echo "Generating the pdf"

pdflatex pranger.tex
bibtex pranger.aux
pdflatex pranger.tex
pdflatex pranger.tex