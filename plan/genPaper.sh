#!/bin/bash
pdflatex ./proposal.tex 
bibtex proposal
pdflatex proposal.tex 
pdflatex proposal.tex 
