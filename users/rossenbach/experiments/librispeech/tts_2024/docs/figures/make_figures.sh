#!/bin/bash
for texfile in *.tex ;
do
  name="${texfile%.*}"
  pdflatex -synctex=1 -interaction=nonstopmode $texfile
  pdf2svg "${name}.pdf" "${name}.svg"
done
rm *.aux
rm *.log
rm *.synctex.gz
rm *.pdf