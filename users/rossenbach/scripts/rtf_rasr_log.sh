#!/bin/bash

AM_RTF=$(zcat $1/output/search.log.*.gz | grep -o "AM_RTF: [0-9.]*" | grep -o "[0-9.]*" | awk 'BEGIN{sum=0}{sum+=$1}END{print sum/NR}')
echo "AM_RTF (Onnx-only): ${AM_RTF}"

TOTAL_RTF=$(zcat $1/output/search.log.*.gz | grep -A1 "flf-recognizer-rtf" | grep -o "[0-9.]*" | awk 'BEGIN{sum=0}{sum+=$1}END{print sum/NR}')
echo "TOTAL_RTF (flf-recognizer-rtf): ${TOTAL_RTF}"