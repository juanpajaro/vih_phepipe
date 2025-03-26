#!/usr/bin/env bash
mkdir -p shap-plots

python3 run_shap_analysis.py $1 $2 $3
