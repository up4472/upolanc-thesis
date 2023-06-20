#!/bin/bash

# shellcheck disable=SC2231
# shellcheck disable=SC2086

# Define directory
NOTEBOOK=/d/hpc/projects/FRI/up4472/upolanc-thesis/notebook

# Print status
echo "Converting .ipynb files to .py files in [$NOTEBOOK]"

# Convert every .ipynb in directory to .py
for filepath in $NOTEBOOK/*.ipynb; do
	jupyter nbconvert --no-prompt --to script $filepath > /dev/null 2>&1
done
