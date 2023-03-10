#!/bin/bash

# Setup paths
ROOT="/d/hpc/home/up4472/workspace/upolanc-thesis/"

if [[ ":$PYTHONPATH:" != *":$ROOT:"* ]]; then
	export PYTHONPATH="$PYTHONPATH:$ROOT"
fi

if [[ ":$PATH:" != *":$ROOT:"* ]]; then
	export PATH="$PATH:$ROOT"
fi

# Run script
python /d/hpc/home/up4472/workspace/upolanc-thesis/source/python/sysinfo.py