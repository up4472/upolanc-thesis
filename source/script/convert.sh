#!/bin/bash

cd /d/hpc/home/up4472/workspace/upolanc-thesis/lab

echo "Converting nbp01-analysis.ipynb to python script ..."
jupyter nbconvert --to script nbp01-analysis.ipynb    > /dev/null 2>&1

echo "Converting nbp02-anndata.ipynb to python script ..."
jupyter nbconvert --to script nbp02-anndata.ipynb     > /dev/null 2>&1

echo "Converting nbp03-tsne.ipynb to python script ..."
jupyter nbconvert --to script nbp03-tsne.ipynb        > /dev/null 2>&1

echo "Converting nbp04-feature.ipynb to python script ..."
jupyter nbconvert --to script nbp04-feature.ipynb     > /dev/null 2>&1

echo "Converting nbp05-target.ipynb to python script ..."
jupyter nbconvert --to script nbp05-target.ipynb      > /dev/null 2>&1

echo "Converting nbp06-tuner.ipynb to python script ..."
jupyter nbconvert --to script nbp06-tuner.ipynb       > /dev/null 2>&1

echo "Converting nbp07-zrimec2020c.ipynb to python script ..."
jupyter nbconvert --to script nbp07-zrimec2020c.ipynb > /dev/null 2>&1

echo "Converting nbp07-zrimec2020r.ipynb to python script ..."
jupyter nbconvert --to script nbp07-zrimec2020r.ipynb > /dev/null 2>&1

echo "Converting nbp08-washburn.ipynb to python script ..."
jupyter nbconvert --to script nbp08-washburn.ipynb    > /dev/null 2>&1

cd /d/hpc/home/up4472/workspace/upolanc-thesis
