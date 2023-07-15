#!/bin/bash

# Define roots
ROOT=/d/hpc/projects/FRI/up4472
MEME=$ROOT/meme/db/motif_databases

# Define query paths
QUERY_PATH=$ROOT
QUERY_PATH=$QUERY_PATH/motifs.meme

# Define 1st database
DAP_MEME=$MEME/ARABD/ArabidopsisDAPv1.meme
DAP_OUTS=$ROOT/meme/out/tomtom/dap

mkdir -p $DAP_OUTS

# Define 2nd database
PBM_MEME=$MEME/ARABD/ArabidopsisPBM_20140210.meme
PBM_OUTS=$ROOT/meme/out/tomtom/pbm

mkdir -p $PBM_OUTS

# Compare to 1st database
echo ""
echo "Comparing $QUERY_PATH to $DAP_MEME ..."
echo ""

tomtom -png -verbosity 1 -oc $DAP_OUTS $QUERY_PATH $DAP_MEME

# Compare to 2nd database
echo ""
echo "Comparing $QUERY_PATH to $PBM_MEME ..."
echo ""

tomtom -png -verbosity 1 -oc $PBM_OUTS $QUERY_PATH $PBM_MEME
