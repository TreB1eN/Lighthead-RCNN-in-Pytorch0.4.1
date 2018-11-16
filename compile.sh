#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo "Building nms op..."
cd functions/nms/
make clean
make PYTHON=${PYTHON}

echo "Building psroialign op..."
cd ../psroialign/cuda/
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py install