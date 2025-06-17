#!/bin/bash

echo "Installing private dependencies..."
pip install -e ./private/pytorch-minimize
pip install -e ./private/gpid
pip install -e ./private/dit

echo "Installing main package..."
pip install -e .

echo "Done!"
