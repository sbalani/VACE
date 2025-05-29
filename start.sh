#!/bin/bash

# Install jupyter if not already installed
pip install jupyter

# Start Jupyter in the background
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' &

# Keep container running
tail -f /dev/null 