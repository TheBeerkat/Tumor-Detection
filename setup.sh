#!/bin/bash

# Check for platform and set appropriate commands
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    # download the dataset if it doesn't exist
    if [ ! -d data/INbreast_2012 ]; then
        curl -L -o data/INbreast_2012.zip https://www.kaggle.com/api/v1/datasets/download/tommyngx/inbreast2012
        # unzip the dataset
        unzip data/INbreast_2012.zip -d data/INbreast_2012
        # remove the zip file
        rm -rf data/INbreast_2012.zip
    fi
    # create the processed directory
    mkdir -p data/processed
    # create the virtual environment
    python3 -m venv venv
    source venv/bin/activate
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    # download the dataset if it doesn't exist
    if [ ! -d data/INbreast_2012 ]; then
        curl -L -o data/INbreast_2012.zip https://www.kaggle.com/api/v1/datasets/download/tommyngx/inbreast2012
        # unzip the dataset
        powershell -Command "Expand-Archive -Path 'data/INbreast_2012.zip' -DestinationPath 'data/INbreast_2012'"
        # remove the zip file
        Remove-Item data/INbreast_2012.zip
    fi
    # create the processed directory
    mkdir -p data/processed
    # create the virtual environment
    python -m venv venv
    .\venv\Scripts\activate
else
    echo "Unsupported OS"
    exit 1
fi

# install the required packages
pip install -r requirements.txt