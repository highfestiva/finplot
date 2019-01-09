#!/bin/bash

python3 setup.py sdist bdist_wheel

echo "Deploying in 10..."
sleep 10
python3 -m twine upload dist/*
