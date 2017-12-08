#!/bin/bash
python clean_data.py ../data/train_uncleaned.csv
mv -f cleaned.csv ../data/train.csv

python clean_data.py ../data/test_uncleaned.csv
mv -f cleaned.csv ../data/test.csv
