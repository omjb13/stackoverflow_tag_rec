from sys import argv
import csv
from collections import defaultdict

headers = ["Id", "Title", "Body", "Tags"]

training_file = argv[1]
top_tags_file = argv[2]
outfile = argv[3]

top_100_tags = open(top_tags_file, 'r').readlines()
top_100_tags = map(lambda x: x.split(',')[0], top_100_tags)
top_100_tags = set(top_100_tags)

results = []

training_data = csv.reader(file(training_file), quotechar='"')
outwriter = csv.writer(file(outfile, 'w'), quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')

outwriter.writerow(headers)

for line in training_data:
    question_id, title, body, tags = line
    tag_set = set((tags.split(' ')))
    if len(tag_set.intersection(top_100_tags)) == len(tag_set):
        outwriter.writerow(line)
