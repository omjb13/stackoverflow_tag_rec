from sys import argv
from collections import defaultdict
import operator

topN = 500
tag_counter = defaultdict(int)

for line in file(argv[1]):
    index, tags = line.split(',')
    for tag in tags.split(' '):
        tag_counter[tag.strip()] += 1

sorted_tag_counter = sorted(tag_counter.items(), key=operator.itemgetter(1), reverse=True)

for (tag, count) in sorted_tag_counter[0:topN]:
    print ",".join((tag, str(count)))
