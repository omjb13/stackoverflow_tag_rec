import pandas as pd
import re
import string
from IPython.display import display, HTML
from sys import argv
from nltk.corpus import stopwords

# stop words
stop = stopwords.words('english')
if 'for' in stop:
    stop.remove('for')
stop = map(lambda x: x.encode('utf-8'), stop)

# define regexes
punctuations = string.punctuation
punctuations_to_keep = '#'
punctuations = ''.join((set(punctuations) - set(punctuations_to_keep)))
replace_punctuations = re.compile('[%s]' % re.escape(punctuations))
regexURLs = re.compile(r'http\S+|www.\S+', flags=re.IGNORECASE)
regexWhiteSpaces = re.compile(r'\s\s+', flags=re.IGNORECASE)
regexNewline = re.compile(r'\n', flags=re.IGNORECASE)
regexCode = re.compile(r'<code>.*?<\/code>', flags=re.IGNORECASE)
regexOpeningTags = re.compile(r'<.*?>', flags=re.IGNORECASE)
regexClosingTags = re.compile(r'<\/.*?>', flags=re.IGNORECASE)
regexNumbers = re.compile(r'\d+')

df = pd.read_csv(argv[1])

# New line removal
df["Body"] = df["Body"].str.replace(regexNewline, " ")
df["Title"] = df["Title"].str.replace(regexNewline, " ")

# Whitespace removal
df["Body"] = df["Body"].str.replace(regexWhiteSpaces, " ")
df["Title"] = df["Title"].str.replace(regexWhiteSpaces, " ")

# Remove all URLs
df["Body"] = df["Body"].str.replace(regexURLs, "")
df["Title"] = df["Title"].str.replace(regexURLs, "")

# Remove all code blocks
df["Body"] = df["Body"].str.replace(regexURLs, "")
df["Title"] = df["Title"].str.replace(regexURLs, "")

# Remove all HTML opening and closing tags
df["Body"] = df["Body"].str.replace(regexOpeningTags, "")
df["Body"] = df["Body"].str.replace(regexClosingTags, "")
df["Title"] = df["Title"].str.replace(regexOpeningTags, "")
df["Title"] = df["Title"].str.replace(regexClosingTags, "")

# Remove all punctuation
df["Body"] = df["Body"].str.replace(replace_punctuations, " ")
df["Title"] = df["Title"].str.replace(replace_punctuations, " ")

# Convert to lowercase
df["Body"] = df["Body"].str.lower()
df["Title"] = df["Title"].str.lower()

# Remove stopwords
df['Title'] = df['Title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df['Body'] = df['Body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Remove numbers
df["Body"] = df["Body"].str.replace(regexNumbers, "")
df["Title"] = df["Title"].str.replace(regexNumbers, "")

df = df.set_index('Id')

df.to_csv("cleaned.csv", encoding='utf-8', index=False)
