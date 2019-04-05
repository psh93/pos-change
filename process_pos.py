from nltk.stem import WordNetLemmatizer
import os, sys
import argparse
import re
import json

year_pos = {};

date_str = '(\d\d\d\d),(\d+)'
token_str = '(.+)\/(.+)\/.+\/\d+'

lm = WordNetLemmatizer()

sYear = 1800
fYear = 2000

def grab_pos( line ):

  # Exclude 'head' and 'total counts'
  syntactic_ngram = []
  year_counts = []
  
  items = line.split()
  for i, item in enumerate(items[1:]):
  
    if item.isdigit():
      break
  
    syntactic_ngram.append(item)
  
  year_counts = items[i+2:]

  # Extract five tokens from 5-gram corpus
  tokens, tags = [], []
  for item in syntactic_ngram:
        
    output = re.findall(token_str, item)
    if output != [] and len(output[0]) == 2:

      token = output[0][0]
      token = lm.lemmatize(token)
      pos = output[0][1]
      
    else: continue
    
    tokens.append(token)
    tags.append(pos)
    
    # Initialize dictionary
    if token not in year_pos:
      year_pos[token] = dict()

  # For each token, record counts, c, by year and pos tag
  # year_pos[token][year][pos] = c
  for record in year_counts:

    output = re.findall(date_str, record)[0]

    if len(output) == 2:
        
      year = int(output[0])
      count = int(output[1])
    
    else: continue

    if year not in range(sYear, fYear + 1, 10): continue

    for token, tag in zip(tokens, tags):

      if year not in year_pos[token]:
        year_pos[token][year] = dict()
      
      year_pos[token][year][tag] = year_pos[token][year].get(tag, 0) + count
    
def main():
    
  indir = '/u/parksh/csc2611/postrack-master/data/pos_ngrams/'
  #indir = './test'

  for subdir, dirs, files in os.walk(indir):
      for i, file in enumerate(files):

          fullFile = os.path.join(subdir, file)
          corpus = open(fullFile, 'r').readlines()

          N = len(corpus)
          for j, line in enumerate(corpus):
    
            grab_pos(line)
            print("Extracting POS counts %d.. %d / %d" % (i, j, N))

  outdir = './year_pos.json'

  fout = open(outdir, 'w', encoding='utf-8')
  json.dump(year_pos, fout)

if __name__ == "__main__":

    main()
