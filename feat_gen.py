from __future__ import division
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from proba_distribution import *
import multiprocessing as mp
import numpy as np
import json
import math
import time
import re

ngrmas = dict()
year_pos = dict()
lm = WordNetLemmatizer()
topn = list()
tokens = list()
types = list()
N = 0
sYear = 1800
fYear = 2000

# dynamic variables
c_contexts = dict()
contexts = list()
docs = dict()

def load_data():

  global ngrams, year_pos

  print("Loading data..")

  # Load n-grams: X[year][' .. '] = counts
  f = '/u/parksh/csc2611/postrack-master/output/ngrams.json'
  handler = open(f, 'r')
  ngrams = json.load(handler)
  build_vocab()
  print('Loaded N-grams')

  # Load POS freq.: X[token][year][POS] = counts
  f = '/u/parksh/csc2611/postrack-master/output/year_pos.json'
  handler = open(f, 'r')
  year_pos = json.load(handler)
  
  print('Loaded POS occurrences')

def build_vocab():
 
  global types, tokens
  global N

  for corpus in ngrams.values():
    for line in corpus:
      tokens.extend(line.split())

  N = len(tokens)
  types = set(tokens)

  print("%d tokens and %d types" % (N, len(types)))

def get_topn():

  #f = '/u/parksh/csc2611/postrack-master/common_vocab.txt'
  stoplist = stopwords.words('english')
  global topn
  choice = 2

  if choice == 1:

    handler = open(f, 'r')
    for word in handler.readlines():
    
      word = lm.lemmatize(word).lower()
      if word not in stoplist and word in types:
        topn.append(word)

    print("Analyzing %d common words" % len(topn))
    return;

  else:

    f = '/u/parksh/csc2611/postrack-master/feat_list.txt'
    handler = open(f, 'r')
    for word in handler.readlines():

      word = word.strip()
      topn.append(word)

    print("Analyzing %d common words" % len(topn))
    return; 

  # Load most frequent N words
  brownc = brown.words()
  fdist = FreqDist(brownc)
  W = [ word[0] for word in fdist.most_common(n) ]  

  for word in W:

    word = lm.lemmatize(word).lower()
    if word not in stoplist and word in types:
        topn.append(word)

  handler = open(f, 'w')
  for word in topn:

    handler.write(word)
    handler.write('\n')

  handler.close()
  print("From top 5000 brown word types: %d words selected" % len(topn))

# Output: Distribution vectors
# X[year] = [ 0.x, 0.y, .. ]
def get_dist( word ):

  probs = proba_distribution(word)
  prob_t = dict()

  for year in range(sYear, fYear + 1, 10):

    year = str(year)
    if year not in probs:

      prob_t[year] = [0]
      continue

    tmps = probs[str(year)]
    dist = list()

    for POS in ['NN', 'NNP', 'ADV', 'ADJ', 'VB' ]:

      if POS in tmps:
        dist.append(tmps[POS])

      else:
        dist.append(0.0)

    prob_t[year] = dist

  return prob_t;

# Output: JS divergences per decade
# X[year] = 0.x [0,1]
def extract_rates( i, word ):

  JSDs = get_jsd(word,[sYear,fYear])
  print(JSDs)
  rates = dict()

  # Store from t = 1810 to 2000
  for j, year in enumerate(range(sYear+10,fYear+1,10)):

    rates[year] = JSDs[j]

  print("Processed %d word = %s" % (i,word))
  fpath = '/u/parksh/csc2611/postrack-master/output/jsd/' + word + '_j.json'
  handler = open(fpath, 'w')
  json.dump(rates,handler,default=str)

  handler.close()

# X[year] = [ D, I, F ]
def extract_feats( i, word ):

  print("Extracting %s" % word) 

  word = lm.lemmatize(word)
  contexts = dict()
  c_contexts = dict()
  docs = dict()
  feats = dict()

  # Extract features from t = 1 to t = n - 1
  # i.e., 2000 - 10 = 1990
  for year in range(sYear, fYear - 10 + 1, 10):

    f1 = compute_D(word, year)
    f2 = compute_I(word, year)
    f3 = compute_F(word, year)
    fvec = [f1, f2, f3] 
  
    feats[year] = fvec
    print("Extracted %d word = %s:" % (i,word),year,fvec)

  fpath = '/u/parksh/csc2611/postrack-master/output/feats/' + word + '_f.json'
  handler = open(fpath, 'w')
  json.dump(feats,handler,default=str)

  handler.close()

# Computes degree of contextual diversity (D_t)
# Output: D_t (a floating number)
def compute_D( word, year ):

  # Get neighbors and their frequencies
  get_contexts(word,year)
  if contexts == []: 
    return 0.0;

  count_context(word,contexts)

  # Compute joint probabilities (jprobs)
  total = 0
  jprobs = list()
  for neighbor in c_contexts:

    jprob = joint_proba(neighbor)

    total += jprob
    jprobs.append(jprob)

  # Compute contextual diversity
  D_t = 0
  for joint_P in jprobs:

    if total == 0:
      return 0;

    div = joint_P / total
    D_t += joint_P * math.log(div, 2)

  D_t = -(D_t)

  return D_t;

# Computes word informativeness (I_t)
# Output: I_t (a floating number)
def compute_I( word, year ):

  # Get neighbors and their frequencies
  if contexts == []:
    return 0.0;

  # Frequency of a word in a corpus
  n_word = count_word()

  # Compute informativeness
  I_t = 0

  # Compute join probabilities
  neighbors_P = proba_neighbors(year,n_word)
  for neighbor in neighbors_P:

    neigh_P = neighbors_P[neighbor]
    joint_P = joint_proba(neighbor)

    if neigh_P > 0.0:

      div = joint_P / neigh_P
      I_t += math.log(div,2)

  I_t = -(1/n_word) * I_t

  return I_t;

# Output: probability of neighbor word
# X[neighbor word] = 0.x
def proba_neighbors( year, n_word ):

  # P(neighbor)
  year = str(year)
  if year not in ngrams:
    return [];

  # docs initiated from previous step
  neighbor_P = dict()
  c_neighbors = dict()
  
  neighbors = list(c_contexts.keys())
  for i,line in enumerate(docs):

    tokens = line.split()
    counts = docs[line]

    # Use set for efficiency
    for word in set(tokens).intersection(neighbors):
      
      c_neighbors[word] = c_neighbors.get(word, 0) + counts
   
  for neighbor in c_neighbors:
    neighbor_P[neighbor] = c_neighbors[neighbor] / n_word  

  return neighbor_P;

# Computes probabilities of NN and VB tag (F_NN, F_VB)
# Output: [F_NN, F_VB]
def compute_F( word, year ):

  dist = proba_distribution(word)
  if str(year) not in dist:
    return 0.0;

  dist = dist[str(year)]

  # Probability of Noun
  P_NN = dist.get('NN', 0.0)

  # Frequency of Verb
  P_VB = dist.get('VB', 0.0)

  # Frequency of Proper noun
  P_NNP = dist.get('NNP', 0.0)

  # Frequency of Adjective
  P_ADJ = dist.get('ADJ', 0.0)

  # Frequency of Adverb
  P_ADV = dist.get('ADV', 0.0)

  P = [ P_NN, P_VB, P_NNP, P_ADJ, P_ADV ]
  ind = int(np.argmax(P)) + 1.0 

  return ind;

# Collect all documents(ngrams) that contain word, w
# Output: contexts
# i.e., X = [(line,count), .. ]
def get_contexts( word, year ):

  global contexts, docs

  contexts = list()
  year = str(year)
  if year not in ngrams:
    return [];

  docs = ngrams[year]
  for line in docs:

    if word in line.split():

      freq = docs[line]
      contexts.append((line, freq))

# Collect all contextual words within a window of 2
# X[word] = counts
def count_context( word, contexts ):

  # Consider two consecutive neighbors as c_i
  # i.e., in { A B C }, A and C are contexts
  dist = 2

  global c_contexts
  c_contexts = dict()
  for item in contexts:

    context = item[0].split()
    counts = item[1]

    index = context.index(word)
    tmp = list()

    j = index + 1

    # Traverse right
    while j < len(context)-1:

      token = context[j]
      tmp.append(token)
      j += 1

    j = index - 1
    # Traverse left
    while j > 0:

      token = context[j]
      tmp.append(token)
      j -= 1

    for token in tmp:
      c_contexts[token] = c_contexts.get(token, 0) + counts

# Computes occurrence of a word in the corpus at time t
def count_word():

  freq = 0
  for item in contexts:

    count = item[1]
    freq += count

  return freq;

# Compute a join probability of a word and its neighbor
# P(w, c_i) =  count(w, c_i)
#             --------------
#              count(w, c_j)
def joint_proba( neighbor ):

  numContexts = c_contexts[neighbor]
  prob = numContexts / len(c_contexts)

  return prob;

def main():

  get_topn()

  print("Number of CPUs: %d" % mp.cpu_count())
  pool = mp.Pool(10)

  #feats = pool.starmap(extract_feats, ((i,word) for i,word in enumerate(topn)))
  rates = pool.starmap(extract_rates, ((i,word) for i,word in enumerate(topn)))

  pool.close()
  pool.join()

if __name__ == '__main__':

  #load_data()

  word = ''
  if word:

    rates = extract_rates(0,word)
    print(rates)
    #rates = compute_rates(word)
    #print(rates)

    #sTime = time.time()
    #extract_feats(word)
    #print(time.time() - sTime)

  else: main()

