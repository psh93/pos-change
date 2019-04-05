from __future__ import division
from scipy.spatial.distance import jensenshannon as JSD
import json
import math
import scipy as sp
import numpy as np
import multiprocessing as mp
import pandas

# Read POS-year counts from process_pos.py
indir = '/u/parksh/csc2611/postrack-master/output/year_pos.json'
word_pos = json.load(open(indir, 'r'))
vocab = word_pos.keys()

# Fine-grained Penn Treebank POS tagset
nouns = ['NN', 'NNS' ]
pnouns = ['NNP', 'NNPS']
verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
adjs = ['JJ', 'JJR', 'JJS']
advs = ['RB', 'RBR', 'RBS']

def shannon_entropy( dist ):
  
  output = 0
  
  for p_i in dist:
    
    if p_i <= 0: continue
    
    v = math.log(p_i, 2) * p_i
    output += v
  
  output = -np.sum(output)
  
  return output;

# Compute JSD between two consecutive decades
def compute_divergence( dists ):

  JSD_list = list()
  dists = np.array(dists).T
  nRows = dists.shape[0] # 21

  for t in range(nRows - 1):

    X = dists[t]
    Y = dists[t+1]

    jsd = JSD(X,Y)
    if np.isnan(jsd):

      JSD_list.append(0.0)
      continue

    JSD_list.append(jsd * jsd)

  # Returns a list of JSD scores
  return JSD_list;

def merge_tag( tag ):
  
  if tag in nouns: return 'NN'
  if tag in pnouns: return 'NNP'
  if tag in verbs: return 'VB'
  if tag in adjs: return 'ADJ'
  if tag in advs: return 'ADV'
  else:
      return '';

  return tag;

# Computes probability distribution of POS for a given word
# Return: dist[year][tag] = probability
def proba_distribution( word ):
  
  year_pos = word_pos[word]

  dist = {}
  for year in range(1800,2000+1,10):
   
    year = str(year)
    dist[year] = {}
    if year not in year_pos: 

      dist[year] = dict()
      for pos in ['NN','NNP','VB','ADJ','ADV']:

        dist[year][pos] = 0.0
    
      continue
           
    pos_counts = year_pos[year]
    for pos, count in pos_counts.items():
        
        tag = merge_tag(pos)
        if tag == '': continue
        dist[year][tag] = dist[year].get(tag, 0) + count
    
    total = sum(pos_counts.values())
    for tag in dist[year]:
        dist[year][tag] /= total

  return dist;

def top_changes(n):

  tuples = list()

  pool = mp.Pool(5)
  tuples = pool.map(get_max_jsd, [ word for word in vocab ])

  # Sort list by JSD values
  indices = list(reversed(sorted(range(len(tuples)), key=lambda i: tuples[i][1])))

  f = open('top_n_JSD.txt', 'w')
  for index in indices[:n]:

    f.write(str((tuples[index][0], tuples[index][1])))
    f.write('\n')

  f.close()

def get_jsd( word, years ):

  probs = proba_distribution(word)

  start = years[0]
  end = years[1]

  data = list()
  for year in range(start,end+1,10):

    year = str(year)
    if year not in probs:

      tmp = dict()
      for POS in ["NN","NNP","ADV","ADJ","VB"]:

        tmp[POS] = 0.0

    else:

      tmp = probs[year]

    data.append(tmp)

  df = pandas.DataFrame(data)
  x = df.columns
  y = list()
  if 'NN' in x:

    Y = df['NN'].fillna(0).tolist()
    y.append(Y)

  if 'NNP' in x:

    Y = df['NNP'].fillna(0).tolist()
    y.append(Y)

  if 'ADV' in x:

    Y = df['ADV'].fillna(0).tolist()
    y.append(Y)

  if 'ADJ' in x:

    Y = df['ADJ'].fillna(0).tolist()
    y.append(Y)

  if 'VB' in x:

    Y = df['VB'].fillna(0).tolist()
    y.append(Y)

  JSDs = compute_divergence( y )
  return JSDs;

# Test
if __name__ == '__main__':
  
  y = [ 0.5, 0.2, 0.1, 0.1, 0.1 ]
  x = [[0.5, 0.5, 0],[0, 0.1, 0.9],
       [0.33, 0.33, 0.33]]
  
  # Expected: 0.5466 
  #print(jenssen_shannon(x))
  
  # Expected: 1.9
  #print(shannon_entropy(y))

  #top_changes(100)

  #get_max_jsd('apple')
  #print(get_jsd('apple', [1950, 2000]))
