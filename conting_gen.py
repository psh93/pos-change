from proba_distribution import *

indir = "/u/parksh/csc2611/postrack-master/output/"
f_trans = open(indir+"POS_transition.txt", "w")
f_pop= open(indir+"POS_popularity.txt", "w")

sYear = 1800
fYear = 2000

vocab = list()
JSD = dict()
trans_matrix = ''
pop_matrix = ''

POS = ['NN', 'NNP', 'VB', 'ADJ', 'ADV' ]

def load_vocab():

  global vocab

  print("Loading vocabulary..")
  fpath = "/u/parksh/csc2611/postrack-master/feat_list.txt"
  for fname in open(fpath, 'r').readlines():

    fname = fname.strip()
    vocab.append(fname)

  print("Loaded")

def generate_top_jsd( word = '', store = True ):

  if not store:

    jsds = np.array(get_jsd(word,[sYear,fYear]))
    ind = np.argmax(jsds)

    return ind;

  global JSD
  for word in vocab:

    jsds = get_jsd(word,[sYear,fYear])
    if len(jsds) > 0:
     
        JSD[word] = max(jsds)

    else:

        JSD[word] = 0.0

  top_jsd = [ item for item in reversed(sorted(JSD.items(), key=lambda item: item[1])) ]  
  return top_jsd;

def update_matrix( word ):

  global trans_matrix
  global pop_matrix

  prob_dist = proba_distribution(word)

  # When is the peak of JSD value?
  years = list(range(sYear,fYear+1,10))
  ind = generate_top_jsd(word,store=False)

  maxYear = years[ind]
  minYear = min(prob_dist.keys())

  sPOS = prob_dist[str(minYear)]
  fPOS = prob_dist[str(maxYear)]

  # Compute transition matrix 
  # For example: "how many times NN --> NNP?"
  for POSi in sPOS:
    if POSi not in fPOS:
      for POSj in fPOS:
        if POSj != POSi:

            trans_matrix[POSj][POSi] += 1

            # Write word, years, initial vs after POS tags
            f_trans.write("%s\n" % (word))
            f_trans.write("%s: " % minYear)
            f_trans.write(str(sPOS) + "\n")
            f_trans.write("%s: " % maxYear)
            f_trans.write(str(fPOS) + "\n")
            f_trans.write(trans_matrix.to_string())
            f_trans.write("\n")

  # Compute popularity matrix
  # For example: "has frequency of NN decreased AND other frequencies increased?"
  for POSi in sPOS:

    Pi = sPOS[POSi]
    Pj = fPOS.get(POSi, 0)

    if Pi > Pj:
      for POSj in fPOS:
        if POSj != POSi:

            pop_matrix[POSj][POSi] += 1

            # Write word, years, initial vs after POS tags
            f_pop.write("%s\n" % (word))
            f_pop.write("%s: " % minYear)
            f_pop.write(str(sPOS) + "\n")
            f_pop.write("%s: " % maxYear)
            f_pop.write(str(fPOS) + "\n")
            f_pop.write(trans_matrix.to_string())
            f_pop.write("\n")

def main():

  # Load vocabulary for analysis
  load_vocab()

  # Generate top JSD words
  # top_jsd = generate_top_jsd()

  # Initialize the transition table
  global trans_matrix
  trans_matrix = pandas.DataFrame(index=POS, columns=POS)
  trans_matrix = trans_matrix.fillna(0)

  # Initialize the popularity table
  global pop_matrix
  pop_matrix = pandas.DataFrame(index=POS, columns=POS)
  pop_matrix = pop_matrix.fillna(0)

  # Store top JSD words
  f = open("/u/parksh/csc2611/postrack-master/output/top_JSD.txt", "w")
  for word in vocab:

    f.write(word + "\n")

    update_matrix(word)
    
    print("Transitions:\n %s\n" % trans_matrix)
    print("Freq. changes:\n %s\n" % pop_matrix)

  trans_matrix.loc["TOTAL",:] = trans_matrix.sum(axis=0)
  trans_matrix.loc[:,"TOTAL"] = trans_matrix.sum(axis=1)
  pop_matrix.loc["TOTAL",:] = pop_matrix.sum(axis=0)
  pop_matrix.loc[:,"TOTAL"] = pop_matrix.sum(axis=1)

  f_trans.write("\n" + trans_matrix.to_string())
  f_pop.write("\n" + pop_matrix.to_string())

  f_trans.close()
  f_pop.close()

  print("Transitions:\n %s\n" % trans_matrix)
  print("Freq. changes:\n %s\n" % pop_matrix)

if __name__ == '__main__':

  main()
