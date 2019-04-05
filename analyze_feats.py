from mixed_linear import *

sYear = 1800
fYear = 2000

feat_decade = dict()

def get_ind( feat_name ):

  return {

    'D': 0,
    'I': 1,
    'F': 2

  }[feat_name];

def get_top_words( feat_name ):

  ind = get_ind(feat_name)
  top_words = dict()
  for word in feat_decade:

    max_val = 0.0
    for year in feat_decade[word]:

      feat = feat_decade[word][year]
      feat = feat[ind]
     
      if max_val < feat:
        max_val = feat

    top_words[word] = max_val

  print("Finished computing max features.")
  print("Sorting..")

  if feat_name == 'D':

    ranking = list(sorted(top_words.items(), key=lambda kv: kv[1]))
   
  else:

    ranking = list(sorted(top_words.items(), key=lambda kv: kv[1]))

  N = 10 # Experiment with this value
  print(ranking[:N])

def get_top( feat_name, N = 10 ):

  feat_max = dict()
  for word in feat_decade:

    ind = get_ind(feat_name)
    i_max = np.argmax([ val[ind] for val in feat_decade[word].values() ])

    yr = list(feat_decade[word].keys())[i_max]
    feat_max[word] = feat_decade[word][yr]

  # Lower D value indicate less contextually diverse
  if feat_name == 'D':

    top_N = sorted(feat_max.items(), key=lambda item: item[1])[:N]    

  else:

    top_N = sorted(feat_max.items(), key=lambda item: item[1], reverse=True)[:N]

  output = dict()
  for item in top_N:

    word = item[0]
    yr = item[1]

    feat_val = feat_decade[word][yr]
    output[word] = feat_val[ind]

  return output;

def main():

  global feat_decade

  get_topn()
  feat_decade = read_features()

  if len(feat_decade) < 0: return;

  get_top_words('F')
  return;

  outdir = "./top_feats.txt"
  handler = open(outdir, 'w')
  
  # Compute N most contextually diverse words
  words_d = get_top('D')

  handler.write("Top 10 contextually diverse words(F_D):\n")
  for item in words_d.items():

    handler.write(item[0] + " " + str(item[1]) + '\n')

  handler.write('\n')

  # Compute N most informative words
  words_i = get_top('I')

  handler.write("Top 10 informative words(F_I):\n")
  for item in words_i.items():

    handler.write(item[0] + " " + str(item[1]) + '\n')

if __name__ == "__main__":

  main()
