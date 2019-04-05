import json

# Output: X[word][year] = counts
def build_vocab(fpath):

  year_pos = json.load(open(fpath+'year_pos.json'))
  vocab = dict()

  for i,word in enumerate(year_pos):

    vocab[word] = dict()
    for (year,pos_counts) in year_pos[word].items():


      counts = sum([ x for x in pos_counts.values() ])
      vocab[word][year] = counts

    print("Processed %d / %d" % (i,len(year_pos)))

  # Save to file
  handler = open(fpath+'vocab.json','w')
  json.dump(vocab,handler)

def main():

  fpath = '/h/173/parksh/csc2611/postrack-master/output/'
  build_vocab(fpath)

if __name__ == '__main__':

  main()
