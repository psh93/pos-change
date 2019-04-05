import json
import pandas

ngrams = ''
year_range = [1800, 2000]
ctokens_yr = dict()
ctypes_yr = dict()

def load_data():

  global ngrams

  fpath = '/u/parksh/csc2611/postrack-master/output/ngrams.json'
  ngrams = json.load(open(fpath))

# Output: X[year] = total counts
def count_tokens():

  sYear = year_range[0]
  fYear = year_range[1]
  for year in range(sYear, fYear + 1, 10):

    year = str(year)
    corpus = ngrams[year]

    tokens = list()
    for line in corpus:
      tokens.extend(line.split())

    ctokens_yr[year] = [len(tokens)]
    ctypes_yr[year] = [len(set(tokens))]

  print("Number of tokens by year:")
  df = pandas.DataFrame(ctokens_yr)
  print(df)

  print("Number of types by year:")
  df_ = pandas.DataFrame(ctypes_yr)
  print(df_)

  print("Writing to file..")
  write_data()

def write_data():

  fpath = './counts_tokens.json'
  fout = open(fpath, 'w', encoding='utf-8')
  json.dump(ctokens_yr, fout)

  fpath = './counts_types.json'
  fout = open(fpath, 'w', encoding='utf-8')
  json.dump(ctypes_yr, fout)

def main():

  load_data()
  count_tokens()

if __name__ == '__main__':

  main()
