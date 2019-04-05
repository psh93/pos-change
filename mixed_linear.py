from __future__ import division
from nltk.corpus import stopwords
from sklearn import preprocessing as scaler
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import statsmodels.formula.api as smf
import numpy as np
import json
import time
import os
import re
import pandas
import warnings
import math

warnings.filterwarnings("ignore")

# Static global variables

sYear = 1800
fYear = 2000
th = ['JSD', 'index', 'D', 'I', 'F' ]

# Dynamic global variables
topn = list()
feat_decade = dict()
rate_decade = dict()
mdf = ''
lms = list()

# Output: X[word][year] = [ D, I, F ]
def read_features():

  global feat_decade
  DI_dir = './output/feats/'

  # Load F
  F_dir = './output/vocab.json'
  handler = open(F_dir,'r')

  F = json.load(handler)
  if len(F) == 0: return [];

  print("Loading features..")
  sTime = time.time()
  for word in topn:

    # Load D and I
    file = word + '_f.json'
    fullFile = os.path.join(DI_dir, file)
    handler = open(fullFile,'r')
    
    DI = json.load(handler)
    if len(DI) == 0: return [];

    feat_decade[word] = dict()
    for year in range(sYear,fYear+1,10):

      year = str(year)
      # Store D and I
      if year not in DI:

        feat_decade[word][year] = [0.0,0.0]

      else:

        feat_decade[word][year] = DI[year][:2]

      # Store F
      if year in F[word]:

        feat = F[word][year]
        feat = normalize_counts(feat,year)

        feat_decade[word][year].append(feat)

      else:

        feat_decade[word][year].append(0.0)

  fTime = time.time()

  print("Loaded features(%.4fs)" % (fTime-sTime))
  return feat_decade;

# Output: X[word][year] = 0.x
def read_JSD():

  global rate_decade
  indir = 'output/jsd/'

  print("Loading JS divergence..")
  sTime = time.time()
  for word in topn:

    file = word + '_j.json'
    fullFile = os.path.join(indir, file)
    handler = open(fullFile, 'r')
    
    rates = json.load(handler)
    if rates is not None:
      rate_decade[word] = rates

  fTime = time.time()
  print("Loaded jsd(%.4fs)" % (fTime-sTime))

def get_topn():

  f = './feat_list.txt'
  global topn

  if True:

    handler = open(f, 'r')
    for word in handler.readlines():
    
      word = word.strip()
      topn.append(word)
  
  print("Analyzing %d common words" % len(topn))
  #return topn;

def regr( X_yr, Y_yr ):

  global mdf
  tmp = list()
  for i,word in enumerate(topn):

    # Features(D,I,F)
    X = feat_decade[word][str(X_yr)]

    # Rate
    Y = rate_decade[word][str(Y_yr)]
    tmp.extend([ [Y, (i+1)] + X ])
 
  data = pandas.DataFrame(tmp, columns=th)

  md = smf.mixedlm("JSD ~ D + I + F", data, groups='index')
  mdf = md.fit()

  #print(mdf.summary())
  return mdf;

def perform_regr():

  global lms

  feat_yrs = list(range(1800,1990+1,10))
  rate_yrs = list(range(1810,2000+1,10))

  for X_yr, Y_yr in zip(feat_yrs, rate_yrs):

    mdf = regr(X_yr, Y_yr)
    lms.append(mdf)

    #print("JSD(Y) at %s vs features(X) at %s" % (Y_yr, X_yr))

# X[index] = coff
# Where all coeff's are positive at index
def range_coff( feat_name, log = True ):

  coff = list() 
  for mdf in lms:

    val = mdf.params[feat_name]
    coff.append(val)

  minV = min(coff)
  maxV = max(coff)

  print("Range of feature %s: [%f, %f]" % (feat_name,minV,maxV))
  return coff;

def range_p( feat_name  ):

  pVals = list()
  for mdf in lms:

    pVal = mdf.pvalues[feat_name]
    if type(pVal) is not None:
      pVals.append(pVal)

  print("p-values:\n")
  print(pVals)
  print(min(pVals), max(pVals))

def plot_feats():

  decs = [ decade+1 for decade in range(20) ]

  # Initialization
  Y = dict()
  error_Y = dict()
  for feature in th[2:]: # [D,I,F]

    Y[feature] = list()
    error_Y[feature] = list()

  feats = { 'D': "Contextual diversity",
            'I': "Word informativeness",
            'F': "Word frequency"}

  ranges = { 'D': [-0.0007, 0.000400],
             'I': [-0.0006, 0.003],
             'F': [-0.005, -0.0009] }

  # Get data
  for mdf in lms:

    for feature in th[2:]:

      # Coefficient
      coef = mdf.params[feature]
      Y[feature].append(coef)

      # Std. dev.
      std = mdf.bse[feature]
      error_Y[feature].append(std)

  # Plot (1 = 'all')
  choice = 1

  if choice == 1:

    traces = list()
    for feature in th[2:]:

      trace = go.Bar(
              x = decs,
              y = Y[feature],
              name = feats[feature],
              error_y = dict( type = 'data', array = error_Y[feature], visible = True ))

      traces.append(trace)

    layout = go.Layout( barmode = 'group',
                        yaxis = dict( range = ranges[feature] ))
    fig = go.Figure(data = traces, layout = layout)
    py.iplot(fig, filename="comb-bar")

  else:

    for feature in th[2:]:
     
      trace1 = go.Bar(
                x = decs,
                y = Y[feature],
                error_y = dict( type = 'data', array = error_Y[feature], visible = True ))

      if feature == 'D':
          layout = go.Layout(
          xaxis = dict( title = "Decades", showticklabels = True, range = [1,20]),
          yaxis = dict( title = "Coefficients (" + feats[feature]  + ")", showticklabels = True,
                        exponentformat = 'e', showexponent = "all"))
      else: 
          layout = go.Layout(
          xaxis = dict( title = "Decades", showticklabels = True, range = [1,20] ),
          yaxis = dict( title = "Coefficients (" + feats[feature]  + ")", showticklabels = True,
                        range = ranges[feature]))

      data = [trace1]
      fig = go.Figure(data = data, layout = layout)
      py.iplot(fig, filename = feature + '_bar')

def plot_errorbar():

  # Get a range of coeff.
  ranges = dict()
  for feature in th[2:]:

    r = np.array(range_coff(feature))
    ranges[feature] = r

  # Get a range of rates
  ranges["Rate"] = list()
  for mdf in lms:

    rate = mdf.params["Intercept"]
    ranges["Rate"].append(rate)

  F_mean = np.mean(ranges['F'])
  I_mean = np.mean(ranges['I'])
  D_mean = np.mean(ranges['D'])

  F_std = np.std(ranges['F'])
  I_std = np.std(ranges['I'])
  D_std = np.std(ranges['D'])

  feat_means = [ F_mean, I_mean, D_mean ]
  error = [ F_std, I_std, D_std ]
  x_pos = np.arange(len(th[2:]))

  fig,ax = plt.subplots()
  ax.bar(x_pos,feat_means,yerr=error,align='center',alpha=0.5,ecolor='black',capsize=10)
  ax.set_ylabel('Coefficient of Features')
  ax.set_xticks(x_pos)
  ax.set_xticklabels(th[2:])
  ax.yaxis.grid(True)

  plt.tight_layout()
  plt.show()

def stat_analysis():

  print("Storing the parameters in the trained models..")
  ranges = dict()
  for feature in th[2:]:

    r = np.array(range_coff(feature))
    ranges[feature] = r
    range_p(feature)

  ranges["Rate"] = list()
  for mdf in lms:

    rate = mdf.params["Intercept"]
    ranges["Rate"].append(rate)

  df = pandas.DataFrame(ranges,columns=list(ranges.keys()))
  print(df)
  print("Converting to sns-readable format.")
  
def normalize_counts( counts, year ):

  indir = './output/counts_tokens.json'
  total_c = json.load(open(indir,'r'))

  total = total_c[year][0]
  div = counts / total
  
  if div == 0: return 0;

  return math.log(div,2);

def get_ind( feat_name ):

  return {

    'D': 0,
    'I': 1,
    'F': 2

  }[feat_name];

def main():
 
  # -Step1- Load top N words for analysis
  get_topn()

  # -Step2- Fit the regression model
  read_features()
  read_JSD()

  # -Step3- Summarize
  perform_regr()
  #stat_analysis()

  # -Step4- Plot
  plot_feats()

if __name__ == '__main__':

  main()
