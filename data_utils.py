import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations



def load_project_dataset(fname, typelabel):
  df = pd.read_csv(fname)
  N = df['ID_TimeSeries'].max() + 1
  F = 4
  D = 1007

  data=np.zeros((N, F, 1, D))
  lb=[]

  jj=0
  ii=0
  index=0
  for el in df['Values']:
    if(jj==4):
      jj=0
      ii+=1
          
    el=el.split(';')
    data[ii][jj]=np.asarray(list(map(float,el[:1007])))

    if(jj==0):
      lb.append(df[typelabel][index])    
      
    jj+=1
    index+=1

  labels=np.asarray(lb)

  return data, labels


def get_data(df):
  N = df.shape[0]
  F = 4
  D = 1007

  data=np.zeros((N, F, 1, D))
  lb=[]

  ii=0
  for index, row in df.iterrows():
    one=[]
    two=[]
    three=[]
    four=[]

    for jj in range(0,1007):
      #print(jj)
      #print(row[0][jj][0])
      one.append(row[0][jj][0])
      two.append(row[0][jj][1])
      three.append(row[0][jj][2])
      four.append(row[0][jj][3])

    for zz in range(0,4):
      if zz==0: data[ii][zz]=np.asarray(list(map(float,one)))
      if zz==1: data[ii][zz]=np.asarray(list(map(float,two)))
      if zz==2: data[ii][zz]=np.asarray(list(map(float,three)))
      if zz==3: data[ii][zz]=np.asarray(list(map(float,four)))
    
    lb.append(row['Class'])    
    ii+=1

  labels=np.asarray(lb)

  return data, labels 
 

def visualize_timeseries (dataset, n):
  for i in range(len(dataset)):
    #print(i)
    sample = dataset[i]

    print(i, sample[0].shape)
    print(sample[0])
    print("label:",sample[1])

        
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(14, 7)
    axs[0, 0].set_title("1 dimension")
    axs[0, 0].plot(np.arange(0, 1007), sample[0][0][-1],'tab:orange')

    axs[0, 1].set_title("2 dimension")
    axs[0, 1].plot(np.arange(0, 1007), sample[0][1][-1],'tab:green')

    axs[1, 0].set_title("3 dimension")
    axs[1, 0].plot(np.arange(0, 1007), sample[0][2][-1],'tab:red')

    axs[1, 1].set_title("4 dimension")
    axs[1, 1].plot(np.arange(0, 1007), sample[0][3][-1])
    fig.tight_layout()
    plt.show()

    if i == n-1:
     return


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets
      
        return torch.LongTensor(np.array(triplets))