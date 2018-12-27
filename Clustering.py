import pickle

from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


with (open("data_batch_1", "rb")) as openfile:
    while True:
        try:
            dict = pickle.load(openfile, encoding='bytes')
        except EOFError:
            break

data = dict.get(b'data')
datasetLabel = dict.get(b'labels')
features=[]
features2D=[]
for i in data:
    avgRed=0
    avgGreen=0
    avgBlue=0
    for j in range(1024):
     avgRed+=i[j]
    for j in range(1024,2048):
     avgGreen+=i[j]
    for j in range(2048,3072):
     avgBlue+=i[j]
    ourAverage=(avgRed/1024,avgGreen/1024,avgBlue/1024)
    ourAverage2D=(avgBlue/1024,avgGreen/1024)
    features.append(ourAverage)
    features2D.append(ourAverage2D)
plt.plot()
X=np.array(features)
plt.title('k means ')
#X= np.array(features)
#clusters --> 10*3(10 clusters each have average of R B G
CentersOfCluster=[]
distortion=[]
k=2
#get Random positions For Clusters
for i in range(0,k):
    CentersOfCluster.append(features[i])
#number of max iterations
iterations=20
#number pf clusters

BelongToCluster=[0]*(len(features))*k
for it in range(0,iterations):
  dist = 0
  #get distance betweeen labels and clusters
  for labels in range(0,len(features)):
    mnDis=1e16
    for clusters in range(0,k):
      ed=0;
      for dim in range(0,len(features[labels])):
         ed+=(features[labels][dim] - CentersOfCluster[clusters][dim])**2
      if ed < mnDis:
        mnDis=ed
        BelongToCluster[labels]=clusters

  #get new centers
  for centers in range(0,k):
    total=0
    points=[0]*len(features[0])
    for labels in range(0,len(features)):
        if BelongToCluster[labels] == centers:
           total+=1
           for dim in range(0,len(features[0])):
             points[dim]+=features[labels][dim]
    if total>0:
      for dim in range(0,len(features[0])):
        points[dim]/=total
      CentersOfCluster[centers]=points

  #get distortion after each iteration
  for cluster in range(0,k):
      total = 0
      points = [0] * len(features[0])
      for labels in range(0, len(features)):
          if BelongToCluster[labels] == cluster:
              total += 1
              for dim in range(0, len(features[0])):
                  points[dim] += features[labels][dim]
      if total > 0:
          for dim in range(0, len(features[0])):
              points[dim] /= total
          for label in range(0,len(features)):
             if BelongToCluster[label] == cluster:
                for dim in range(0, len(features[labels])):
                  dist += (CentersOfCluster[cluster][dim] - features[label][dim]) ** 2
  distortion.append(dist)

plt.plot(distortion)
plt.xlabel("iterations")
plt.ylabel("summation of diff. between each object and nearest cluster to it")
plt.show()













