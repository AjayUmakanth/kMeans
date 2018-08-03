import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_iris
class k_means:
    def __init__(self,name):
        self.name=name
        self.trained=False        
    def train(self,dataSet):
        dataSet=np.array(dataSet)
        self.trained=True
        f=open(self.name,'wb')
        pickle.dump(dataSet,f)
        f.close()
    def getDataSet(self):
        try:
            f=open(self.name,'rb')
        except:
            pass
            raise Exception(f"{self.name} dosen't exist")
        dataSet=pickle.load(f)
        if(self.trained==False):
            raise Exception("Dataset not trained")
        return dataSet
    def assignCent(self,dataSet,cent,assign):
        for i in range(np.shape(dataSet)[0]):
            d=np.sum((cent-dataSet[i])*(cent-dataSet[i]),axis=1);
            assign[i]=np.argmin(d)
        return assign
    def computeCent(self,dataSet,cent,assign):
        newCent=np.zeros(np.shape(cent))
        num=np.zeros(np.shape(cent)[0])
        for i in range(np.shape(assign)[0]):
            ind=assign[i]
            newCent[ind]+=dataSet[i]
            num[ind]+=1
        num[num==0]=1
        newCent=(newCent.T/num).T
        cost=0.0
        for i in range(len(assign)):
            d=dataSet[i]-newCent[assign[i]]
            cost+=sum(d*d)
        return newCent,cost
    def cluster(self,k,maxItr=10):
        dataSet=self.getDataSet()
        cent=np.zeros((k,np.shape(dataSet)[1]))
        assign=np.zeros(np.shape(dataSet)[0]).tolist()
        cost=0
        for i in range(k):
            cent[i]=dataSet[random.randint(0,np.shape(dataSet)[0]-1)]
        for i in range(maxItr):
            assign=self.assignCent(dataSet,cent,assign)
            [cent,newCost]=self.computeCent(dataSet, cent, assign)        
            if(cost==newCost):
                break
            cost=newCost    
        return cent,assign,cost
    def plotDiffK(self,k):
        points=np.zeros((k,2)).tolist()
        for i in range(1,k+1):
            [dummy,dummy2,cost]=self.cluster(i)
            points[i-1]=[i,cost]
        print(points)
        x=[row[0] for row in points]
        y=[row[1] for row in points]
        plt.plot(x,y,marker='x')
        plt.show()
