import pandas as pd
import numpy as np
data = pd.read_csv("spatialtest.csv")

def transform(arr,time): 
    submatrix = []
    for i in range(0,4):
        for j in range(0,2):
            submatrix.append((arr.reshape(6,4)[i:i+3,j:j+3]).flatten())
    submatrix = np.array(submatrix)
    location = ("loc22","loc23","loc32","loc33","loc42","loc43","loc52","loc53")
    timestep = np.repeat(time,8)
    location = np.array(location)
    neighboor = np.c_[timestep,location,submatrix]
    return neighboor
nrow = data.shape[0]
trans=[]
for k in range(0,nrow):
        trans.append(transform(data.iloc[k,1:],data.iloc[k,0]))
df = np.concatenate(np.array(trans))
dfindex = pd.DataFrame(df).iloc[:,0:2]
zeros = np.zeros((8,9))
df1=pd.DataFrame(np.concatenate((zeros,df[:-8,2:])))
df2=pd.DataFrame(np.concatenate((zeros,zeros,df[:-16,2:])))
df3=pd.DataFrame(np.concatenate((zeros,zeros,zeros,df[:-24,2:])))
df4=pd.DataFrame(np.concatenate((zeros,zeros,zeros,zeros,df[:-32,2:])))
df5=pd.DataFrame(np.concatenate((zeros,zeros,zeros,zeros,zeros,df[:-40,2:])))
alldf = pd.concat([dfindex,df1, df2, df3, df4, df5], axis=1).iloc[40:,]
alldf.columns = ['timestep', 'location',
                 'lt1','lm1','lb1','mt1','mm1','mb1','rt1','rm1','rb1',
                 'lt2','lm2','lb2','mt2','mm2','mb2','rt2','rm2','rb2',
                 'lt3','lm3','lb3','mt3','mm3','mb3','rt3','rm3','rb3',
                 'lt4','lm4','lb4','mt4','mm4','mb4','rt4','rm4','rb4',
                 'lt5','lm5','lb5','mt5','mm5','mb5','rt5','rm5','rb5']
                 

attr = pd.read_csv("test.csv")
attr['location_index'] = "loc"+ attr['location_index'].astype(str)
time= range(1,50)
timestep = pd.DataFrame(np.repeat(time, 24))
loc = ("loc11","loc12","loc13","loc14",
       "loc21","loc22","loc23","loc24",
       "loc31","loc32","loc33","loc34",
       "loc41","loc42","loc43","loc44",
       "loc51","loc52","loc53","loc54",
       "loc61","loc62","loc63","loc64")
location=pd.DataFrame(np.tile(loc,time[-1]))
datapre = pd.concat((timestep,location),axis=1)
datapre.columns=['timestep','location_index']
attr['location_index'] = attr['location_index'].astype(str)
alldata = pd.merge(datapre, attr, on=['location_index','timestep'],how="left").fillna(0)
ts= range(6,50)
timestep = pd.DataFrame(np.repeat(ts, 8))
loc = ("loc22","loc23",
       "loc32","loc33",
       "loc42","loc43",
       "loc52","loc53")
location=pd.DataFrame(np.tile(loc,ts[-6]))
pre = pd.concat((timestep,location),axis=1)
pre.columns=['timestep','location_index']
for m in range(2,alldata.shape[1]):
    col = [0,1,m]
    subdata = alldata.iloc[:,col]
    subdata = subdata.pivot(index='timestep', columns='location_index', values=subdata.columns[2])
    subdata.reset_index(inplace=True)
    nrow = subdata.shape[0]
    trans=[]
    for k in range(0,nrow):
        trans.append(transform(subdata.iloc[k,1:],subdata.iloc[k,0]))
    df = np.concatenate(np.array(trans))
    dfindex = pd.DataFrame(df).iloc[:,0:2]
    zeros = np.zeros((8,9))
    df1=pd.DataFrame(np.concatenate((zeros,df[:-8,2:])))
    df2=pd.DataFrame(np.concatenate((zeros,zeros,df[:-16,2:])))
    df3=pd.DataFrame(np.concatenate((zeros,zeros,zeros,df[:-24,2:])))
    df4=pd.DataFrame(np.concatenate((zeros,zeros,zeros,zeros,df[:-32,2:])))
    df5=pd.DataFrame(np.concatenate((zeros,zeros,zeros,zeros,zeros,df[:-40,2:])))
    alldf = pd.concat([dfindex,df1, df2, df3, df4, df5], axis=1).iloc[40:,2:]
    pre = pd.concat([pre,alldf.reset_index(drop=True)],axis=1)
