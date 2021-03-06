Birds Project

#import package
import findspark
findspark.init('/data/spark-1.6.0-bin-hadoop2.6')
from pyspark import SparkContext, HiveContext
from pyspark.sql import functions as F
from pyspark.sql import Window as w
%matplotlib inline
import seaborn as sns
import datetime as datetime
import pandas as pd
import numpy as np
import math
import time
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import StringType, DoubleType, IntegerType
from shapely.wkb import loads
from shapely import wkt
from numpy.lib.stride_tricks import as_strided
from pyspark.sql import SQLContext

# load Spark and HiveContext
sc = SparkContext()
hc = HiveContext(sc)

Set the area of Polderbaan and calculate the max relative distance of target bird in each trajectory
Extend the Polderbaan area in terms of the max distance

# the area of Polderbaan
l_lon = 4.706
t_lat = 52.368
r_lon = 4.717
b_lat = 52.325
dlon =  r_lon-l_lon
dlat =  t_lat-b_lat 
length0 =  r_lon*111.699 * np.cos(t_lat * np.pi/180) - l_lon*111.699 * np.cos(t_lat * np.pi/180)
length = (int(np.ceil(length0*10)))/10.
width0 =  t_lat*110.574 - b_lat*110.574
width =  (int(np.ceil(width0*10)))/10.

# the potential extended distance in terms of the max relative distance in each trajectory
# read track table which contanis trajectory without the planes
track_table = "birds.track"
track = (hc.read.table(track_table).where("classification_id != 1")
                                   .where("classification_id != 4")
                                   .where("classification_id != 5")
                                   .where("classification_id != 9")
                                   .where("classification_id != 10"))

# define the function for calculate the max relative distance
def max_dist(b):
    return max(((np.array(b)[1:,0]*111.321*np.cos(np.array(b)[1:,1]*np.pi/180) 
                 - b[0][0]*111.321*np.cos(b[0][1]*np.pi/180))**2 + 
                ((np.array(b)[1:,1] - b[0][1])*111)**2)**.5)

def do_something_to_cell(geo_string):
    return np.array([cell.split(' ') for cell in str(geo_string[14:-1]).split(',')]).astype(float)

udf = UserDefinedFunction(lambda x: str(max_dist(do_something_to_cell(x))), StringType())


track_trajectory = (track
                       .select('st_astext','classification_id','timestamp_start','timestamp_end')
                       .dropna()
                       .withColumn('max_dist', udf(F.col('st_astext')))
                       .withColumn('max_dist', F.col('max_dist').astype('float'))
                       .select('max_dist','classification_id')
                   )

nth_percentile = math.ceil(track_trajectory
                    .sort(track_trajectory.max_dist.desc())
                    .limit(int((1-0.95) * track_trajectory.count()))
                    .sort(track_trajectory.max_dist.asc())
                    .first()[0]
                          )
# calculate the 95% largest distance and to extend the exist area in terms of this value
# drop those crazy large distance

# extend the area in terms of the max relative distance
lat_det = nth_percentile/111
lon_det = nth_percentile/(np.cos(b_lat*np.pi/180)*111.321)
bound_x = [(l_lon-lon_det,b_lat-lat_det)]
bound_y = [(r_lon+lon_det,t_lat+lat_det)]
print nth_percentile


Divided the area into grid (6 by 4)


x_n= 4
y_n= 6

x_cells_index = pd.RangeIndex(0,x_n,1)
y_cells_index = pd.RangeIndex(0,y_n,1)
x_cells=pd.qcut(([l_lon-lon_det,r_lon+lon_det]),x_n,retbins=True)[1:]
y_cells=pd.qcut(([b_lat-lat_det,t_lat+lat_det]),y_n,retbins=True)[1:]
# determine the bound of each cells and the index 

bins_x=np.array(x_cells).tolist()[0]
bins_y=np.array(y_cells).tolist()[0]
# get the bound value
interval_lon = (r_lon+lon_det-(l_lon-lon_det))/x_n
interval_lat = (t_lat+lat_det-(b_lat-lat_det))/y_n
min_lon = l_lon-lon_det
min_lat = b_lat-lat_det

Get the coordinates of each position


# read trackestimate table which contanis each location
trackestimate_table = "birds.trackestimate"
trackestimate = hc.read.table(trackestimate_table) 
track_subset =trackestimate.persist()
# transform the time format to drop those half seconds
trackestimate_subset = track_subset.withColumn('dt', F.date_format('timestamp', 'yyyy-MM-dd HH:mm'))
# define function to get the coordinates
# udf_x = UserDefinedFunction(lambda x: str(loads(x,hex=True).__geo_interface__['coordinates'][0]), StringType())
# udf_y = UserDefinedFunction(lambda x: str(loads(x,hex=True).__geo_interface__['coordinates'][1]), StringType())
def do_something_to_cell(geo_string):
    return [cell.split(' ') for cell in str(geo_string[9:-1]).split(' ')]

udf_x = UserDefinedFunction(lambda x: do_something_to_cell(x)[0][0], StringType())
udf_y = UserDefinedFunction(lambda x: do_something_to_cell(x)[1][0], StringType())
# transform the coordinates and the datatype
trackestimate_subset_coord=trackestimate_subset.withColumn('position_x', udf_x(F.col('st_astext')).astype('float'))
trackestimate_subset_coord=trackestimate_subset_coord.withColumn('position_y',udf_y(F.col('st_astext')).astype('float'))


Assign the coordinates into cells

# read track table which contanis trajectory
track_table = "birds.track"
track = hc.read.table(track_table) # input track table which contanis trajectory
track = track.select('id','classification_id')
track_join = (trackestimate_subset_coord
            .join(track,on=track.id==trackestimate_subset_coord.track_id)
            .drop(track.id)
            .drop(trackestimate_subset_coord.st_astext)
            )
track_grid = (track_join.filter(track_join.position_x > bound_x[0][0])
                .filter(track_join.position_x < bound_y[0][0])
                .filter(track_join.position_y > bound_x[0][1])
                .filter(track_join.position_y < bound_y[0][1])).persist()
track_grid = track_grid[track_grid['classification_id'].isin(2,3,6,7,8)]


# check the mean position change
data_kernal = track_grid.select("dt","position_x","position_y")
data_kernal = data_kernal.withColumn('date', F.date_format('dt', 'yyyy-MM-dd HH'))
data_kernal_mean=data_kernal.groupBy('date').mean('position_x','position_y')
data_kernal_group=(data_kernal.join(data_kernal_mean,data_kernal.date==data_kernal_mean.date,"inner").drop(data_kernal_mean.date)
                   .withColumn('subX',(F.col('position_x')-F.col('avg(position_x)'))**2)
                   .withColumn('subY',(F.col('position_y')-F.col('avg(position_y)'))**2)
                   )
data_kernal_group=data_kernal_group.groupBy('date').mean('subX','subY')
data_kernal_group=data_kernal_group.withColumn('sdist',(F.col('avg(subX)')+F.col('avg(subY)'))**0.5)
kernal = data_kernal_group.join(data_kernal_mean,data_kernal_group.date==data_kernal_mean.date).drop(data_kernal_mean.date)
kernal = kernal.withColumn('dates', F.date_format('date', 'yyyy-MM-dd')).withColumn('hours', F.date_format('date', 'HH'))


Aggregate time into one minutes and join all the features


# assign time scale in order to aggregate data into it 
#time_interval = 60
#start_timestep = 1435708800 - 7200 # 2015-07-01 00:00:00 2 hours difference 
#data = (data
#        .withColumn('timestep', F.ceil((F.unix_timestamp('dt')-sc._jsc.startTime())/time_interval))
#        .drop('radar_id')
#        )
# assign each location to the cells index
track_grid_x = track_grid.withColumn('x_categories', F.ceil((F.col('position_x') - min_lon)/interval_lon))
data = track_grid_x.withColumn('y_categories', F.ceil((F.col('position_y') - min_lat)/interval_lat))
data = data.fillna(0).drop('radar_id')
data = data.withColumn("location_index",F.concat(data.y_categories,data.x_categories)).drop('x_categories').drop('y_categories')
# join the attribute features
data_count=data.groupBy('location_index', 'dt').count()
attribute=data.groupBy('location_index', 'dt').mean('position_x','position_y',
                                                     'velocity','airspeed',
                                                     'heading','heading_vertical',
                                                     'peak_mass','mass','mass_correction')
cond = [data_count.location_index == attribute.location_index, 
        data_count.dt == attribute.dt]
data_grid = (attribute.join(data_count, cond, 'inner')
                 .drop(attribute.dt)
                 .drop(attribute.location_index)
             )
data_grid = data_grid.orderBy('dt','location_index')
oldColumns = data_grid.schema.names
newColumns = ["position_x","position_y","velocity","airspeed","heading","heading_vertical",
              "peak_mass","mass","mass_correction","location_index","dt","count"]

data_grid_new = reduce(lambda data_grid, idx: data_grid.withColumnRenamed(oldColumns[idx], newColumns[idx]), xrange(len(oldColumns)), data_grid)



EXAMPLE
data_grid_new = (data_grid_new.where("dt > '2016-04-07 00:00'")
                              .where("dt < '2016-04-20 00:00'"))
                              
Reshape and cast count as example and aggregate it into neighboorhood cell information structure

# extract the count of location points
grid_space = data_grid_new.select('location_index','dt','count').withColumn('dt', F.date_format('dt', 'yyyy-MM-dd HH:mm:ss'))
grid_space = grid_space.groupBy('dt').pivot('location_index').avg('count').fillna(0).sort('dt')



data = grid_space.toPandas()
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
alldf = pd.concat([dfindex,df1, df2, df3, df4, df5], axis=1).iloc[40:,].reset_index(drop=True)
alldf.columns = ['timestep', 'location',
                 'lt1','lm1','lb1','mt1','mm1','mb1','rt1','rm1','rb1',
                 'lt2','lm2','lb2','mt2','mm2','mb2','rt2','rm2','rb2',
                 'lt3','lm3','lb3','mt3','mm3','mb3','rt3','rm3','rb3',
                 'lt4','lm4','lb4','mt4','mm4','mb4','rt4','rm4','rb4',
                 'lt5','lm5','lb5','mt5','mm5','mb5','rt5','rm5','rb5']
                 
                 
Aggregate all features into neighboorhood cell information structure

attr = data_grid_new.toPandas()
attr=attr.rename(columns = {'dt':'timestep'})
attr['timestep'] =  pd.to_datetime(attr['timestep'], format='%Y-%m-%d %H:%M:%S')
attr['location_index'] = "loc"+ attr['location_index'].astype(str)
time =pd.date_range(attr['timestep'].min(),attr['timestep'].max(), freq='min')
timestep = pd.DataFrame(np.repeat(time, 24))
timestep = pd.DataFrame(np.repeat(time, 24))
loc = ("loc11","loc12","loc13","loc14",
       "loc21","loc22","loc23","loc24",
       "loc31","loc32","loc33","loc34",
       "loc41","loc42","loc43","loc44",
       "loc51","loc52","loc53","loc54",
       "loc61","loc62","loc63","loc64")
location=pd.DataFrame(np.tile(loc,time.shape[0]))
datapre = pd.concat((timestep,location),axis=1)
datapre.columns=['timestep','location_index']
attr['location_index'] = attr['location_index'].astype(str)
alldata = pd.merge(datapre, attr, on=['location_index','timestep'],how="left").fillna(0)


ts =pd.date_range(attr['timestep'].min() +  pd.Timedelta(minutes=5),attr['timestep'].max(), freq='min')
timestep = pd.DataFrame(np.repeat(ts, 8))
loc = ("loc22","loc23",
       "loc32","loc33",
       "loc42","loc43",
       "loc52","loc53")
timediff = attr['timestep'].max() - (attr['timestep'].min() +  pd.Timedelta(minutes=5))
timediff = timediff / np.timedelta64(1, 'm') +1
location=pd.DataFrame(np.tile(loc,timediff))
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
    columns =['lt1','lm1','lb1','mt1','mm1','mb1','rt1','rm1','rb1',
              'lt2','lm2','lb2','mt2','mm2','mb2','rt2','rm2','rb2',
              'lt3','lm3','lb3','mt3','mm3','mb3','rt3','rm3','rb3',
              'lt4','lm4','lb4','mt4','mm4','mb4','rt4','rm4','rb4',
              'lt5','lm5','lb5','mt5','mm5','mb5','rt5','rm5','rb5']
    columns = [s + alldata.columns.values[m] for s in columns]
    alldf.columns = columns
    pre = pd.concat([pre,alldf.reset_index(drop=True)],axis=1)
    
    
total = pre.merge(attr[['location_index','timestep','peak_mass']],how="left",on=['location_index','timestep']).fillna(0)

# weather table
ts =pd.date_range(attr['timestep'].min() +  pd.Timedelta(minutes=5),attr['timestep'].max(), freq='min')
timediff = attr['timestep'].max() - (attr['timestep'].min() +  pd.Timedelta(minutes=5))
timediff = timediff / np.timedelta64(1, 'm') +1
index =range(timediff.astype('int'))
timestep = pd.DataFrame({'timestep': np.repeat(ts, 8),'index': np.repeat(index, 8)})
loc = ("loc22","loc23",
       "loc32","loc33",
       "loc42","loc43",
       "loc52","loc53")
location=pd.DataFrame(np.tile(loc,timediff))
pre_weather = pd.concat((timestep,location),axis=1)
pre_weather.columns=['index','timestep','location_index']
pre_weather['hour'] =pre_weather.timestep.apply(lambda x:pd.to_datetime(x).hour)
pre_weather['day_index'] = pre_weather.hour.apply(lambda x: math.trunc(x/6)).astype('category')
pre_weather['tenmin'] = pre_weather.timestep.apply(lambda x: 
                                          x - datetime.timedelta(minutes=pd.to_datetime(x).minute 
                                                                 - math.trunc(pd.to_datetime(x).minute/10)*10))

weather=pd.read_csv('/data/capdec/raw/capdec/weather_2016.txt',sep='\t')
weather = weather[["DATE-LT","TIME-LT","VIS","CEIL","GML","BZO"," ~VIS","~CEIL","~GML","~WDIR","~WSPD","~WGUS","~SHWR"]]
weather['DATE-LT'] = pd.to_datetime(weather['DATE-LT'],format='%d-%m-%Y')
weather['tenmin'] =  pd.to_datetime(pd.to_datetime(weather['DATE-LT'].astype(str) + ' ' + weather['TIME-LT']))
weather = weather.drop('TIME-LT',axis=1).drop('DATE-LT',axis=1)
weather_join = pd.merge(pre_weather, weather,on='tenmin',how='left')


birrrds = pd.merge(total,weather_join,on=['location_index','timestep'],how='left')

birrrds['folds'] = pd.cut(birrrds.index,20,labels=range(20))
import h2o
h2o.init()

from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch
features = birrrds.columns.drop(['timestep','folds','peak_mass']).tolist()
rmse = performance = all_models = best_lambda = best_performance = []
alpha_opts =  [0,1]
lam = np.arange(0, 1, 0.01)
for j, alpha in enumerate(alpha_opts):
    hyper_parameters =  alpha_opts[j]
    for k in range(1,3,1):
        Lambda = lam[k]
        for i in range(9):
            train = birrrds.loc[birrrds['folds'].isin(range(10+i))]
            valid = birrrds.loc[birrrds['folds'].isin([i+11])].reset_index(drop=True)
            training =  h2o.H2OFrame((train.values.tolist()))
            validation = h2o.H2OFrame((valid.values.tolist()))
            training.names = train.columns.tolist()                               
            validation.names = valid.columns.tolist()
            models = H2OGeneralizedLinearEstimator(family = "gaussian",Lambda = Lambda,standardize=True,
                                            alpha = hyper_parameters, max_iterations= 100)
            models.train(y = 'peak_mass', x = features, training_frame = training,validation_frame=validation)
            rmse = rmse + [np.sqrt(models.mse(valid=True))]
    performance = performance + [np.mean(rmse)]
    loc = np.argmin(performance) + 1
    b_lambda = lam[loc]
best_lambda =  best_lambda +  [b_lambda]
best_performance = best_performance + [min(performance)]


# find best neural nets 
rmse_NN = performance_NN = []
features = birrrds.columns.drop(['timestep','folds','peak_mass']).tolist()
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
import random
for j in range(1):
    hidden = random.sample(range(1, 200), 4)
    for i in range(9):
        train = birrrds.loc[birrrds['folds'].isin(range(10+i))]
        valid = birrrds.loc[birrrds['folds'].isin([i+11])].reset_index(drop=True)
        training =  h2o.H2OFrame((train.values.tolist()))
        validation = h2o.H2OFrame((valid.values.tolist()))
        training.names = train.columns.tolist()                               
        validation.names = valid.columns.tolist()
        models_NN = H2ODeepLearningEstimator(activation="Rectifier",hidden=hidden,distribution ="gaussian",epochs =10,
                                      loss ="Quadratic",l1=1e-05, max_w2=10 )
        models_NN.train(y = 'peak_mass', x = features, training_frame = training,validation_frame=validation)
        rmse_NN = rmse_NN + [np.sqrt(models_NN.mse(valid=True))]
    performance_NN = performance_NN + [np.mean(rmse_NN)]
    
# NN classification set the cut off value
cutoff = np.mean(birrrds.peak_mass)
maxcutoff = np.max(birrrds.peak_mass)
birrrds['class'] = pd.cut(birrrds['peak_mass'],bins=[0,cutoff,maxcutoff],labels=['low','high'],include_lowest=True)


features = birrrds.columns.drop(['timestep','folds','class','peak_mass']).tolist()
logloss_NN = performance_NN = performance_auc = auc_NN = layer = []
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
import random
for j in range(1):
    hidden = random.sample(range(1, 200), 4)
    for i in range(9):
        train = birrrds.loc[birrrds['folds'].isin(range(10+i))]
        valid = birrrds.loc[birrrds['folds'].isin([i+11])].reset_index(drop=True)
        training =  h2o.H2OFrame((train.values.tolist()))
        validation = h2o.H2OFrame((valid.values.tolist()))
        training.names = train.columns.tolist()                               
        validation.names = valid.columns.tolist()
        models_NN = H2ODeepLearningEstimator(activation="Tanh",hidden=hidden,distribution ="bernoulli",epochs =10,
                                      loss ="CrossEntropy",l2=1e-05)
        models_NN.train(y = 'class', x = features, training_frame = training,validation_frame=validation)
        logloss_NN = logloss_NN + [models_NN.logloss(valid=True)]
        auc_NN = auc_NN + [models_NN.auc(valid=True)]
    performance_NN = performance_NN + [np.mean(logloss_NN)]
    performance_auc = performance_auc + [np.mean(auc_NN)]
    layer = layer + hidden
    
train_all = birrrds.loc[birrrds['folds'].isin(range(19))]
test = birrrds.loc[birrrds['folds'].isin([19])].reset_index(drop=True)
training_all =  h2o.H2OFrame(train.values.tolist())
tests = h2o.H2OFrame(test.values.tolist())
training_all.names = train_all.columns.tolist()                               
tests.names = test.columns.tolist()
models_NN = H2ODeepLearningEstimator(activation="Tanh",hidden=hidden,distribution ="bernoulli",epochs =10,
                                      loss ="CrossEntropy",l2=1e-05)
models_NN.train(y = 'class', x = features, training_frame = training_all)
pred = models_NN.predict(tests)


data = pd.concat([pd.DataFrame(test['class']),h2o.as_list(pred)],axis=1)
accurency = sum(data['class'] == data['predict'])/round(data.shape[0])
h2o.shutdown()
