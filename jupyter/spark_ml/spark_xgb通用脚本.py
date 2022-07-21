
# spark-submit --master yarn --queue "root.ai.offline" --driver-memory 15G --executor-memory 20G --num-executors 10 --executor-cores 10 --properties-file /etc/spark/conf/spark-defaults.conf --archives hdfs:///data/ai/anaconda3.zip\#anaconda3 --conf spark.pyspark.driver.python=python3 --conf spark.pyspark.python=./anaconda3/anaconda3/envs/python3.6/bin/python3 --jars hdfs:///data/ai/lib/xgb/0.90/xgboost4j-0.90.jar,hdfs:///data/ai/lib/xgb/0.90/xgboost4j-spark-0.90.jar --py-files=hdfs:///data/ai/lib/xgb/0.90/sparkxgb.zip model_9_baseline_modify_3.py
        
#!/usr/bin/env python
# coding: utf-8

# w,../../helper.zip predict_in.py

import pyspark.ml.feature as ft
from pyspark.ml import Pipeline
import pyspark.ml.evaluation as ev
import pyspark.sql.types as typ
from sparkxgb import XGBoostRegressor

import sys
import time
import math
import feather
import datetime
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast
from pyspark.sql import Window
from pyspark.sql import Row
from pyspark.sql.types import *

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler

# CITYS = (668, 575, 466, 181, 320)
# CITYS_STR = "(" + ",".join([str(c) for c in CITYS]) + ")"
EVENT_DAY = '20210607'

SUFFIX = f"in_model_version_0607_ver21_offline_457_3"
LABEL = "in_num"
APP_NAME = "in_predict_split_457"
def open_spark_session(app_name="ai-train"):
    conf = SparkConf().setMaster('yarn').setAppName(app_name).set('spark.yarn.queue','root.ai.train')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark
spark = open_spark_session(app_name=APP_NAME)


# ========数据拉取========
saved_columns = ['city_id', 'block_id', 'datadate', 'station_id', 'span_id', 'cur_minutes', 'today_minutes', 'is_weekend', 'is_holiday', 'dayofweek', 'tmp', 'cond_code', 'cond_txt', 'wind_deg', 'wind_sc', 'wind_spd', 'hum', 'pcpn', 'pres', 'vis', 'cloud', 'is_rain', 'in_num', 'out_num', 'true_sample', 'last_day_is_rain', 'last_week_is_rain'] +['mean_latest_7_days_bike_in','max_latest_7_days_bike_in','min_latest_7_days_bike_in','med_latest_7_days_bike_in','mean_latest_7_days_last_bike_in','mean_latest_7_days_next_bike_in','mean_latest_14_days_bike_in','mean_latest_28_days_bike_in','mean_latest_5_workdays_bike_in','mean_latest_4_same_dayofweek_bike_in','max_latest_4_same_dayofweek_bike_in','min_latest_4_same_dayofweek_bike_in','med_latest_4_same_dayofweek_bike_in','mean_latest_4_last_dayofweek_bike_in','mean_latest_4_next_dayofweek_bike_in','last_day_last_span_bike_in','last_day_same_span_bike_in','last_day_next_span_bike_in','last_week_last_span_bike_in','last_week_same_span_bike_in','last_week_next_span_bike_in','last_span_bike_in','last_last_span_bike_in','station_id']

feature_columns = ['span_id', 'is_weekend', 'is_holiday', 'dayofweek', 'tmp', 'pcpn', 'is_rain', 'mean_latest_7_days_bike_in', 'med_latest_7_days_bike_in', 'mean_latest_7_days_last_bike_in', 'mean_latest_7_days_next_bike_in', 'mean_latest_14_days_bike_in', 'mean_latest_28_days_bike_in', 'mean_latest_5_workdays_bike_in', 'mean_latest_4_same_dayofweek_bike_in', 'med_latest_4_same_dayofweek_bike_in', 'mean_latest_4_last_dayofweek_bike_in', 'mean_latest_4_next_dayofweek_bike_in', 'last_day_same_span_bike_in',  'last_week_same_span_bike_in']

train_df = spark.sql(f"""
    select city_id, datadate, block_id, span_id, is_weekend, is_holiday, dayofweek,tmp, pcpn,  is_rain, mean_latest_7_days_bike_in, med_latest_7_days_bike_in, mean_latest_7_days_last_bike_in, mean_latest_7_days_next_bike_in, mean_latest_14_days_bike_in,  mean_latest_28_days_bike_in, mean_latest_5_workdays_bike_in, mean_latest_4_same_dayofweek_bike_in, med_latest_4_same_dayofweek_bike_in, mean_latest_4_last_dayofweek_bike_in,  mean_latest_4_next_dayofweek_bike_in, last_day_same_span_bike_in,  last_week_same_span_bike_in, in_num
    
    
from ai.ft_ai_jw_offline_block_in_out_feature_30min_da
where 

    event_day>='20210501'
    and event_day<='20210606'
    and mean_latest_28_days_bike_in>0.3
    and block_version_code = 21
        

""")

# 样本大小：
train_df.first()
train_df.cache()

# ========模型训练========
xgboost_small = XGBoostRegressor(
    featuresCol="features", labelCol=LABEL, predictionCol="prediction", #weightCol="weight",
    numRound=100, eta=0.1, gamma=0.1, alpha=0.0, minChildWeight=1.0, maxDepth=6,
    seed=123, evalMetric="rmse",
    objective="reg:squarederror",
    #objectiveType=None,
    numEarlyStoppingRounds=0,
    trainTestRatio=0.95,
    #customObj=adjusted_mse,
    subsample=1.0,
    colsampleBylevel=1.0,
    colsampleBytree=1.0,
    ## EXCLUDED: customEval=None,
    baseScore=0.5,
    cacheTrainingSet=False,
    growPolicy="depthwise",
    lambdaBias=0.0,
    maxDeltaStep=0.0,
    #maxLeaves=None,
    missing = 0.0,
    normalizeType="tree",
    nthread=1,
    numWorkers=1,
    rateDrop=0.0,
    sampleType="uniform",
    scalePosWeight=1.0,
    sketchEps=0.03,
    skipDrop=0.0,
)
#A feature transformer that merges multiple columns into a vector column.
vectorAssembler = VectorAssembler().setInputCols(feature_columns).setOutputCol("features")
xgb_pipeline = Pipeline().setStages([vectorAssembler, xgboost_small])

# 训练测试数据
testDF = train_df.filter(train_df.datadate>="2021-05-31")
testDF.cache()
trainDF = train_df.filter(train_df.datadate<="2021-05-30")
trainDF.cache()

# 结果保存
def save_result(model, df, prefix, save_hive):
    pre = model.transform(df)    #predict
#     predictResult = pre.select(['datadate', 'block_id', 'span_id', 'prediction', 'in_num', 'dayofweek']).toPandas()
#     predictResult.to_csv(f"xgbresult_in_{prefix}.csv",index=False)
    
    if save_hive:
        pre.registerTempTable("ai_in_result_temp_table")
        spark.sql(f"""insert overwrite table ai.dws_ai_block_in_predict_detail partition(version='{prefix}', event_day='{EVENT_DAY}')
    select city_id, block_id, span_id, prediction predict, in_num real_in from ai_in_result_temp_table""")
        
    return pre


hdfs_path = "hdfs:///data/ai/models/liuxuanheng/"

# 训练
model = xgb_pipeline.fit(trainDF)   

# 模型存hdfs
model.write().overwrite().save(hdfs_path+SUFFIX+'.model')

# 模型存本地jupyter
model.stages[-1].nativeBooster.saveModel(
    f"file:///data/apps/modules/jupyter_multi_users/liuxuanheng/jupyter/spark-shell/model_pkg/{SUFFIX}.model")

# 预测存hive
_ = save_result(model, testDF, 'in_model_version_0607_ver21_offline_457_2', True)

# 关闭
spark.stop()

