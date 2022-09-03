### Spark应用

&emsp;&emsp;Spark应用（Application）是用户提交的应用程序。执行模式又Local、Standalone、YARN、Mesos。根据Spark Application的Driver Program是否在集群中运行，Spark应用的运行方式又可以分为Cluster模式和Client模式。
下面是Spark应用涉及的一些基本概念：

    Application：Spark 的应用程序，用户提交后，Spark为App分配资源，将程序转换并执行，其中Application包含一个Driver program和若干Executor
    SparkContext：Spark 应用程序的入口，负责调度各个运算资源，协调各个 Worker
    Node 上的 Executor
    Driver Program：运行Application的main()函数并且创建SparkContext
    RDD Graph：RDD是Spark的核心结构， 可以通过一系列算子进行操作（主要有Transformation和Action操作）。当RDD遇到Action算子时，将之前的所有算子形成一个有向无环图（DAG）。再在Spark中转化为Job，提交到集群执行。一个App中可以包含多个Job
    Executor：是为Application运行在Worker node上的一个进程，该进程负责运行Task，并且负责将数据存在内存或者磁盘上。每个Application都会申请各自的Executor来处理任务
     Worker Node：集群中任何可以运行Application代码的节点，运行一个或多个Executor进程
    
下面介绍Spark Application运行过程中各个组件的概念：

    Job：一个RDD Graph触发的作业，往往由Spark Action算子触发，在SparkContext中通过runJob方法向Spark提交Job
    Stage：每个Job会根据RDD的宽依赖关系被切分很多Stage， 每个Stage中包含一组相同的Task， 这一组Task也叫TaskSet
    Task：一个分区对应一个Task，Task执行RDD中对应Stage中包含的算子。Task被封装好后放入Executor的线程池中执行
    DAGScheduler：根据Job构建基于Stage的DAG，并提交Stage给TaskScheduler
    TaskScheduler：将Taskset提交给Worker node集群运行并返回结果


### Spark 任务提交参数

  spark-submit --master yarn --driver-memory 10G --executor-memory 20G --conf spark.kryoserializer.buffer.max=2000M --conf spark.driver.maxResultSize=4G --executor-cores 3 --num-executors 40 --conf spark.speculation=true  --conf spark.dynamicAllocation.enabled=false --conf spark.yarn.executor.memoryOverhead=4G --properties-file /etc/spark/conf/spark-defaults.conf --queue root.ai.offline --archives hdfs:///data/ai/anaconda3.zip\#anaconda3 --conf spark.dynamicAllocation.enabled=false --conf spark.pyspark.driver.python=/data/apps/modules/anaconda3/envs/python3.6/bin/python3 --conf spark.pyspark.python=./anaconda3/anaconda3/envs/python3.6/bin/python3 efficient_t+1.py

 spark-submit --master yarn **--queue "offline" --driver-memory 10G --executor-memory 15G --num-executors 24 --executor-cores 3** --properties-file /etc/spark/conf/spark-defaults.conf --archives hdfs:///data/ai/anaconda3.zip\#anaconda3 --conf spark.sql.shuffle.partitions=300 --conf spark.pyspark.driver.python=python3 --conf spark.driver.maxResultSize=10G --conf spark.pyspark.python=./anaconda3/anaconda3/envs/python3.6/bin/python3 --jars hdfs:///data/ai/lib/xgb/0.90/xgboost4j-0.90.jar,hdfs:///data/ai/lib/xgb/0.90/xgboost4j-spark-0.90.jar --py-files=hdfs:///data/ai/lib/xgb/0.90/sparkxgb.zip --conf spark.dynamicAllocation.enabled=false eff.py

### Spark 参数配置
    设置驱动进程的内存
    一般默认即可，如果程序中使用collect算子拉取rdd到驱动节点上，那就需要设置相应的内存大小（大于几十k建议使用广播变量）
    --driver-memory 10G

    设置每个执行器进程的内存，一般为4～8G
    --executor-memory 20G

    设置执行spark作业的执行器进程个数，一般为50～100个
    --num-executors 40 

    设置执行器进程的cpu核数量，决定了并发执行task线程的个数
    --executor-cores 3
    
    动态分区设置
    --conf spark.hadoop.hive.exec.dynamic.partition.mode=nonstrict --conf spark.hadoop.hive.exec.dynamic.partition=true

    关闭动态分配资源
    --conf spark.dynamicAllocation.enabled=false

    增加堆外内存 
    error：Container killed by YARN for exceeding memory limits. 10.4 GB of 10.4 GB physical memory used
    --conf spark.yarn.executor.memoryOverhead=4G

    合并大量中间文件
    --conf spark.shuffle.consolidateFiles=true 

    防止gc暂停作业 
    --conf spark.storage.blockManagerTimeoutIntervalMs=100000

    任务并行度，每个stage默认任务数量200
    --conf spark.default.parallelism=400

    每个executor执行者对driver驱动程序的心跳间隔。心跳让驱动程序知道执行者仍然活着,默认10s
    --conf spark.executor.heartbeatInterval=60s
    
    开启task预测执行机制，出现执行较慢任务时，会在另一节点尝试执行该任务的副本
    –conf spark.speculation=true

    spark df 与 pandas df 相互转化性能优化  
    config("spark.sql.execution.arrow.enabled”,'true')

    开启笛卡尔积
    spark.conf.set("spark.sql.crossJoin.enabled", "true”) 

    调大端口重试次数
    set('spark.port.maxRetries' , 50)

    自动广播join，默认10m以下的表缓存到内存中
    原理：将小表聚合到driver端，再广播到各个大表分区中，那么再次进行join的时候，就相当于大表的各自分区的数据与小表进行本地join，从而规避了shuffle，此时driver内存不能太小
    set("spark.sql.autoBroadcastJoinThreshold","10m")

    禁用自动广播join
    set("spark.sql.autoBroadcastJoinThreshold","-1")


### Spark MLlib Pipelines
MLlib中的Pipeline主要受scikit-learn项目的启发，旨在更容易地将多个算法组合成单个管道或工作流，向用户提供基于DataFrame的更高层次的API库，以更方便地构建复杂的机器学习工作流式应用。一个Pipeline可以集成多个任务，如特征变换、模型训练、参数设置等。下面介绍几个重要的概念。

* DataFrame：相比于RDD，DataFrame还包含schema信息，可以将其近似看作数据库中的表。
* Transformer：Transformer可以看作将一个DataFrame转换成另一个DataFrame的算法。例如，模型即可看作一个Transformer，它将预测集的DataFrame转换成了预测结果的DataFrame。
* Estimator：一种可以适应DataFrame来生成Transformer的算法，操作于DataFrame数据并生成一个Transformer。
* Pipeline：可以连接多个Transformer和Estimator形成机器学习的工作流。
* Parameter：设置Transformer和Estimator的参数。
Pipeline是多个阶段形成的一个序列，每个阶段都是一个Transformer或者Estimator。这些阶段按顺序执行，当数据通过DataFrame输入Pipeline中时，数据在每个阶段按相应规则进行转换。在Transformer阶段，对DataFrame调用transform（）方法。在Estimator阶段，对DataFrame调用fit（）方法产生一个Transformer，然后调用该Transformer的transform（）。
示例 

```python
from sparkxgb import XGBoostRegressor
xgb_model = XGBoostRegressor(
    featuresCol="features", labelCol='label', predictionCol="prediction",  # weightCol="weight",
    numRound=100,
    eta=0.1, 
    maxDepth=6,
    seed=123, 
    evalMetric="rmse",
    objective="reg:squarederror",
    trainTestRatio=0.95,
    subsample=0.9,
    colsampleBytree=0.9,
    missing = 0.0
)

vector_assembler = VectorAssembler().setInputCols(period_feature + trend_feature + context_feature).setOutputCol("features")
xgb_pipeline = Pipeline().setStages([vector_assembler, xgb_model])

grid = ParamGridBuilder().addGrid(xgb_model.gamma, [0.09, 0.1]).addGrid(xgb_model.eta, [0.09, 0.1, 0.11]).addGrid(xgb_model.maxDepth, [6, 7]).addGrid(xgb_model.subsample, [0.8, 0.9, 1]).build()

evaluator = RegressionEvaluator()
tvs = TrainValidationSplit(estimator=xgb_pipeline, estimatorParamMaps=grid, evaluator=evaluator,
    parallelism=1, seed=42).setTrainRatio(0.95)

model_tmp = tvs.fit(train_data)
model = model_tmp.bestModel             #得到最佳模型
model.extractParamMap().items()            #得到最佳模型的参数
print({param[0].name: param[1] for param in model.extractParamMap().items()})   #打印参数
```

[pyspark](https://sparkbyexamples.com/pyspark/pyspark-sql-expr-expression-function/)

### Spark Dataframe
```python
删除字段
df = df.drop('age')

删除null值
df = df.na.drop()  # 扔掉任何列包含na的行
df = df.dropna(subset=['col_name1', 'col_name2'])  # 扔掉col1或col2中任一一列包含na的行

更改字段类型
df = df.withColumn('age',df.age.cast('string'))  

指定列填充缺失值
df = df.na.fill({ 'age' : 50, 'name' : 'x'}) 

filter函数，此处将col_a列大于0的数据筛选出来
df = df.filter(F.col('col_a')>0)

将旧字段改为新字段名(parms旧，parms新)
df = df.withColumnRenamed('event_day','bike_event_day')

改名方法2
df = df.select(['event_day','bike_event_day']).toDF('ed','bed')

expr函数可使用hive语法，此处为将日期date格式转为字符串
df = df.withColumn('st_event_day',F.expr("FROM_UNIXTIME(UNIX_TIMESTAMP(cast(to_date(start_time) as string),'yyyy-mm-dd'),'yyyymmdd')"))

when函数，如果条件为真，则赋某值，否则赋另一个值
df = df.withColumn('profit_start',F.when(F.col('consumption')==0.5,F.col('span_start')).otherwise(F.col('wait_start_time')))

将一行展开为多行  将score按照 ',' 分割，然后对分割后的数组每个元素都 explode 为一行
df = df.withColumn('score', F.explode(F.split(df.score, ','))).show()     

聚合相同字段data下的number列，返回列表array<int>
df.groupBy(['data']).agg(F.collect_list('number').alias('newcol')).show() 

创建键值对,并聚合返回array<map<string,bigint>>
df = df.withColumn('key',F.create_map(["bike_sn", "bike_24_cnt"]))
df = df.groupBy(['city_id','station_id','event_day','span_id']).agg(F.collect_list('key').alias('key_list')) 

同上，用于合并多列字段值不同的数据
df.groupBy("d").agg(*[collect_set(col) for col in ['s','f']]).show()   

对某列求和   转rdd，取出需要求和的列的下标（此处为3），应用reduce求和
df.rdd.map(lambda x:x[3]).reduce(lambda x,y:x+y)   

取出前n行数据返回dataframe格式数据
df.orderby(['age']).limit(100)

df.head(100) 取出前n行数据返回list格式数据 如下所示取出具体的数据
df.orderBy(['month_in_num'],ascending=False).head(2276)[-1][-1]    

可以指定取出的列 
df.head(5)[3]['city_id']表示返回长度为5的列表中的第四个Row数据的city_id字段值

udf函数用法，输入多字段计算，返回一个字段，可指定返回类型
@udf(returnType=ArrayType(IntegerType()))
def func(col_a,col_b)
    res = 各种操作
    return res
df = df.withColumn('xxx',func(F.col('col_a'),F.col('col_b')))

pandas_udf函数用法之一,分组计算,每组数据在函数内部都是一个pandas dataframe，返回字段需一一对应
@pandas_udf("city_id int,station_id int", PandasUDFType.GROUPED_MAP)
def get_target(df):
    pass
    return df[['city_id','station_id']]
end = df.groupby(['city_id','scene_flag','span']).apply(get_target)  

操作多列，返回多列
def get_move_in(x):
    x_dict = x.asDict() # 每行数据转为python字典
    x_dict['col_a'])  #多列操作
    new_x = Row(**x_dict)
    return new_x
schema_move_in = StructType([
        StructField("city_id", IntegerType(), True),
        StructField("span_id", IntegerType(), True),
        StructField("station_id", IntegerType(), True),
        StructField("station_point", StringType(), True),
        StructField("event_day", StringType(), True),
        StructField("city_center_lat", StringType(), True),
        StructField("city_center_lon", StringType(), True),
        StructField("end_point_list", ArrayType(StringType()), True),
        StructField("move_in", IntegerType(), True)
    ])
df_rdd = df.rdd.map(get_move_in) 
df = spark.createDataFrame(df_rdd,schema=schema_move_in) 
# schema 确定数据类型，若有字段有空值，不定义好schema就会报错

```

### 数据倾斜问题

最常见的场景：聚合函数groupby(‘city_id’).agg(F.count(df.col_a)),某个key对应的数据量很大的就会造成倾斜

后果及影响：1.OOM（单或少数的节点）；2.拖慢整个Job执行时间（其他已经完成的节点都在等这个还在做的节点）

本质：Shuffle时，需将各节点的相同key的数据拉取到某节点上的一个task来处理，若某个key对应的数据量很大就会发生数据倾斜。比方说大部分key对应10条数据，某key对应10万条，被分配10条数据很快做完，个别task分配10万条数据，不仅运行时间长，且整个stage的作业时间由最慢的task决定。

数据倾斜只会发生在Shuffle过程，以下算法可能触发Shuffle操作：

    distinct：
    distinct的操作其实是把原RDD进行map操作,根据原来的key-value生成为key,value使用null来替换,并对新生成的RDD执行reduceByKey的操作,也就是说,Distinct的操作是根据key与value一起计算不重复的结果.只有两个记录中key与value都不重复才算是不重复的数据。
    
    groupByKey：
    groupByKey会将RDD[key,value] 按照相同的key进行分组，形成RDD[key,Iterable[value]]的形式， 有点类似于sql中的groupby，例如类似于mysql中的group_concat
    
    reduceByKey：
    reduceByKey，就是将key相同的键值对，按照Function进行计算。如代码中就是将key相同的各value进行累加。得到的结果就是类似于[(key2,2), (key3,1), (key1,2)] 形式。
   
    aggregateByKey 函数：
    对PairRDD中相同的Key值进行聚合操作，在聚合过程中同样使用了一个中立的初始值。和aggregate函数类似，aggregateByKey返回值的类型不需要和RDD中value的类型一致。因为aggregateByKey是对相同Key中的值进行聚合操作，所以aggregateByKey'函数最终返回的类型还是PairRDD，对应的结果是Key和聚合后的值，而aggregate函数直接返回的是非RDD的结果。
    
    join:
    join类似于SQL的inner join操作，返回结果是前面和后面集合中配对成功的，过滤掉关联不上的。
    
    cogroup:
    对两个RDD中的kv元素，每个RDD中相同key中的元素分别聚合成一个集合。与reduceByKey不同的是针对两个RDD中相同的key的元素进行合并。
    
    repartition:返回一个恰好有numPartitions个分区的RDD，可以增加或者减少此RDD的并行度。内部，这将使用shuffle重新分布数据，如果你减少分区数，考虑使用coalesce，这样可以避免执行shuffle

解决方法：

1.过滤掉sample算子采样后数据量最多的key；

2.提高Shuffle操作并行度 spark.sql.shuffle.partitions=400

3.两阶段聚合（局部聚合+全局聚合）

场景：对RDD进行reduceByKey等聚合类shuffle算子，SparkSQL的groupBy做分组聚合这两种情况

思路：首先通过map给每个key打上n以内的随机数的前缀并进行局部聚合，即(hello, 1) (hello, 1) (hello, 1) (hello, 1)变为(1_hello, 1) (1_hello, 1) (2_hello, 1)，并进行reduceByKey的局部聚合，然后再次map将key的前缀随机数去掉再次进行全局聚合；

原理：对原本相同的key进行随机数附加，变成不同key，让原本一个task处理的数据分摊到多个task做局部聚合，规避单task数据过量。之后再去随机前缀进行全局聚合；

优点：效果非常好（对聚合类Shuffle操作的倾斜问题）；

缺点：范围窄（仅适用于聚合类的Shuffle操作，join类的Shuffle还需其它方案）

```python
from pyspark import SparkContext, SQLContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as fn
from pyspark.sql.functions import udf
import random
spark = open_spark_session(app_name='validate_online_lxh')
tmpdict = [
    {'Col1': 'A', 'Col2': 1},
    {'Col1': 'A', 'Col2': 1},
    {'Col1': 'A', 'Col2': 1},
    {'Col1': 'A', 'Col2': 1},
    {'Col1': 'A', 'Col2': 1},
    {'Col1': 'A', 'Col2': 1},
    {'Col1': 'A', 'Col2': 1},
    {'Col1': 'A', 'Col2': 1},
    {'Col1': 'B', 'Col2': 1},
    {'Col1': 'B', 'Col2': 1},
    {'Col1': 'B', 'Col2': 1},
    {'Col1': 'A', 'Col2': 1},
    {'Col1': 'A', 'Col2': 1},
    {'Col1': 'A', 'Col2': 1}
]
df = spark.createDataFrame(tmpdict)
@udf
def adSalt(adid):
    salt = random.sample(range(0, 4), 1)[0]
    salt = str(salt) + '_'
    return salt + adid
def row_dealWith(data):
    Col1, Col2 = data[0], data[1]
    
    tups = (
        str(Col1),
        int(sum(Col2))
    )
    return tups    
df = df.withColumn('Col1', adSalt(fn.col('Col1')))
df = df.groupBy('Col1').agg(fn.collect_list('Col2').alias('Col2')).rdd.map(row_dealWith).toDF(schema=['Col1', 'Col2'])
getidUDF = fn.udf(lambda x: x.split('_')[1])
df = df.withColumn('Col1', getidUDF(fn.col('Col1')))
df = df.groupBy('Col1').agg(fn.collect_list('Col2').alias('Col2')).rdd.map(row_dealWith).toDF(schema=['Col1', 'Col2'])
```


### 调参心得

由于我们在执行Spark任务是，读取所需要的原数据，数据量太大，导致在Worker上面分配的任务执行数据时所需要的内存不够，直接导致内存溢出了，所以我们有必要增加Worker上面的内存来满足程序运行需要。 在Spark Streaming或者其他spark任务中，会遇到在Spark中常见的问题，典型如Executor Lost相关的问题(shuffle fetch失败，Task失败重试等)。这就意味着发生了内存不足或者数据倾斜的问题。这个目前需要考虑如下几个点以获得解决方案：

    A.相同资源下，增加partition数可以减少内存问题。 原因如下：通过增加partition数，每个task要处理的数据少了，同一时间内，所有正在运行的task要处理的数量少了很多，所有Executor占用的内存也变小了。这可以缓解数据倾斜以及内存不足的压力。 

    B.关注shuffle read阶段的并行数。例如reduce, group 之类的函数，其实他们都有第二个参数，并行度(partition数)，只是大家一般都不设置。不过出了问题再设置一下，也不错。 

    C.给一个Executor核数设置的太多，也就意味着同一时刻，在该Executor的内存压力会更大，GC也会更频繁。我一般会控制在3个左右。然后通过提高Executor数量来保持资源的总量不变。


### 常用包
    import pyspark.sql.functions as F
    from pyspark.conf import SparkConf
    from pyspark.context import SparkContext
    from pyspark.sql import SparkSession
    from pyspark.sql import Window
    from pyspark.sql import Row
    from pyspark.sql.functions import broadcast, udf, pandas_udf, PandasUDFType
    from pyspark.sql.types import ArrayType, StructField, StructType, StringType, IntegerType, DecimalType, FloatType

### 快速demo
    import datetime
    import warnings
    from pyspark.sql import functions as F
    from pyspark.conf import SparkConf
    from pyspark.sql import SparkSession
    warnings.filterwarnings("ignore")
    from pyspark.sql.types import LongType, IntegerType,Row, StructType, StructField, StringType, TimestampType, FloatType

    def open_spark_session(app_name="ai-train"):
        conf = (SparkConf().setMaster("local").setAppName(app_name).set("spark.yarn.queue", "offline").set(
            "spark.sql.crossJoin.enabled", "true").set("hive.exec.dynamic.partition.mode", "nonstrict"))
        spark = SparkSession.builder.config(conf=conf).getOrCreate()
        return spark

    spark = open_spark_session(app_name='dws_ai_dispatch_city_span_da')
    spark.stop()
