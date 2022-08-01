### 1.配置gitlab/github等仓库公钥
1.1公钥:

很多代码服务器都是需要认证的，ssh认证是其中的一种。在客户端生成公钥，把生成的公钥添加到代码服务器，后续连接服务器拉取代码时就不用每次都输入用户名和密码了。


1.2生成公钥:

    git bash窗口输入ssh-keygen
    公钥路径可默认，直接回车
    输入密码，不输入可直接回车

1.3查看公钥:

    MAC :  cat ~/.ssh/id_rsa.pub
    Windows : 在git bash窗口输入MAC命令

### 2.常用git操作
Tips: 一定要cd到项目文件下再创建分支

    git clone xxx.git           克隆项目到本地工作区
    git checkout -b <name>      创建并切换到自己分支
    git add .                   提交当前目录下所有文件（把自己负责部分提交到自己分支）
    git commit -m '说明'         添加说明
    git push                    提交代码到服务器
    git pull origin master      同步最新master代码到本地分支

### 3.pip换国内源安装
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==2.2.0

### 4.hadoop shell操作

    # 单个删除、级连删除目录下所有文件
    hadoop fs -rm hdfs:///data/ai/models/liuxuanheng/TFrecord/20210906_prediction/t1.txt
    hadoop fs -rmr hdfs:///data/ai/models/liuxuanheng/TFrecord/20210906_prediction

    # 文件内容显示
    hadoop fs -cat hdfs:///data/ai/models/liuxuanheng/test.csv

    # 显示目录下所有文件
    hadoop fs -ls hdfs:///data/ai/models/online/in 

    # 复制文件到另一目录下       
    hadoop fs -cp hdfs:///data/ai/models/liuxuanheng/non_important_0630_latest_feature_sample.model  hdfs:///data/ai/models/online/in    
    
    # 创建目录
    hadoop fs -mkdir /data/ai/models/jw/20210531

    # 移动文件
    hadoop fs -mv  源 目标

    # 目录下文件个数
    hadoop fs -count /user/hive/warehouse/ai.db/dws   目录下文件总个数

    # 本地jupyter上传文件到hdfs
    hadoop fs -put /tf/liuxuanheng/tf_0910_window_5_retrain.h5 hdfs:///data/ai/models/liuxuanheng/TFrecord/tf_20210910_window_5_retrain.h5
    
    # 加载hdfs文件到hive表
    load data inpath 'hdfs:///data/ai/models/liuxuanheng/ex_table_test/20210808/*.csv' overwrite into table ai.ex_table_test;
    
    # 加载本地文件到hive表
    load data local inpath '/data/apps/modules/jupyter_multi_users/liuxuanheng/jupyter/city_cell_recon/city_cell_order_not_full/*.csv' overwrite into table ai.ex_table_test;

    # 上述两个加载文件方式必须使用 jupyter 终端输入 hive 进入交互模式输入命令才可执行，hue里执行会报错，而且想要批量导入数据，可直接用*.csv匹配所有csv文件, 记得加分号;表示语句结束

    # 批量本地文件传至hdfs脚本, $1/ 表示绝对路径，不加会找不到
    ===
    #!/bin/bash
    
    for file in `ls $1`
        do
            echo $file 
            hadoop fs -put  $1/$file hdfs:///data/ai/models/liuxuanheng/ex_table_test/20210808
        done
    ===
    1. 更改脚本权限为可执行 : chmod u+x bash2.sh    
    2. 终端执行 : ./bash2.sh /data/apps/modules/jupyter_multi_users/liuxuanheng/jupyter/city_cell_recon/cell_folder
    3. 第二步语句表示将本地cell_folder目录下所有文件循环传到hdfs的20210808目录下

### 5.spark yarn操作
    # 彻底关闭任务
    yarn application -kill application_1613868486791_27511

    # 导出日志到本地
    yarn logs -applicationId application_1614586399404_72720 > app.log

    # 导出详细日志到本地
    yarn logs -applicationId application_1614586399404_103993|less
    

### 6.常用包
    import sys
    # sys.path.append('/data/apps/modules/jupyter')
    import os
    import h3
    os.environ["ARROW_PRE_0_15_IPC_FORMAT"]='1'
    import feather
    import time
    import math
    import json
    import pickle
    import datetime
    import collections
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import warnings
    warnings.filterwarnings("ignore”)

    import pyspark.sql.functions as F
    from pyspark.conf import SparkConf
    from pyspark.context import SparkContext
    from pyspark.sql import SparkSession
    from pyspark.sql import Window
    from pyspark.sql import Row
    from pyspark.sql.functions import broadcast, udf, pandas_udf, PandasUDFType
    from pyspark.sql.types import ArrayType, StructField, StructType, StringType, IntegerType, DecimalType, FloatType

### jupyter使用规范

创建用户脚本
```python
useradd -g ai ai.xxx
sudo su - ai.xxx
cd /data/apps/modules/jupyter_multi_users/shell
./init_jupyter_user.sh xxx
```

启动用户jupyter脚本
```python
#用户对应的端口从记录中查找
sudo su - ai.xxx
cd /data/apps/modules/jupyter_multi_users/shell
./start_jupyter_user.sh xxx 8700
```

重启jupyter客户端
```python
#切换到要关停jupyter的用户
sudo su - ai.xxx
#进入到对应用户的jupyter目录
cd /data/apps/modules/jupyter_multi_users/xxx/jupyter
#查看当前用户jupyter所占用的端口
jupyter notebook list
#停止对应得jupyter占用的端口
jupyter notebook stop 8888
或者
#通过端口获得pid
lsof -n -i4TCP:[port-number]
#删除进程
kill -9 [PID]
#重新启动对应用户得jupyter
cd /data/apps/modules/jupyter_multi_users/shell
./start_jupyter_user.sh xxx 8888
```
    