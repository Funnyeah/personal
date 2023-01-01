### 编码风格
[docs](https://docs.python.org/zh-cn/3/)

[code-style](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)

### 工具包
[scipy](https://docs.scipy.org/doc/scipy/reference/)

[scikit-learn](https://scikit-learn.org/stable/)

### 日期获取

    #当前日期时间获取
    time.strftime("%Y-%m-%d %H-%M-%S",time.localtime(time.time()))       
    EVENT_DAY =  datetime.datetime.strftime(datetime.date.today(),"%Y%m%d")

    #根据所给日期向前后偏移
    e = (datetime.datetime.strptime('20220509', "%Y%m%d") - datetime.timedelta(days=1)).strftime("%Y%m%d")

    #字符串转日期格式
    d1 = datetime.datetime.strptime('2022-06-05 20:51:45','%Y-%m-%d %H:%M:%S')
    >> datetime.datetime(2022, 6, 5, 20, 51, 45)
    d2 = d1.date()
    >> datetime.date(2022, 6, 7)
    d3 = d2.strftime('%Y%m%d')
    >> '20220607'
    time.mktime(d1.timetuple())
    >> 1667209327.0  # 秒级时间戳


### 坐标排序

（1）此方法会丢失尖点

    data2 = [(-516, -53), (-516, -53), (-511, -60), (-511, -60), (-511, -60), (-509, -55), (-509, -55), (-512, -59), (-517, -56), (-515, -59), (-515, -59), (-510, -57), (-510, -57), (-514, -52), (-510, -53), (-510, -53), (-517, -58), (-515, -51), (-515, -51), (-515, -51), (-518, -58), (-518, -58), (-510, -56), (-513, -59), (-513, -52), (-513, -52), (-515, -52), (-511, -58), (-514, -59), (-518, -57), (-518, -57), (-511, -53), (-516, -58), (-516, -54), (-509, -54), (-509, -54), (-511, -59), (-512, -53), (-517, -55), (-517, -55)]
    data = list(set(data2))    # 必须去重
    # import matplotlib.pyplot as plt
    # plt.scatter(x=[i[0] for i in data], y=[i[1] for i in data])
    # plt.scatter(x=[i[0] for i in a], y=[i[1] for i in a], color='red')

    nodes_list = [(-518, -58)]
    res = [nodes_list[0]]
    while True:
        if len(nodes_list) == 0:
            break
        node = nodes_list[-1]
        nodes_list.pop()
        x = node[0]
        y = node[1]
        candidate = [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y+1), (x+1, y+1), (x+1, y), (x+1, y-1), (x, y-1)]
        for i in candidate:
            if i in data:
                res.append(i)
                nodes_list.append(i)
                data.remove(i)
                break
    from shapely.geometry import Point, Polygon
    Polygon(res)   # 验证数据顺序

（2）此方法不会丢点

    from math import atan2
    from matplotlib import pyplot as plt

    point_list = [(-516, -53), (-516, -53), (-511, -60), (-511, -60), (-511, -60), (-509, -55), (-509, -55), (-512, -59), (-517, -56), (-515, -59), (-515, -59), (-510, -57), (-510, -57), (-514, -52), (-510, -53), (-510, -53), (-517, -58), (-515, -51), (-515, -51), (-515, -51),
                (-518, -58), (-518, -58), (-510, -56), (-513, -59), (-513, -52), (-513, -52), (-515, -52), (-511, -58), (-514, -59), (-518, -57), (-518, -57), (-511, -53), (-516, -58), (-516, -54), (-509, -54), (-509, -54), (-511, -59), (-512, -53), (-517, -55), (-517, -55)]

    def center(points):
        x, y = list(zip(*points))
        return sum(x)/len(x), sum(y)/len(y)

    # drop duplicate
    point_list = list(set(point_list))

    center_point = center(point_list)
    angles = [atan2(y-center_point[1], x-center_point[0]) for x, y in point_list]    # x,y以求出的中心点作为原点的坐标
    # print(angles)

    sorted_point_list = [i for _, i in sorted(zip(angles, point_list))]

    plt.scatter(*zip(*sorted_point_list))
    for idx, (x, y) in enumerate(sorted_point_list):
        plt.text(x, y, f'{idx}')
    plt.show()

### 多线程/进程统计文件行数

多线程统计文件行数

    import os
    from multiprocessing.dummy import Pool
    def get_hangshu(path):
        with open(path,encoding='utf-8') as f:
            count = len(f.readlines())
            print(f'文件夹{path}共有{count}行内容')
            
    path = [os.path.join('test/',d) for d in os.listdir('test/') if d.endswith('.txt')]
    pool = Pool(4)
    pool.map(get_hangshu,path)

多进程统计文本行数


    from multiprocessing import Pool
    def get_hangshu2(path):
        with open(path,encoding='utf-8') as f:
            count = len(f.readlines())
            print(f'文件夹{path}共有{count}行内容')
            
    path = [os.path.join('test/',d) for d in os.listdir('test/') if d.endswith('.txt')]
    po = Pool(4)

    for p in path: 
        po.apply_async(get_hangshu2,(p,))
    po.close()
    po.join()

### 文件操作

    """
    f.read()、f.write  #读写整个文件 ,read(size)可读取size大小的文件
    f.readline()  # 每次读一行
    f.readlines()、f.writelines（） # readlines按行读取文件，并存入一个字符串列表; writelines将一个字符串列表的形式写入文件
    mode为读取模式，默认为r，即只读模式
    b表示二进制，r表示读，w表示写，a表示追加。无论什么模式，有+则意味着可读可写。写入一般会覆盖原文件，追加则在原文件尾部开始写。如果文件不存在，w, w+, a, a+, wb会创建新文件
    """
    with open(path='data/info.txt' ,mode='r') as f: 
        lines = f.readlines() # 行文本列表 
        for l in lines:
            ls = l.strip()
        f.close()

### http get请求

    import requests
    import json
    url = f'http://10.100.3.240:5000/dispatch/neighbor?requestData=%7B%22debug%22:1,%22city_id%22:635,%22labor_id%22:32746,%22labor_latitude%22:{labor_lat},%22labor_longitude%22:{labor_lon},%22move_num%22:0,%22move_type%22:5,%22trace_id%22:%22d488f67a-c41b-4afd-bc7a-9c5814b55598%22%7D'
    cookies = {'Cookie':'xxxxx'}
    r = requests.get(url, cookies = cookies)

    # 网页内容 r.text 返回的是Unicode格式的数据  content 为二进制数据
    print(r.text)
    print(r.content)

    # 内容返回json格式 
    r.json() # 1
    json_info = json.loads(r.text.strip()) # 2


### 项目导包路径问题

    import sys
    sys.path.append('hx2')
    sys.path.append('../lib')
    from c import getsd
    getsd()

    from b import cc
    cc()

    from lbe import getf
    getf()

    （1）需要导入的文件与当前运行文件处于同一层
    from c import getsd # 直接在运行文件上方写入 from 文件名 import 需要的函数/类

    （2）需要导入的文件的父目录与当前运行文件处于同一层
    import sys
    sys.append('hx2')
    from b import cc # 将需要导入文件的父目录加入sys中，然后直接 from 文件名 import 需要的函数/类

    （3）需要导入的文件的父目录与当前运行文件的父目录处于同一层
    import sys
    sys.path.append('../lib')
    from x import getws # 将需要导入文件的父目录的相对路径加入sys中，然后直接 from 文件名 import 需要的函数/类
    
    （4）其他方法
    2和3也可以在需要导入的文件的所有父目录下添加__init__.py 空文件，然后在运行文件中直接from 需要导入文件的父目录.次目录.次目录.文件名 import 函数

### 模型重要性分析
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import graphviz as gph
    from matplotlib.pylab import rcParams
    import xgboost as xgb
    import matplotlib.pyplot as plt

    model = xgb.Booster(model_file="/data/apps/modules/jupyter_multi_users/liuxuanheng/jupyter/spark-shell/model_pkg/budongche/lxh_1109_model_20211110_1159.model")
    feature_columns =  ['dayofweek',
                'is_weekend',
                'is_holiday',
                'area_distance',
                'is_area_outside',
                'is_near_link',
                'is_not_stopping',
                'poi_cnt',
                'station_cnt',
                'mean_latest_7_stop_bike_cnt',
                'mean_latest_14_stop_bike_cnt',
                'mean_lastest_7_day_bike_consume_time',
                'mean_lastest_14_day_bike_consume_time',     
                'mean_latest_4_week_stop_bike_cnt',
                'mean_latest_4_week_bike_consume_time']
    model.feature_names = feature_columns
    model.feature_types = None
    fig, ax  = plt.subplots(1,1, figsize=(8,16))
    xgb.plot_importance(model,importance_type='weight', ax=ax)

    %matplotlib inline
    rcParams['figure.figsize'] = 30,20
    sns.set(font_scale = 1.5)


### mysql数据读取+spark计算
```python
import pymysql
data=[]
# 打开数据库连接
db = pymysql.connect(host='10.100.45.131',
        port = 3306,
        user='ulb_mozi_read',
        passwd='!L5rY6g#H',
        db ='mozi' )
 
# 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()
# 使用 execute()  方法执行 SQL 查询 
cursor.execute("SELECT * from jw_block_station where block_version_code=21")
data.extend(cursor.fetchall())
df = spark.createDataFrame(data,schema=['self_id','block_id','city_id','station_id','block_version_code','create_time','update_time'])
cursor.close()

```
### pyhive数据读取
```python
from pyhive import hive
import pandas as pd
HOST = "10.100.168.143"  # backup "10.100.164.150"
PORT = 10000
USERNAME = "liuxuanheng"
database='ai'
conn = hive.Connection(host=HOST, port=PORT, username=USERNAME, database=database)

# 查询 pandas处理
def fetch_order():
    query = "SELECT * from ai.dws_etp_test_da"        
    df = pd.read_sql(query, conn)
    return df
fetch_order()

# 增删 加载文件入外部表 
# dws_etp_test_da：city_id,score,event_day(分区)
# test文件格式无列名和行标 
cursor = conn.cursor()
# 1.此处未指定分区，into 和 overwrite into 均表示直接插入，不会覆盖任何分区，hdfs文件不移动到warehouse表名及对应日期分区目录下
cursor.execute('load data inpath "hdfs:///data/ai/models/liuxuanheng/station/test.csv" [overwrite] into table ai.dws_etp_test_da')  

# 2.此处指定分区，overwrite 会将指定分区数据覆盖，into为追加数据，文件会移动到warehouse表名及对应日期分区目录下，文件数据load时会按照表创建的字段分隔方式，如','分割,依次将数据插入表中，test文件中多余的列字段会被丢弃（本例中的event_day）
cursor.execute('load data inpath "hdfs:///data/ai/models/liuxuanheng/station/test.csv"  into table ai.dws_etp_test_da partition(event_day="20220801")' )  
cursor.close()
```

###  查询hdfs指定目录下是否有csv文件

```python
# 1.服务器由多个节点组成hadoop集群，在节点安装jupyter可以通过终端使用hadoops shell命令交互，操作hdfs上文件
# 2.在jupyter上py脚本中，我们想要读取hdfs上数据需要pyhdfs等工具连接，然后操作hdfs目录文件
# 3.可以编写bash shell脚本，在jupyter终端执行
# 4.可以通过os库执行 shell 命令，但是仅可以根据返回值查看是否成功，并不可以获取值等交互行为
import os
sign = os.system('hadoop fs -ls hdfs:///data/ai/models/liuxuanheng/station/test.csv') #执行cmd命令
flag = True if sign==0 else False # 上述返回值一般系统为0表示正确执行，其余返回值为异常
# if flag==True 则load数据入hive写今天分区 else 复制前一天数据写今天分区
```


### 全局配置文件读取
    # transfer.conf 配置文件
    [trans]
    less_bike_rate = 0.40
    order_decay_rate = 1.0
    bike_one_worker = 21
    explore_max_increase_rate = 1
    explore_max_value = 15
    big_city = 415
    explore_citys = 248,520,528,529,531,535,545,559,560,566,64,580,68,582,72,588,589,590,592,597,601,89,603,94,98,613,624,118,635,636,637,125,639,132,135,649,138,144,658,659,151,153,666,155,667,158,160,161,675,164,678,679,684,685,687,178,693,184,696,698,187,186,701,190,192,194,195,198,199,200,717,209,210,211,215,218,731,741,744,745,749,238,751,752,243,246,760,250,764,255,256,259,260,779,268,781,784,272,792,281,797,286,799,291,803,805,295,297,302,305,306,308,313,314,316,833,322,323,837,326,840,330,332,333,847,342,854,857,859,860,351,864,868,871,872,874,876,366,374,375,376,895,896,385,388,390,391,393,404,405,411,413,415,933,940,429,431,432,433,949,439,953,958,453,455,466,468,470,473,479,486,487,492,496,499,505,233,630,355,220,172,556,258,585,82,13,359,444,827,142,437,420,422,453,240,165,436,307,320,464,221

    # 参数解析
    import configparser
    import os
    def get_transfer_config():
        current_dir = os.path.abspath(os.path.dirname(__file__))
        root_dir = os.path.dirname(current_dir)
        cf = configparser.ConfigParser()
        cf.read(root_dir + "transfer.conf")
        return cf
    cf = get_transfer_config()
    less_bike_rate_t = float(cf.get("trans", "less_bike_rate"))
    order_decay_rate = float(cf.get("trans", "order_decay_rate"))
    bike_one_worker = float(cf.get("trans", "bike_one_worker"))


### KDtree
```python
from scipy import spatial
points = [(1,-1),(2,3),(2,-3),(-2,3)]
tree = spatial.KDTree(points)
res = tree.query_ball_point((0,0), 3)
# 返回（0,0）坐标欧式距离为3米的点的列表
```
### 交互式输出和打印输出的区别
    b= 'D:\game\pal4' 
    print(b)     
    返回 D:\game\pal4 # print调用的是经过Python优化，打印的是对象的__str__方法输出，更便于人类阅读的样子。
    print(repr(b)) 
    返回 'D:\\game\\pal4' # print调用repr方法输出就和命令行输出一致了
    b       
    返回 'D:\\game\\pal4' # 交互式命令行调用的是__repr__方法输出，这是字符串在Python里面的真正的样子，所以斜杠会变多

### 字符串传参数
    "io.{2},{1}".format(1,2,3)
    返回 'io.3,2'

    "io,%s,%.2f"%(1,2)
    返回 'io,1,2.00'   

### 字典赋值

    dic = {}
    dic.setdefault(1,[]).append((2,3))  #{1: [(2, 3)]}
    dic.setdefault(1,[]).append((4,5))  #{1: [(2, 3), (4, 5)]}


### 列表元祖->元祖列表
    -- list(tuple(x,y)) ——>  list(tuple(x),tuple(y))

    lis4= [(-516, -53), (-516, -53), (-511, -60), (-511, -60), (-511, -60), (-509, -55), (-509, -55), (-512, -59), (-515, -59), (-515, -59),
    (-510, -57), (-510, -57), (-514, -52), (-510, -53), (-510, -53), (-515, -51), (-515, -51), (-515, -51), (-518, -58), (-510, -56), (-513, -59), (-513, -52), (-513, -52), (-515, -52), (-511, -58), (-514, -59), (-511, -53), (-516, -58), (-516, -54),(-509, -54), (-509, -54), (-511, -59),
    (-512, -53),(-517, -55),(-517, -55)]
    
    方法：list(zip(*lis4))

### 日志时间记录
    import datetime
    class Logger:
        def __init__(self, path):
            self.fp = open(path, 'w')
            
        def info(self, message):
            dateTime = datetime.datetime.now()
            dateTime = datetime.datetime.strftime(dateTime,'%m/%d/%Y %H:%M:%S')
            self.fp.write(dateTime + ' - ' +  message + '\n')
            self.fp.flush()
            
        def close(self):
            self.fp.close()
            
    log = Logger(f"../%s_log.log"%('test_file_name'))
    log.info(f"xx.done")
    log.info(f"xx2.done")
    log.close()

### 类方法
    """
    实例方法，静态方法，类方法
    
    普通方法带self参数，必须实例化调用
    静态方法不带self参数，可直接类名.方法调用，相当于类的专属方法
    类方法带cls参数,可直接类名.方法调用
    静态、类也都可被实例化对象.方法名调用
    抽象类不可被实例化，继承抽象类的子类必须重写其抽象方法（如果父类有的话）
    """
    
    class Kls(object):
        def foo(self, x):
            print('executing foo(%s,%s)' % (self, x))

        @classmethod
        def class_foo(cls,x):
            print('executing class_foo(%s,%s)' % (cls,x))

        @staticmethod
        def static_foo(x):
            print('executing static_foo(%s)' % x)

    ik = Kls()

    # 实例方法
    ik.foo(1)
    print(ik.foo)
    print('==========================================')

    # 类方法
    ik.class_foo(1)
    Kls.class_foo(1)
    print(ik.class_foo)
    print('==========================================')

    # 静态方法
    ik.static_foo(1)
    Kls.static_foo('hi')
    print(ik.static_foo)

    # 对于实例方法，调用时会把实例ik作为第一个参数传递给self参数。因此，调用ik.foo(1)时输出了实例ik的地址。

    # 对于类方法，调用时会把类Kls作为第一个参数传递给cls参数。因此，调用ik.class_foo(1)时输出了Kls类型信息。
    前面提到，可以通过类也可以通过实例来调用类方法，在上述代码中，我们再一次进行了验证。

    # 对于静态方法，调用时并不需要传递类或者实例。其实，静态方法很像我们在类外定义的函数，只不过静态方法可以通过类或者实例来调用而已。

    # 原文链接：https://blog.csdn.net/lihao21/article/details/79762681

    """
    实现抽象类
    从abc模块导入ABC类，和abstractmethod抽象方法装饰器。基于ABC类可以实现一个抽象类。
    通过@abstractmethod装饰一个方法，让它成为一个抽象方法。
    +++抽象方法在子类中必需被实现，抽象类不能被实例化+++
    抽象类是软件开发中一个非常重要的概念，通过定义抽象类，我们可以约定子类必需实现的方法。
    当我们一个类有几十上百个方法时，用抽象方法来防止子类漏掉某些方法是非常方便的做法。
    """
    from abc import ABC,abstractmethod
    class people(ABC):

        @abstractmethod
        def eat(self,):
            return 
        def walk(self):
            return
    # p1=people() # TypeError: Can't instantiate abstract class people with abstract methods eat

    class july(people):
        def eat(self,):  
            return
        def hh(self):
            return
    p2 = july()    #TypeError: Can't instantiate abstract class july with abstract methods eat 

### 某目录下所有文件打包
    import os
    import tarfile

    def recursive_files(dir_name='.', ignore=None):
        for dir_name,subdirs,files in os.walk(dir_name):
            if ignore and os.path.basename(dir_name) in ignore: 
                continue

            for file_name in files:
                if ignore and file_name in ignore:
                    continue

                yield os.path.join(dir_name, file_name)

    def make_tar_file(dir_name='.', tar_file_name='tarfile.tar', ignore=None):
        tar = tarfile.open(tar_file_name, 'w')

        for file_name in recursive_files(dir_name, ignore):
            tar.add(file_name)

        tar.close()

    dir_name = '.'
    tar_file_name = 'archive.tar' # 打包生成的文件
    ignore = {'.ipynb_checkpoints', '__pycache__', tar_file_name} # 需要忽略的文件
    make_tar_file(dir_name, tar_file_name, ignore)

### 获取车站半径26m内的所有13级h3列表
    import geog
    from h3 import h3
    import numpy as np
    from shapely import geometry
    from coord_convert.transform import wgs2gcj, wgs2bd, gcj2wgs, gcj2bd, bd2wgs, bd2gcj

    def get_station_round_block_func(x):
        x_dict = x.asDict()

        station_point = x_dict['station_point']
        lon, lat = bd2wgs(float(station_point.split('|')[0]), float(station_point.split('|')[1]))
        station_h3_index = h3.geo_to_h3(lat, lon, 11)  # 车站对应11级h3
        x_dict['station_block_id'] = station_h3_index

        # 车站半径25m内所有h3 13级索引
        p = geometry.Point([lon, lat]) # 圆
        n_points = 20
        d = 26  # meters
        angles = np.linspace(0, 360, n_points)  #角度分20个
        polygon = geog.propagate(p, angles, d)  #得到圆的近似多边形,这里的p只接受经纬度
        geo_js = geometry.mapping(geometry.Polygon(polygon)) #写成geojson格式
        block_list = h3.polyfill(geo_js, 13, True) # 获取图形内包含的所有13等级h3

        x_dict['block_id_list'] = list(block_list)
        return Row(**x_dict)

### ELO等级分
```python
#定义elo score 等级评分类
class EloScore:
    #初始积分
    ELO_RATING_DEFAULT = 1500

    #定义初始化方法
    def __init__(self,Sa,ratingA=ELO_RATING_DEFAULT,ratingB=ELO_RATING_DEFAULT):
        self.Sa = Sa
        self.ratingA = ratingA
        self.ratingB = ratingB
      
    #定义阈值 k值
    def computeK(self,rating):
        if rating >=2400:
            return 16
        elif rating >= 2100:
            return 24
        else:
            return 36

    #使用公式推算
    def computeScore(self,):
        Eb_S = 1 / (1+pow(10,(self.ratingA-self.ratingB)/400))  #B对A的胜概率
        Ea_S = 1 - Eb_S #A对B的胜概率
        return Ea_S,Eb_S
    
    def balanceK(self,):
        avg_K = (self.computeK(self.ratingA) + self.computeK(self.ratingB))/2
        return avg_K
    
    def main(self,):
        K = self.balanceK()
        Ea_S, Eb_S = self.computeScore()
        
        Sa, Sb = self.Sa , 1 - self.Sa    # 实际胜负 WIN = 1, LOSS = 0, TIE = 0.5
        
        Ra = K * (Sa - Ea_S)
        Rb = K * (Sb - Eb_S)
        return Ra,Rb
if __name__ == "__main__":

    eloscore = EloScore(0.5,1800,1500)
    print(eloscore.main())
```


### 空间地图数据相关

&emsp;&emsp; GIS（Geographic Information System，地理信息系统）是一门综合性学科，结合地理学与地图学以及遥感和计算机科学，已经广泛的应用在不同的领域，也有称GIS为"地理信息服务"（Geographic Information service）。GIS是一种基于计算机的工具，它可以对空间信息进行分析和处理（简而言之，是对地球上存在的现象和发生的事件进行成图和分析）。

&emsp;&emsp; LBS（Location Based Services，基于位置的服务）的核心是位置与地理信息。一个单纯的经纬度坐标只有置于特定的地理信息中，代表为某个地点、标志、方位后，才会被用户认识和理解。用户在通过相关技术获取到位置信息之后，还需要了解所处的地理环境，查询和分析环境信息，从而为用户活动提供信息支持与服务。

#### 坐标系
    1、WGS-84坐标系：地心坐标系，GPS原始坐标体系
    在中国，任何一个地图产品都不允许使用GPS坐标。

    2、GCJ-02 坐标系：国测局坐标，火星坐标系
    1）国测局02年发布的坐标体系，它是一种对经纬度数据的加密算法，即加入随机的偏差。
    2）互联网地图在国内必须至少使用GCJ-02进行首次加密，不允许直接使用WGS-84坐标下的地理数据，同时任何坐标系均不可转换为WGS-84坐标。
    3）是国内最广泛使用的坐标体系，高德、腾讯、Google中国地图都使用它。

    3、BD-09坐标系
    百度中国地图所采用的坐标系，由GCJ-02进行进一步的偏移算法得到。

    4、CGCS2000坐标系：国家大地坐标系
    该坐标系是通过中国GPS 连续运行基准站、 空间大地控制网以及天文大地网与空间地网联合平差建立的地心大地坐标系统。


#### 坐标系转换

```python
下述两个方法相同
# 1
from coord_convert.transform import wgs2gcj, wgs2bd, gcj2wgs, gcj2bd, bd2wgs, bd2gcj

# 2
x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 偏心率平方

def bd09_to_wgs84(bd_lon, bd_lat):
    lon, lat = bd09_to_gcj02(bd_lon, bd_lat)
    return gcj02_to_wgs84(lon, lat)

def bd09_to_gcj02(bd_lon, bd_lat):
    """
    百度坐标系(BD-09)转火星坐标系(GCJ-02)
    百度——>谷歌、高德
    :param bd_lat:百度坐标纬度
    :param bd_lon:百度坐标经度
    :return:转换后的坐标列表形式
    """
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
    gg_lng = z * math.cos(theta)
    gg_lat = z * math.sin(theta)
    return [gg_lng, gg_lat]

def gcj02_to_wgs84(lng, lat):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    if out_of_china(lng, lat):
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]

def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret

def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret

def out_of_china(lng, lat):
    """
    判断是否在国内，不在国内不做偏移
    :param lng:
    :param lat:
    :return:
    """
    return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)

```
#### 距离估算

φ1和φ2表示两点的纬度；
λ1和λ2表示两点的经度。
r为地球半径

$\begin{aligned} dis &=2 r \arcsin \left(\sqrt{\operatorname{hav}\left(\varphi_2-\varphi_1\right)+\cos \left(\varphi_1\right) \cos \left(\varphi_2\right) \operatorname{hav}\left(\lambda_2-\lambda_1\right)}\right) \\ &=2 r \arcsin \left(\sqrt{\sin ^2\left(\frac{\varphi_2-\varphi_1}{2}\right)+\cos \left(\varphi_1\right) \cos \left(\varphi_2\right) \sin ^2\left(\frac{\lambda_2-\lambda_1}{2}\right)}\right) \end{aligned}$


    # 方法一
    b_lon =    109.67249220637174 
    b_lat =    27.449656246000266 
    s_lon =    109.69888390656 
    s_lat =    27.4442168870677 
    dis = 1000 * 6371.393 * np.arccos(
        np.cos(np.radians(s_lat)) * np.cos(np.radians(b_lat)) * np.cos(np.radians(b_lon)-np.radians(s_lon))
        + 
        np.sin(np.radians(b_lat)) * np.sin(np.radians(s_lat))
    ) #米

    # 方法二
    EARTH_REDIUS = 6378.137

    def rad(d):
        return d * math.pi / 180.0

    def getDistance(lat1, lng1, lat2, lng2):
        radLat1 = rad(lat1)
        radLat2 = rad(lat2)
        a = radLat1 - radLat2
        b = rad(lng1) - rad(lng2)
        s = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
        s = s * EARTH_REDIUS
        return s * 1000
    dis = getDistance(b_lat,b_lon,s_lat,s_lon)

    # 方法三
    from geopy.distance import geodesic
    geodesic((30.28708,120.12802999999997), (28.7427,115.86572000000001)).m # .km是千米

    # 方法四

    def haversine(latlon1, latlon2):
    """
    计算两经纬度之间的距离
    """
    if (latlon1 - latlon2).all():
        lat1, lon1 = latlon1
        lat2, lon2 = latlon2
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6370996.81  # 地球半径m
        distance = c * r
    else:
        distance = 0
    return distance

    




#### 底层数据存储

1.地理空间数据结构复杂，它们的存储关系到 GIS 数据交换、显示、查询、分析的能力。

2.GIS 的数据模型有矢量、栅格；矢量模型数据用点、线、面来描述地理实体，两点成线，三线可成面，线和面在计算机存储时其实记录的还是点的坐标。

3.矢量模型常用的数据格式有Shapefile、KML、dwg、dxf 等；栅格数据用二维矩阵的位图来表示空间地物，常见的格式有TIFF、JPEG、BMP、PCX、GIF等。

4.矢量和栅格各有优缺点：比如矢量图与分辨率无关，就像你在手机地图中无论把地图放大到多大，都不影响显示的质量和效果，而栅格数据放大几倍后，就会明显地出现“马赛克”的现象；但矢量数据结构复杂，现势性差，而栅格数据可以通过卫星拍摄快速获取，等等。GIS 应该根据使用场景，来确定使用矢量模型还是栅格模型。

5.当数据达到一定规模后，文件存储方式已经不能满足需求，按照传统的解决方式，自然想到的是创建数据库啊！地理数据包含非结构化的空间数据、结构化的属性数据、空间关系数据，传统的关系型数据库无法提供存储、管理、索引、查询等常规的数据库功能，所以空间数据库应用而生，现在常见的空间数据库有GeoDatabase，PostgreSQL，Oracle Spatial等。

#### 数据制图

&emsp;&emsp; 因为地球是个三维近椭球体，而地图是个二维平面，如何将球面上地物的相对位置，准确的在二维平面上表示，就需要针对实际应用场景采用合适的坐标系统（Beijing54、Xian80、WGS84等）和地图投影（高斯克吕格、墨卡托等）

#### 空间索引

    1.KD树空间索引（二叉树索引）、KDB树索引
    2.R树、R+树空间索引
    3.G树索引
    4.四叉树索引及其分类（点四叉树索引、MX四叉树索引、PR四叉树索引、CIF四叉树索引、基于固定网格划分的四叉树索引）
    5.CELL树索引
    6.BSP树空间索引

#### 常用工具
- 1.[Shapely-doc](https://shapely.readthedocs.io/en/latest/manual.html#introduction) [Shapely-示例](https://www.osgeo.cn/pygis/shapely-geometry.html?highlight=polygon)

- 2.[folium-doc](https://python-visualization.github.io/folium/quickstart.html#Markers) [folium-示例](https://blog.csdn.net/weixin_38169413/article/details/104806257?spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-8-104806257-blog-110427178.topnsimilarv1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EOPENSEARCH%7ERate-8-104806257-blog-110427178.topnsimilarv1&utm_relevant_index=9) [folium-示例](https://zhangphil.blog.csdn.net/article/details/110414544?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-110414544-blog-110427178.topnsimilarv1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-110414544-blog-110427178.topnsimilarv1&utm_relevant_index=3) [folium-进阶](https://zhuanlan.zhihu.com/p/502601498) [folium-git](https://github.com/HuStanding/show_geo/blob/master/Folium%20Visualization%20Examples.ipynb)

- 3.[geo_json简明](https://zhuanlan.zhihu.com/p/539689986)
- 4.[geopandas-doc](https://geopandas.org/en/stable/gallery/polygon_plotting_with_folium.html)
- 5.[h3-doc](https://h3geo.org/docs/3.x/api/indexing)
- 6.[Sedona](https://sedona.apache.org/api/sql/Overview/)


### python对象序列化

- Python pickle模块是对二进制协议的一种实现，用于对于python中的对象结构进行（正向的）序列化(serialization)和（反向的）解序列化(de-serialization)处理。
- 序列化(serialization)将结构化的python对象（如list, dict等）转化为字节流(byte stream)，通常也称为pickling，flattening,或者marshalling. 
- 解序列化处理即将由pickling处理生成的字节流反向变换回原来的python对象，也称为unpickling.
- 为什么需要序列化处理呢？ 一言以蔽之，方便数据在不同的系统等之间进行传输、以及以文件或者数据库的方式进行存储

```python
# 以模型保存加载为例

# 方法一
import joblib
# 保存
joblib.dump(value=uplift_model, filename='uplift_model_v1.m')
# 加载
model = joblib.load('uplift_model_v1.m')
# 预测
model.predict()

# 方法二
import pickle
# 保存 由于转换成了字节序列，所以只能以二进制的格式存入文件，所以open()命令的选项为'wb'，其中b表示binary
with open('model_file.pkl',mode='wb') as f:
    pickle.dump(uplift_model,f) # 将模型uplift_model序列化后存入文件f中
# 加载
with open('model_file.pkl', 'rb') as fh:
    model = pickle.load(fh) # 解序列化文件中内容，恢复出原python对象层次结构
# 预测
model.predict()


# dumps()这个函数与dump()的区别在于它将序列化后的数据存储在一个字节流对象，而非文件中
# loads()与dumps()构成一对,直接从pickle序列化后的字节流对象恢复出原始的结构层次
data1 = [ { 'a':'A', 'b':2, 'c':3.0 } ]
print ('BEFORE:',)
pprint.pprint(data1)
 
data1_string = pickle.dumps(data1)
 
data2 = pickle.loads(data1_string)
print ('AFTER:',)
pprint.pprint(data2)
 
print ('SAME?:', (data1 is data2))
print ('EQUAL?:', (data1 == data2))

```