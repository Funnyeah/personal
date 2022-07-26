### 编码风格

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


### 坐标系转换

```python
# 坐标系转换
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