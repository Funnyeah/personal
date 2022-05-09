### 日期获取

    #当前日期时间获取
    time.strftime("%Y-%m-%d %H-%M-%S",time.localtime(time.time()))       
    EVENT_DAY =  datetime.datetime.strftime(datetime.date.today(),"%Y%m%d")

    #根据所给日期向前后偏移
    e = (datetime.datetime.strptime(EVENT_DAY, "%Y%m%d") - datetime.timedelta(days=1)).strftime("%Y%m%d")

### 字典赋值

    dic = {}
    dic.setdefault(1,[]).append((2,3))  #{1: [(2, 3)]}
    dic.setdefault(1,[]).append((4,5))  #{1: [(2, 3), (4, 5)]}


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

### list(tuple(x,y)) ——>  list(tuple(x),tuple(y))

    lis4= [(-516, -53), (-516, -53), (-511, -60), (-511, -60), (-511, -60), (-509, -55), (-509, -55), (-512, -59), (-515, -59), (-515, -59),
    (-510, -57), (-510, -57), (-514, -52), (-510, -53), (-510, -53), (-515, -51), (-515, -51), (-515, -51), (-518, -58), (-510, -56), (-513, -59), (-513, -52), (-513, -52), (-515, -52), (-511, -58), (-514, -59), (-511, -53), (-516, -58), (-516, -54),(-509, -54), (-509, -54), (-511, -59),
    (-512, -53),(-517, -55),(-517, -55)]
    
    方法：list(zip(*lis4))


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