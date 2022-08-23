## ETP 离线hive
### 1.城市时点需求

|  字段   | 类型  | 含义  |  备注 | 
|  ----      | ----    |  ----  | ----  | 
| city_id  |   int      | 城市id  |   | |
| span | string | 时间段"20-30"，代表10~15点  |  根据近期城市车效划分 |
|   max_demand | int | 城市时点需求上限  |  近一个月每日各时点out量均值 |
|    usable_bike_cnt       |  int    | 城市可用车辆数  |   |
|   bike_usage        |   float   | 城市车效  |   |
|    scene_flag       |   int   | 场景分区 1-工作日，2-周末，3-大型节假日  |  |
|   event_day        |   string   | 日期分区  |   |
```sql
create table IF NOT EXISTS ai.dws_ai_dispatch_city_span_demand_da(
city_id int COMMENT '城市id',
span string COMMENT '时间区间',
max_demand int COMMENT '城市时点需求上限',
usable_bike_cnt int COMMENT '城市可用车辆数',
bike_usage float COMMENT '城市车效',
scene_flag int COMMENT ' 场景分区 1-工作日，2-周末，3-大型节假日'
)
comment '城市时点&需求表'
partitioned by (event_day string COMMENT '日期分区')
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\001' STORED AS TEXTFILE;
```

### 2.车站时空信息

|  字段   | 类型  | 含义  |  备注 | 
|  ----      | ----    |  ----  | ----  | 
| city_id  |   int      | 城市id  |   | |
| station_id  |   int      | 车站id  |   | |
| block_id  |   string      | 区块id  |   | |
| span | string | 时间段"20-30"，代表10~15点  |  |
|   max_capacity | int | 车站在span时间段内车辆上限  |  近一个月out90分位数 |
|   min_capacity | int | 车站在span时间段内车辆下限  |  近一个月out15分位数 |
|    predict_in_bikes |  int    | 车站在span时间段内预测骑入量  | 7d、14d、30d in加权 |
|    scene_flag       |   int   | 场景分区 1-工作日，2-周末，3-大型节假日  |  |
|   event_day        |   string   | 日期分区  |   |
```sql
create table IF NOT EXISTS ai.dws_ai_dispatch_station_st_info_da(
city_id int COMMENT '城市id',
station_id int COMMENT '车站id',
block_id BIGINT COMMENT '车站所在区块id:1版自挖掘区块，2版11等级h3',
span string COMMENT ' 时间区间',
max_capacity int COMMENT '车站在span时间段内车辆上限',
min_capacity int COMMENT '车站在span时间段内车辆下限',
predict_in_bikes int COMMENT '车站在span时间段内预测骑入量',
scene_flag int COMMENT '场景分区 1-工作日，2-周末，3-大型节假日 '
)
comment '车站时空信息表'
partitioned by (event_day string COMMENT '日期分区')
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\001' STORED AS TEXTFILE;
```
### 3.车站时空效率
一期车站收益用效率的对数乘以车站目标量预估
二期车站收益采用车展24h收益预估

|  字段   | 类型  | 含义  |  备注 | 
|  ----      | ----    |  ----  | ----  | 
| city_id  |   int      | 城市id  |   | |
| station_id  |   int      | 车站id  |   | |
| block_id  |   string      | 区块id  |   | |
| span | string | 时间段"20-30"，代表10~15点  |  |
|   efficiency | float | 车站时空效率  | 时空效率 =  订单/ 等待时长  过去30天平均，数量不同使用威尔逊区间平滑  |
|   profit_order | float | 收益  |  近一月车站时段内流出车辆24h平均订单 |
|    scene_flag       |   int   | 场景分区 1-工作日，2-周末，3-大型节假日  |  |
|   event_day        |   string   | 日期分区  |   |

```sql
create table IF NOT EXISTS ai.dws_ai_dispatch_station_st_efficiency_da_v3(
city_id int COMMENT '城市id',
station_id bigint COMMENT '车站id',
block_id string COMMENT 'h3索引',
span string COMMENT '时间段"20-30"，代表10~15点',
profit_order float COMMENT '截止到当天订单',
efficiency float COMMENT '车站时空效率',
scene_flag int COMMENT ' 场景分区 1-工作日，2-周末，3-大型节假日'
)
comment '车站时空效率v3'
partitioned by (event_day string COMMENT '日期分区')
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\001' STORED AS TEXTFILE;
```
### 4.时空最优车辆

|  字段   | 类型  | 含义  |  备注 | 
|  ----      | ----    |  ----  | ----  | 
| city_id  |   int      | 城市id  |   | |
| station_id  |   int      | 车站id  |   | |
| block_id  |   string      | 区块id  |   | |
| span | string | 时间段"20-30"，代表10~15点  |  |
|   target_bike_cnt | int | 时空目标车辆  | 	1.效率值*时段最大容量    排序问题     （简化为贪心）；2.top N 的车站分配的车辆数 < 当前时段+下一时段动车量    （限制挪车数量）  |
|   explore_target_bike_cnt | int | 探索策略下时空目标车辆  |  50%概率增加三辆车 |
|    scene_flag       |   int   | 场景分区 1-工作日，2-周末，3-大型节假日  |  |
|   event_day        |   string   | 日期分区  |   |
```sql
create table IF NOT EXISTS ai.dws_ai_dispatch_station_st_target_bike_da_v2(
city_id int COMMENT '城市id',
station_id bigint COMMENT '车站id',
block_id string COMMENT 'h3索引',
span string COMMENT '时间段"20-30"，代表10~15点',
target_bike_cnt int COMMENT '时空目标车辆',
scene_flag int COMMENT ' 场景分区 1-工作日，2-周末，3-大型节假日',
explore_target_bike_cnt int COMMENT '探索策略下时空目标车辆 '
)
comment '时空最优车辆v2'
partitioned by (event_day string COMMENT '日期分区')
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\001' STORED AS TEXTFILE;
```
### 5.区块机会成本
一期用的是车辆使用率估计成本
二期用预估机会成本使用30天24h收益平均，评估机会成本使用同价值区块AB分层估计成本

|  字段   | 类型  | 含义  |  备注 | 
|  ----      | ----    |  ----  | ----  | 
| city_id  |   int      | 城市id  |   | |
| block_id  |   string      | 区块id  |   | |
| opportunity_cost | float | 区块机会成本:当天产生订单车辆数/当天凌晨快照时存量车辆数 | 近30天统计,场景区分  |
| city_cost | float | 城市机会成本:近30天统计,场景区分  | 所有区块机会成本平均 |
|    scene_flag       |   int   | 场景分区 1-工作日，2-周末，3-大型节假日  |  |
|   event_day        |   string   | 日期分区  |   |
```sql
create table IF NOT EXISTS ai.dws_ai_dispatch_block_opportunity_cost_online_da(
city_id int COMMENT '城市id',
block_id bigint COMMENT '区块id',
opportunity_cost float COMMENT '区块机会成本:近30天统计,场景区分',
city_cost float COMMENT '城市机会成本:近30天统计,场景区分',
scene_flag int COMMENT '场景分区 1-工作日，2-周末，3-大型节假日'
)
comment '区块机会成本线上表'
partitioned by (event_day string COMMENT '日期分区')
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\001' STORED AS TEXTFILE;
```
### 6.区块到车站距离成本
|  字段   | 类型  | 含义  |  备注 | 
|  ----      | ----    |  ----  | ----  | 
| city_id  |   int      | 城市id  |   | |
| from_block_id  |   string      | 起始block_id  |   | |
| to_station_id | bigint | 目标车站id|  |
| line_distance | float | 直线距离  |  |
|    navigation_distance     |   float   | 导航距离 |  |
|   version_code        |   string   | 版本分区  |   |

### 7.结果评估

|  字段   | 类型  | 含义  |  备注 | 
|  ----      | ----    |  ----  | ----  | 
| city_id  |   int      | 城市id  |   | |
| move_id  |   int      | 挪车id  |   | |
| dispatch_type | int | 派单类型|  |
| bike_sn | string | 车辆编号  |  |
| to_station_id     |   int   | 	挪入车站id |  |
| from_block_id        |   string   | 	挪入区块id  |   |
| move_start_time | timestamp | 挪车开始时间  |  |
| move_end_time     |   timestamp   | 挪车结束时间 |  |
| predict_24h_cost        |   float   | 预测挪出区块24h机会成本  |   |
| predict_24h_order | float | 预测挪入车站24h收益  |  |
| real_cost     |   float   | 挪出区块实际机会成本 |  |
| real_order        |   float   | 挪入车站实际收益  |   |
| event_day        |   string   | 日期分区  |   |

## ETP 工程mysql
### 1.区块实时车辆数据

工程端用于描述区块实时分布车辆信息

|  一级字段   | 二级字段  | 三级字段  |  备注 |
|  ----      | ----    |  ----  | ----  |
| block_id  |          |   |  区块id |
| bike_list | bike_sn |   |  车辆编号 |
|           | info |   |  车辆信息 |
|           |      | score  |  不动车预测分、优秀车预测分等 |
|           |      | last_order_time  |  距离上次订单或挪车时，单位h,工程侧实时获取可限制频繁挪出 |
|           |      | bike_type  |  算法识别车辆类型，11-停运/近似停运，12-需求不足，13-异常堆车，14-车辆问题 |
|           |      | bike_model  |  车型 |
|           |      | bike_usage  |  城市车效 |
```json
[{
    "block_id": 111,
    "bike_list": [{
        "bike_sn": "8888888",
        "info": {
            "score": 10,
            "last_order_time": 1,
            "bike_type": 1,
            "bike_model": "Q10",
            "bike_usage": 7.40
        }
    }, {
        "bike_sn": "8888881",
        "info": {
            "score": 10,
            "last_order_time": 1,
            "bike_type": 1,
            "bike_model": "Q10",
            "bike_usage": 7.40
        }
    }]
}]
```

### 2.车站数据

|  一级字段   | 二级字段  | 三级字段  |  备注 |
|  ----      | ----    |  ----  | ----  |
| station_id  |          |   |  车站id |
| info |  |   |  车站信息 |
|           | target_bike_cnt |   |  时空目标车辆，时间分周末和周内 |
|           | explore_target_bike_cnt   |   |  探索策略下时空目标车辆 |
|           | context_bike_cnt    |   |  实时车站周围25m内车辆数,工程侧实时获取|
|           |  in_bike_cnt    |   |  本时间段内截止目前车站25m内流入车辆数,工程侧实时获取 |
|           | predict_in_bike_cnt     |   |  车站在span时间段内预测骑入量 |
|           |  diapatching_bike_cnt    |   |  车站正在派入量,工程侧实时获取 |
|           |  efficiency    |   |  车站时空效率 |
|           |  profit_order    |   |  车站收益:历史30d骑出车辆24h平均订单 |
```json
[{
    "station_id": 11,
    "info": {
        "target_bike_cnt": 10,
        "explore_target_bike_cnt": 15,
        "context_bike_cnt": 5,
        "in_bike_cnt": 1,
        "predict_in_bike_cnt": 2,
        "diapatching_bike_cnt": 1,
        "efficiency": 0.5,
        "profit_order": 3.4
    }
}]
```

### 3.区块与车站距离成本

|  一级字段   | 二级字段  | 三级字段  |  备注 |
|  ----      | ----    |  ----  | ----  |
| block_id  |          |   |  区块id |
| station_list |  |   |  车站列表信息 |
|           | station_id |   |  车站数据中的存在的车站id |
|           | info   |   |  车站信息 |
|           |     |  distance |  block到station的距离，当station在block中，距离为零，否则是区块中心点到车站的距离|
|           |      |  price |  挪车成本，目前未上线，由距离拟合得出 |
```json
[{
    "block_id": 111,
    "station_list": [{
        "station_id": 23,
        "info": {
            "distance": 45.2,
            "price": 1.2
        }
    }]
}]
```

### 4.transfer plan
由前三张信息表通过OT输出区块到车站的挪车信息

|  一级字段   | 二级字段  | 三级字段  |  备注 |
|  ----      | ----    |  ----  | ----  |
| block_id  |          |   |  实时车辆分布中block_id |
| target_type |  |   |  target类型：原始/探索 |
| station_list        |  |   |  车站列表 |
|           | station_id   |   |  车站数据中的车站id |
|           |  info   |   |  车站信息 |
|           |      |  move_cnt |  挪入数量 |
|           |      |  score |  分数 |

```json
[{
    "block_id": 111,
    "target_type":"optimal"
    "station_list": [{
        "station_id": 23,
        "info": {
            "move_cnt": 4,
            "score": 1.45
        }
    }]
}]
```
```python
# 按城市求解过程
import h3
import ot
import math
import json
import pandas as pd
import numpy as np
import configparser
from pandas.io.json import json_normalize
import os
import json

def execute_ot(block, station, cost,target_field):
    #=== config ===
    cf = get_transfer_config()
    explore_max_increase_rate = float(cf.get("transfer", "explore_max_increase_rate"))
    explore_max_value = int(cf.get("transfer", "explore_max_value"))
    print("explore_max_increase_rate:{} explore_max_value:{}".format(explore_max_increase_rate,explore_max_value))
    #=== source ===

    #=== block ===
    block_df = pd.DataFrame(block)
    block_df['bikes'] = block_df.bike_list.map(lambda x:len(x))
    block_df.rename(columns={'block_cost_opportunity':'opportunity_cost'},inplace=True)
    cost_block_list = [b['block_id'] for b in cost]
    block_df = block_df[block_df.block_id.isin(cost_block_list)]
    block_df = block_df.sort_values(['block_id']).reset_index(drop=True)

    #=== station ===
    station_df = pd.DataFrame(station)
    station_df["info"] = station_df["info"].apply(lambda x: revise_explore(x, explore_max_increase_rate,explore_max_value))
    station_df['station_profit'] = station_df['info'].apply(lambda x:0 if (x[target_field]-x['context_bike_cnt']-x['predict_in_bike_cnt'])<=0 else x['profit_order'])

    """
    之前车站收益是用效率计算的，后续已用更合适的方式得出
    """
    # station_df['efficiency'] = station_df['info'].apply(lambda x:x['efficiency'])
    # station_df['eff_coef'] = station_df.apply(lambda x:3 if x['efficiency']>6 else np.log(x['efficiency']+1)+1,axis=1)
    # station_df['station_profit'] = station_df.apply(lambda x:x['station_profit']*x['eff_coef'],axis=1)

    station_df['target_bike_cnt_revise'] = station_df['info'].map( lambda x: get_target(x,target_field))
    ""
    车站目标不减去当前周围存在的车辆数，是因为保留ot时候自己转移自己，不保留的话会都挪出去的
    "" 
    station_df = station_df.sort_values(['station_id']).reset_index(drop=True)
    filter_1_station_list = list(station_df[station_df.target_bike_cnt_revise==1]['station_id'])
    station_df['target_bike_cnt_revise'] = station_df.target_bike_cnt_revise.map(lambda x:0 if x==1 else x)

    #=== cost ===
    cost_df = json_normalize(cost,['station_list'],['block_id'])
    cost_df = cost_df.sort_values(['block_id','station_id']).reset_index(drop=True)
    cost_df['info.distance'] = cost_df['info.distance'].astype('float')
    cost_df['dis_cost'] = cost_df.apply(lambda x:get_dis_cost(x['info.distance']),axis=1)

    #=== fuse ===
    m1 = cost_df.merge(station_df,on=['station_id'],how='left')
    m2 = m1.merge(block_df,on=['block_id'],how='left')
    filter_1_block_list=list(m2[(m2['info.distance']<1) & (m2['station_id'].isin(filter_1_station_list))]['block_id'])
    m2['total_profit'] = m2.apply(lambda x:999 if x['info.distance']< 25 else (x['station_profit']-x['opportunity_cost'])*2.5-x['dis_cost'] ,axis=1)
    m2['end_cost'] = -m2.total_profit
    cost_array = np.reshape(np.array(list(m2['end_cost'])),(-1,len(station_df))) # b --> s
    block_df['bikes'] = block_df.apply(lambda x:x['bikes']-1 if x['block_id'] in filter_1_block_list and x['bikes']>0 else x['bikes'],axis=1)
    #m2.to_json("original.json")

    #=== calcute ===
    b = list(block_df['bikes'])
    s = list(station_df['target_bike_cnt_revise'])
    print("supply accept:",sum(b),sum(s))
    m = min(sum(s),sum(b))
    M = cost_array
    martix = ot.partial.partial_wasserstein(b,s,M=cost_array,m=m)

    va2 = pd.DataFrame(martix,index=list(block_df['block_id']),columns =list(station_df['station_id']))
    res2 = []
    for idx,val in va2.iterrows():
        for col in va2.columns:
            if val[col] > 0:
                res2.append((idx,col,val[col]))
    if not res2:
        return []
    trans_df = pd.DataFrame(res2,columns=['block_id','station_id','move_cnt'])
    trans_df['move_cnt'] = trans_df['move_cnt'].astype('int')
    trans_df = trans_df.merge(m2[['block_id','station_id','end_cost']],on = ['block_id','station_id'] ,how = 'left')
    trans_df.rename({'end_cost':'score'},axis=1,inplace=True)
    trans_df['info'] = trans_df.apply(lambda x:{'station_id':x['station_id'],'info':{'move_cnt':x['move_cnt'],'score':(-x['score'])*x['move_cnt']}},axis=1)

    return trans_df

def revise_explore(x, max_increase_rate,explore_max_value):
    if x["target_bike_cnt"] > explore_max_value:
        x["explore_target_bike_cnt"] = x["target_bike_cnt"]
    else:
        x["explore_target_bike_cnt"] = int(min(x["explore_target_bike_cnt"], x["target_bike_cnt"] * (max_increase_rate + 1.0), x["target_bike_cnt"] +3))
        x["explore_target_bike_cnt"] = 3 if x["explore_target_bike_cnt"] == 2 else x["explore_target_bike_cnt"]
    return x

def get_op_cost(x): # 原先用区块内车辆平均值代替区块机会成本，后续已经用更合适的方式计算得出
    score_list = [b['info']['score'] - 0.5 for b in x]
    for i in range(len(score_list)):
        if score_list[i] == 1:
            score_list[i] = 0
        elif score_list[i] > 1:
            score_list[i] = 1
        else:
            continue
    avg_score = np.mean(score_list)
    bike_usage = x[0]['info']['bike_usage']
    opportunity_cost = (1 - avg_score) * 2.5  # 去除车效
    return [opportunity_cost, len(x)]

def get_dis_cost(x): #距离成本
    if 0<=x<=1000:
        cost = (1.4-0)*(x-0)/(1000-0)
    elif 1000<x<=2500:
        cost = (2.1-1.4)*(x-1000)/(2500-1000) + 1.4
    elif 2500<x<=4000:
        cost = (2.8-2.1)*(x-2500)/(4000-2500) + 2.1
    elif 4000<x<=5500:
        cost = (4.2-2.8)*(x-4000)/(5500-4000) + 2.8
    else:
        cost = 4.2
    return cost

def get_target(x,target_field):
    if x[target_field]-max(x['predict_in_bike_cnt'],x['in_bike_cnt'])-x['diapatching_bike_cnt']>=0: #时段初期需求正常
        tg = x[target_field]-max(x['predict_in_bike_cnt'],x['in_bike_cnt'])-x['diapatching_bike_cnt']
    else:
        if x['predict_in_bike_cnt']>x['in_bike_cnt']:  # 时段初期需求正常但预测流入偏高/末期需求减少
            tg = x[target_field]
        else:      # 时段末期需求突增
            tg = int(0.8*x[target_field])
    return tg

```


```sql
最优运输规划表
CREATE TABLE `transfer_result_v2_202208` (
`id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键',
`plan_id` bigint(20) NOT NULL DEFAULT '0' COMMENT '执行计划id',
`city_id` bigint(8) NOT NULL DEFAULT '0' COMMENT '城市id',
`area_id` int(8) NOT NULL DEFAULT '0' COMMENT '区域id',
`group_id` int(8) NOT NULL DEFAULT '0' COMMENT '分组id',
`tdate` int(8) NOT NULL DEFAULT '0' COMMENT '日期',
`span_id` bigint(16) NOT NULL DEFAULT '0' COMMENT 'span_id',
`source_block_id` varchar(200) NOT NULL DEFAULT '0' COMMENT 'resource_block_id',
`target_type` varchar(50) NOT NULL DEFAULT '' COMMENT '目标车站target类型',
`target_station_id` bigint(20) NOT NULL DEFAULT '0' COMMENT '目标车站ID',
`move_num` int(8) NOT NULL DEFAULT '0' COMMENT '挪车数量',
`index` int(8) NOT NULL DEFAULT '0' COMMENT 'index',
`score` float NOT NULL DEFAULT '0' COMMENT '分数',
`status` tinyint(4) NOT NULL DEFAULT '1' COMMENT '状态；1:正常；0删除',
`create_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
`update_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
PRIMARY KEY (`id`),
KEY `idx_plan_id` (`plan_id`),
KEY `idx_tdate_city_id` (`tdate`, `city_id`),
KEY `idx_tdate_plan_id` (`tdate`, `plan_id`)
) ENGINE = InnoDB AUTO_INCREMENT = 23647228 DEFAULT CHARSET = utf8 COMMENT = 'transfer_result_v2_202208'
```
### 5.transfer result
过滤下列求解出的运输计划：
- 当车站距离所在区块中心的距离< 25 米，当前区块不往车站挪车，这时score 是 999 的倍数，需要过滤掉
- 当roi为0或负的时候，不挪车，即score  <= 0,需要过滤掉
- 当station缺口较少，即对目标车站挪车数量group by求和, 为1的时候过滤

```sql
运输规划过滤表
CREATE TABLE `transfer_plan_v2_202208` (
`id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键',
`city_id` bigint(8) NOT NULL DEFAULT '0' COMMENT '城市id',
`area_id` int(8) NOT NULL DEFAULT '0' COMMENT '区域id',
`group_id` int(8) NOT NULL DEFAULT '0' COMMENT '分组id',
`tdate` int(4) NOT NULL DEFAULT '0' COMMENT '日期',
`plan_id` bigint(16) NOT NULL DEFAULT '0' COMMENT '计划id',
`span_id` int(8) NOT NULL DEFAULT '0' COMMENT 'span_id',
`source_block_id` varchar(200) NOT NULL DEFAULT '0' COMMENT '起始区块id',
`source_block_center_points` varchar(40) NOT NULL DEFAULT '' COMMENT '起始区块中心点',
`source_max_bike` int(8) NOT NULL DEFAULT '0' COMMENT '起始区块最大车辆数',
`source_min_bike` int(8) NOT NULL DEFAULT '0' COMMENT '起始区块最小车辆数',
`target_type` varchar(50) NOT NULL DEFAULT '' COMMENT '目标车站target类型',
`target_station_id` bigint(16) NOT NULL DEFAULT '0' COMMENT '目标车站id',
`target_station_points` varchar(40) NOT NULL DEFAULT '0' COMMENT '目标车站中心点',
`target_max_bike` int(8) NOT NULL DEFAULT '0' COMMENT '目标车站最大车辆数',
`target_min_bike` int(8) NOT NULL DEFAULT '0' COMMENT '目标车站最小车辆数',
`move_num` int(8) NOT NULL DEFAULT '0' COMMENT '移动车辆数',
`score` float NOT NULL DEFAULT '0' COMMENT '分数',
`index` int(8) NOT NULL DEFAULT '0' COMMENT 'index',
`status` tinyint(4) NOT NULL DEFAULT '1' COMMENT '状态；1:正常；0删除',
`create_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
`update_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
PRIMARY KEY (`id`),
KEY `idx_plan_id` (`plan_id`),
KEY `idx_tdate_city_id` (`tdate`, `city_id`),
KEY `idx_tdate_plan_id` (`tdate`, `plan_id`)
) ENGINE = InnoDB AUTO_INCREMENT = 16556884 DEFAULT CHARSET = utf8 COMMENT = 'transfer_plan_v2_202208'
```

### 6.route plan 

- 1.工程端进行上述候选挪车计划的组合派单
- 2.运输规划过滤表基本和工单候选消费表数量为1对1的关系
- 3.如果source到target运输数量超过工人一次可载上限12辆，将会被拆分，形成最终下述工单候选消费表
- 4.工单候选融合派单表生成通过多因素加权得分考虑
- 5.递归枚举一个目标车站所有候选消费id,直到这些挪车数量加起来满足最大挪车数量，即终止，形成一个候选的挪车组合
- 6.将一个目标车站的所有顺序的组合通过下列打分，选出最优的派单出去
- 7.打分包括：车辆得分、挪车距离、挪车体感（成单是否一条线，而不是分散点）、挪车难度等
- 8.最终形成工单候选融合派单表
- 9.每次工人请求接单时候就会进行上述计算，进而派单

```sql
工单候选消费表
CREATE TABLE `route_plan_consume_202208` (
`id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键',
`city_id` bigint(8) NOT NULL DEFAULT '0' COMMENT '城市id',
`tdate` int(4) NOT NULL DEFAULT '0' COMMENT '日期',
`source_block_id` varchar(20) NOT NULL DEFAULT '0' COMMENT '起始blockid',
`source_min_bike` int(4) NOT NULL DEFAULT '0' COMMENT '起始区块最小车辆数',
`bike_num` int(8) NOT NULL DEFAULT '0' COMMENT '移动车辆数',
`station_id` bigint(8) DEFAULT NULL,
`plan_id` bigint(20) NOT NULL DEFAULT '0' COMMENT '计划id',
`transfer_id` bigint(20) DEFAULT NULL COMMENT '对应上述运输规划过滤表中主键id' ,
`span_id` int(8) NOT NULL DEFAULT '0' COMMENT 'span_id',
`batch_num` int(4) DEFAULT NULL,
`trace_id` varchar(200) DEFAULT NULL,
`status` tinyint(4) NOT NULL DEFAULT '1' COMMENT '状态；1:正常；0删除',
`create_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
`update_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
PRIMARY KEY (`id`),
KEY `index_trace_id` (`trace_id`),
KEY `index_city_id` (`city_id`),
KEY `index_tdate` (`tdate`),
KEY `index_source_block_id` (`source_block_id`),
KEY `index_station_id` (`station_id`)
) ENGINE = InnoDB AUTO_INCREMENT = 3874134 DEFAULT CHARSET = utf8 COMMENT = 'route_plan_consume_202208'
```

```sql
工单候选融合派单表
CREATE TABLE `route_plan_202208` (
`id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键',
`city_id` bigint(8) NOT NULL DEFAULT '0' COMMENT '城市id',
`plan_id` int(8) NOT NULL DEFAULT '0' COMMENT '计划id',
`tdate` int(4) NOT NULL DEFAULT '0' COMMENT '日期',
`consume_ids` varchar(200) NOT NULL DEFAULT '' COMMENT '对应上述工单候选消费表中主键id',
`labor_id` bigint(16) NOT NULL DEFAULT '0' COMMENT '工人id',
`points` varchar(40) NOT NULL DEFAULT '0' COMMENT '坐标',
`station_ids` varchar(40) NOT NULL DEFAULT '' COMMENT '车站id集合',
`bike_num` int(8) NOT NULL DEFAULT '0' COMMENT '车辆数量',
`move_type` int(8) DEFAULT '11',
`trace_id` varchar(200) DEFAULT NULL COMMENT 'trace_id',
`order_id` bigint(16) NOT NULL DEFAULT '0' COMMENT '订单id',
`distance` double DEFAULT NULL COMMENT '总的挪车距离：从第一辆车检车到最后一量，再到车站的距离',
`rank` int(8) DEFAULT NULL,
`score` double DEFAULT NULL COMMENT '得分',
`status` tinyint(4) NOT NULL DEFAULT '1' COMMENT '状态；1:正常；0删除',
`order_status` int(4) NOT NULL DEFAULT '0' COMMENT '订单状态，执行成功或者取消',
`create_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
`update_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
`bike_details` varchar(3000) NOT NULL DEFAULT '' COMMENT '车辆详情',
`attribute4Num` int(8) NOT NULL DEFAULT '0' COMMENT '不同的车辆类型，不动车',
`attribute7Num` int(8) NOT NULL DEFAULT '0' COMMENT '仓库车',
`attribute8Num` int(8) NOT NULL DEFAULT '0' COMMENT '服务区边缘车',
`bike_sn_attribute4` varchar(400) NOT NULL DEFAULT '' COMMENT 'bike_sn_attribute4',
`bike_sn_attribute7` varchar(400) NOT NULL DEFAULT '' COMMENT 'bike_sn_attribute7',
`bike_sn_attribute8` varchar(400) NOT NULL DEFAULT '' COMMENT 'bike_sn_attribute8',
`distance1` double NOT NULL DEFAULT '0' COMMENT '人到第一辆车的导航距离' ,
`distance4` double NOT NULL DEFAULT '0' COMMENT '最后一辆到站的导航距离' ,
PRIMARY KEY (`id`),
KEY `index_city_id` (`city_id`),
KEY `index_tdate` (`tdate`),
KEY `index_trace_id` (`trace_id`)
) ENGINE = InnoDB AUTO_INCREMENT = 83759 DEFAULT CHARSET = utf8 COMMENT = 'route_plan_202208'

其中bike_details:{"blockId":车辆所在区块，"lastMoveEndTime"：上次挪车结束时间，"distance"：无意义，"latitude":纬度,"bikeSn":车辆编号,"attribute4Value":3, "lastOrderEndTime":上次订单结束时间,"longitude":经度}

```
## ETP路径

- 城市时段划分：按一段时间车效进行划分不同span，每个城市时间段不宜变化过频，故月度、季度变化
- 城市时点需求：计算各城市时段总需求，后期换做了近期可用车数量，限制时段挪车量上限
- 车站时空信息：车站容量上下限，预估流入数量，用于车站最终目标分配

## ETP三期迭代变化及思考

- 1.一期多人烟囱式的粗放开发，到二期架构升级底表统一收敛到我，三期将中间结果简化不传入工程，且模块拆分职责边界更清晰
- 2.一期车站收益、区块机会成本均粗糙，二期、三期更加细致准确，增加挪车评估，同时和商分多次对齐口径，保持我们评估与其计算的收益正相关，评估指标采用车辆挪后24h真实收益-挪前区块机会成本
- 3.区块机会成本、车站收益、挪后24真实收益计算均采用折扣收益，基于0～23h共24个时段内，各个时段发生订单的次数乘以alpha^h，例如：一个区块（11级h3）在某个span：8～10内停有两辆车，其中一辆车在7:20停入，另一辆在9:10停入，那么这区块的机会成本为这两辆车的平均24h折扣收益，第一辆7点停的车从时段开始8点计算后续第0～23h,假设它在8:35产生1单，在12:15、12:40产生2单，那么其24h收益为:1✖️0.9^0+2✖️0.9^3=1+0.729=1.729,同理得到另一个车的收益，将两车加和平均即为该区块该时段当天的区块机会成本，区块最终的机会成本使用30天的机会成本平均得到。同理可得，车站收益也需要计算每天，用30天的平均值；挪后真实折扣收益只看当天。另外，评估时候的区块机会成本有个争议，因为商分是将区块分200组，用AB看作真实区块机会成本，我们想用预估机会成本，跨部门对齐推动比较难
- 4.建模为基于两个分布（车辆分布和车站分布）的最优运输问题，不同于之前的盈余预测角度














