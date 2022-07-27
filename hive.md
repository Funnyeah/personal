### 基础认识
?> Hive

&emsp;&emsp;Apache Hive是一个构建于Hadoop(分布式系统基础架构)顶层的数据仓库，注意这里不是数据库。Hive可以看作是用户编程接口，它本身不存储和计算数据；它依赖于HDFS(Hadoop分布式文件系统)和MapReduce(一种编程模型，映射与化简；用于大数据并行运算)。其对HDFS的操作类似于SQL—名为HQL，它提供了丰富的SQL查询方式来分析存储在HDFS中的数据；HQL经过编译转为MapReduce作业后通过自己的SQL 去查询分析需要的内容；这样一来，即使不熟悉MapReduce 的用户也可以很方便地利用SQL 语言查询、汇总、分析数据。而MapReduce开发人员可以把己写的mapper 和reducer 作为插件来支持Hive 做更复杂的数据分析。

?> HBase

&emsp;&emsp;Apache HBase是运行于HDFS顶层的NoSQL(=Not Only SQL，泛指非关系型的数据库)数据库系统。区别于Hive，HBase具备随即读写功能，是一种面向列的数据库。HBase以表的形式存储数据，表由行和列组成，列划分为若干个列簇(row family)。例如：一个消息列簇包含了发送者、接受者、发送日期、消息标题以及消息内容。每一对键值在HBase会被定义为一个Cell，其中，键由row-key(行键)，列簇，列，时间戳构成。而在HBase中每一行代表由行键标识的键值映射组合。Hbase目标主要依靠横向扩展，通过不断增加廉价的商用服务器，来增加计算和存储能力。


&emsp;&emsp;逻辑视图：hbase存储单元：cell概念 由rowkey、column、timestamp、type、value五部分组成一个表中的一个字段的值
LSM树结构。 HBase 写数据先写到Hlog中，接着到 Memorystore（跳表实现memorystore，内存），数据写到一定量，触发flush,将数据存到hfile（硬盘），hfile写到一定程度，触发compaction合并

布隆过滤器用于判断一个数据是否存在与hfile文件中（布隆做一个哈希操作判断）对row key 进行布隆操作
Master 管理region  ddl dml

?> 表在hdfs中的存储方式

分区表改变了Hive对数据存储的组织方式。   
如果我们是在mydb.db数据库中创建的employees这个表，那么对于这个表会有employees目录与之对应,如果是双分区表，则会出现对应分区country和state目录： 
"hdfs://master_server/user/hive/warehouse/mydb.db/employees"  
.../employees/country=CA/state=AB           
.../employees/country=CA/state=BC

### 创建删除

?> 创建

```sql
-- 创建表
CREATE TABLE if not exists ai.dws_ai_station_ab_test_mf(    
city_id int COMMENT '城市id',  
block_id bigint COMMENT '区块id',  
online_station_id bigint COMMENT '区块绑定车站id，无绑定车站为-1',  
online_station_point string COMMENT '区块绑定车站坐标,lon|lat，无绑定车站为N',
online_md5_id string COMMENT '区块绑定车站坐标md5值, 无绑定车站为N',
new_station_id bigint COMMENT '区块新车站id默认为-1，无新车站为-2',
new_station_point string COMMENT '区块新车站坐标,lon|lat，无新车站为N',
new_md5_id string COMMENT '区块新车站坐标md5值, 无新车站为N',
stgy_model string COMMENT '策略：A、B'
)
COMMENT '车站AB策略表'
PARTITIONED BY (
model_version string COMMENT '版本')
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\001' STORED AS TEXTFILE;
```

?> 删除
```sql
 -- 删除表（物理）
 drop table if exists ai.dwd_block_object_assess_test_da
```

```sql
-- 删除表所有内容（保留结构）
TRUNCATE table ai.dwd_block_object_assess_test_da
```

```sql
-- 删除表分区
alter table  ai.dwd_block_object_assess_test_da drop partition (event_day='20210710')
```

```sql
-- 删除表分区(分区存在NUll值)
ALTER TABLE table_name DROP IF EXISTS PARTITION (event_day='__HIVE_DEFAULT_PARTITION__',pk_month='__HIVE_DEFAULT_PARTITION__')

```
### 修改查询
```sql
-- 修改字段名和类型
ALTER TABLE ai.dws_ai_jw_block_average_order_da        -- 需要更改的库名.表名
CHANGE COLUMN old_name new_name bigint COMMENT '区块编号'  -- 旧字段名 新字段名 字段类型 备注
```

```sql
-- 新增字段
alter table table_name add columns(new_column string comment '新增字段') cascade 
 -- 不加cascade更改表 再写入分区数据时，新增列数据为null
```

```sql
-- 查询ai数据库中包含的表
show tables in ai    
```

```sql
-- 查询表中存在的所有分区
show partitions ai.ads_ai_jw_block_scene_full_2   
```


### 语法
[hive](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF)
[presto](https://prestodb.io/docs/current/)
#### 基本运算符

```sql
select 10 / 3     
-- +数字相加；-数字相减; *为数字相乘; /为浮点数除法; div为整除; %为取余数

select 5 | 7  
-- |为二进制按位或运算; &为二进制按位与运算; ^为二进制按位异或(不同为1)

select 'd'||2, 2||1.5 --return 'd2','21.5'
-- ||接受数字和字符类型，返回字符类型拼接

```
#### 比较运算符
```sql
select 1 = 2, 1 == 2     
-- 均返回False，同义运算符     

select 1 <=> 2    
-- 非NULL值数据时等价于 = 和 ==  ,若有均为NULL 返回 True, 一方为NULL返回False

select 1 <> 2    
-- 这个是hive sql里的不等于号   等价于 != ,符号间不能有空格, 若有一方为NULL、或者均为NULL, 则返回NULL

select 1 <2 ,1<=2 ,1> 2,1 >= 2   
-- 大于、小于、大小等于符号， 符号间不能有空格,  若有一方为NULL或均为NULL, 则返回NULL  
```
#### 常用函数
```sql
select * from ai.dws_ai_jw_block_flag_da where city_id is null 
-- xx字段 is null、xx字段 is not null（xx字段为空、不为空）

select cast(5.231313 as DECIMAL(10,4))   
-- 保留指定精度, DECIMAL(arg1,arg2)，arg1表示返回数字长度最长为多少（必须大于整数部分+保留小数点位数的长度），arg2表示小数点后保留位数

select ltrim(string A) ,rtrim(string A)  
-- 删除字符串中左右空格 

select trim(' 13 df  ')     
-- 删除前后所有空格，不删除中间

select LPAD('lxh',4,'0'), lpad('lxh',2,'0'), RPAD('df',5,0)     
-- 返回 '0lxh' ,'lx' ,'df000'  填充，若返回的长度小于原先字符串的长度，则不填充并截取所返回长度的原字符串

select length('fef'),locate('e','sfesfe')  
-- 返回 长度为3 ,  第一个出现该字符的位置3 

select reverse('poqw')     
-- 返回 'wqop' 反转

select REPEAT(1,2)+1    
-- 重复1两次，实际是字符串'11' 又加了个数字1, 涉及自动类型转换 输出为12.0浮点数

select round(1.52442001324242,5)   
-- 返回1.52442   保留小数点后5位小数，不指定则默认不保留小数点

select floor(1.96464)
-- 返回1   向下取整

select ceil(1.46464)
-- 返回2   向上取整

select * from ai.dws_ai_jw_block_flag_da m where event_day = '20210607'  order by  rand(12)   
-- 全局随机 rand(可以指定随机种子)

select exp(flag) from ai.dws_ai_jw_block_flag_da    
-- 对某列数值数据取e^col

select ln(flag) from ai.dws_ai_jw_block_flag_da    
-- 对某列数值数据取ln(col) log10() log2() log(3,5) 同理

select pow(2,3),sqrt(4)   
-- 返回2^3 平方函数; 返回2 开根号函数

select bin(5),hex(16)   
-- 返回101 输入10进制返回二进制; 返回10 输出10进制返回十六进制

select factorial(2)  
-- 返回2的阶乘

select 'fosfsfsfsf' like 'f%' 
-- 返回true 通配符 或者 select 'fosf' like 'f_ _ _ _'  返回true  '_'匹配一个字符 

select 'tcghhgsf2s3f3' rlike 'sf'  
-- 返回true 只要字符串包含匹配的字符，无论任何位置都返回true

select space(10)
-- 返回十个长度的空字符串'          '

```

#### 常用时间函数
presto
```sql
select parse_datetime('20220318', 'yyyyMMdd')
-- return 2022-03-18 00:00:00.0  #presto 字符串 2 日期date格式
select to_unixtime(parse_datetime('20220318', 'yyyyMMdd'))  
-- return 1647532800   #presto 日期date格式 2 秒级时间戳
select from_unixtime(1647532800) 
-- return 2022-03-18 00:00:00.0   presto  秒级时间戳 2 日期date格式
select REPLACE(SUBSTRING(cast('2022-01-16 05:01:08.0' as VARCHAR),1,10),'-','')
-- return 20220116  #presto 日期date格式 2 转字符串
select format_datetime(from_unixtime(1609167953000/1000),'yyyy-MM-dd')  
-- return 2020-12-18 #presto 正规的日期date格式 2 字符串, 字符串转date格式用第一个
select format_datetime(from_unixtime(1609167953694/1000)+ interval '8' hour + interval '30' MINUTE,'yyyy-MM-dd HH:mm:ss') 
-- return 2020-12-29 07:35:53  format_datetime 还可以加时间偏移 hh为12小时制，HH为24小时制
```
hive
```sql
select from_unixtime(1609167953694/1000)   
-- 秒级时间戳 2 日期date格式(date格式本质就是字符串)
-- from_unixtime需要注意在hive中他的参数不能是运算表达式，如这条会报错;在presto中不报错
select UNIX_TIMESTAMP('20211106','yyyymmdd') 
-- return 1609863060  #hive 日期字符串 2 秒级时间戳
select FROM_UNIXTIME(UNIX_TIMESTAMP('20211106','yyyymmdd'),'yyyy-mm-dd')  
-- return 2021-11-06 #hive 秒级时间戳 2 日期date
select cast('2022-04-20' as TIMESTAMP),from_unixtime(UNIX_TIMESTAMP('20220420', 'yyyyMMdd') + 24*60*60-1) 
-- return 2022-04-20 00:00:00.0   2022-04-20 23:59:59  
-- cast 可直接将yyyy-mm-dd字符串格式转为日期格式；将字符串转为时间戳加偏移得到当天结束的日期格式
select cast('2022-04-20' as TIMESTAMP)- interval '1' day  , cast('2022-04-20' as date)- interval '1' day  - interval '30' minute
-- return 2022-04-19 00:00:00	2022-04-18 23:30:00 用日期做时间偏移 和presto用法相同
```

#### 其他时间函数 
```sql
select  to_date("1970-01-01 00:00:00")    
-- return "1970-01-01"   返回日期date格式的 年月日
select year("1970-01-01 00:00:00")    
-- return 1970     int格式 年份 
select quarter("1970-01-01 00:00:00")    
-- return  1     int格式 季节(1-4)   
-- month （1-12）月份 day  hour  minute  second  weekofyear  

select datediff('2009-03-01', '2009-03-03') 
-- return -2 日期差（天）前一个减去后一个
select date_add('2008-12-31', 1)  
-- return '2009-01-01'   日期加一天     
select date_sub('2008-12-31', 1)  
-- return '2008-12-30'   日期减一天 
select date_sub(current_date(),1)    
-- return '2022-04-20'    current_date() 当前日期'2022-4-21'

select current_timestamp()  
-- return '2022-04-21 16:44:04.855'
select last_day('2021-06-09 11:16:28.245') 
-- return '2021-06-30' 返回当前日期所在月份的最后一天
select next_day('2021-06-09', 'Monday')     
-- return '2021-06-14'   返回当前日期下一个指定的周几   本例中6.9为周三，下一个周一为6.14   'Mo' 、'Mon'、'Monday'  三种形式均可
select  trunc('2021-06-09', 'YY')    
-- return '2021-01-01'   返回指定的截断月份（'MM'）或者年份的第一天     
select months_between('1997-02-28 10:30:00', '1996-10-30')
-- return 3.94959677  根据毫秒级计算的月份差

```

#### 条件表达式
```sql
select if(5>0,111,000) as ff      
-- return 111   if(条件判断，T_value,F_value)

select * from ai.dws_ai_jw_block_flag_da where  isnotnull(flag)  and isnull(city_id)   
select * from ai.dws_ai_jw_block_flag_da where  flag is not null  and city_id is null
-- retrun 字段flag不为空且字段city_id为空的数据 上述两语句等价,前者只是hive语法，后者presto通用 isnotnull不为空  isnull为空  

select  nvl(23,11) , nvl(null,11)
-- return 23 , 11  若不空则返回原值，若空返回默认值，后一个参数为默认值，前一个为某列值

select coalesce(NULL,5,NULL,NULL)
-- return 5  返回第一个不为空的值   若全为空则返回NULL      

select flag, (case when flag=0 then 999 when flag=1 then 8  [else 0] end ) ff from ai.dws_ai_jw_block_flag_da 
-- 字段符合条件则ff赋对应值

select case a when b then res1 when c then res2 else res3 end as col_a
-- 等值比较，当字段a=b时赋值res1,等于a=c时赋值res2,其余res3
select case  when block_hot>200 then 9 when block_hot>100 then 6 when block_hot>10 then 3 else 1 end as col
-- 上述为不等值比较, 以及包含聚合函数的统计

```

#### 分组聚合函数
```sql
with tmp as (
select 1001 station_id, 1 win_id union 
select 1001 station_id, 2 win_id union 
select 1001 station_id, 3 win_id 
)
select station_id,array_join(array_agg(win_id),',') res from  tmp  group by station_id
-- presto return 1001  '1,2,3'  
-- array_agg() 返回列表 array_distinct(array_agg(b)) 对列表去重复

select xx,group_concat(win_id) from tb group by xx 
-- mysql  默认逗号拼接 

select station_id,collect_list(win_id) from tmp group by station_id
-- hive return 1001  [1,2,3] collect_list()返回聚合列表  collect_set() 列表去重

select sum() over (partition by col_a,b order by col_c,d range between xx and xx )
-- sum()求和  xx：向前使用preceding，向后使用following，当前行使用current row
-- unbounded preceding and current row 表示当前行和之前所有行
-- 不写range between 函数也等同于当前行和之前所有行

count(*) --行数,包括null的所有行
count(1) --行数,不包含null的所有行
count(distinct col_a,col_b) -- 不包含null的所选去重行数
min()、max()、avg() --最大、小、平均

var_pop() 有偏方差、var_samp() 无偏方差、stddev_pop()有偏标准差、stddev_samp()无偏标准差、covar_pop()有偏协方差、covar_samp()无偏协方差、corr(col1, col2) 皮尔逊相关系数

select 
  city_id,
  percentile(order_in, 0.9) max_p,
  COLLECT_LIST(order_in) debug_list
from
  span_in_out_da
group by
  city_id
percentile(BIGINT col, p) -- 返回整形列的分位数值,0<=p<=1，返回值可为浮点数
percentile_approx(DOUBLE col, p [, B]) --返回浮点型列分位数值,0<=p<=1,B为近似计算的参数，越小越精确代价越高

```
#### 字符串拼接函数
```sql
select concat_ws('-',cast(123 as string),cast(9 as string))   
-- return '123-9' 字段类型必须为string 或者 array<string> , 可拼接多列
select concat(city_id ,block_id) from ai.dws_ai_jw_block_flag_da      
--  无需转换类型，自动转换string
select 12||'123'   
-- return '12123' 自动类型转换
select CONCAT_WS('|',ARRAY('1','2','3') ),array(1,2,3)
-- return 1|2|3	 [1,2,3]
```
#### 字符串分割函数
```sql
select split('abcde','c'),   -- 返回 ['ab','de']
split('ab_cd_e','\_')[0] ,  -- 返回 'ab'     特殊字符作为分隔符需用 \ 转义
split('ab?cd_e','\\?')[0]  --  返回 'ab'        有些特殊字符需两个转义字符 \\

select SUBSTRING_INDEX('12|34|', '|', 1) as lng, SUBSTRING_INDEX('12|34|', '|', 2) ,SUBSTRING_INDEX(SUBSTRING_INDEX('12|34|', '|', 2), '|', -1) 
-- hive和mysql可使用 返回'12' 、'12｜34'、'34' 
-- 含义： '|'分隔符，数字表示返回第几个分隔符前面的，如1返回'12',2返回'12｜34'，-1返回分隔符后面的


select substr('hello',0,3),substr('hello',1,3),substr('hello',2,3),substr('hello',4,3),substr('hello',-1,3),substr('h e l l o',1,3)  
-- 返回hel、hel、ell、lo、o、h e(有个空格)
-- hive和presto可用，截取字段  参数：需截取的字符串，表示从第几个位置开始截取，向后截取的长度是多少(所以从-1位置开始向后只有-1位置的字符)
select CONCAT(SUBSTR('20211001',1,4) ,'-',SUBSTR('20211001',5,2),'-',SUBSTR('20211001',7,2))
-- 返回'2021-10-01' 可用于日期格式变换

-- 字符串替换函数
select replace('okns','k','1')    
--  返回o1ns 等同于 regexp_replace

```

#### 字符串解析函数
```sql
regexp_extract(str, regexp， idx) 
    str是被解析的字符串或字段名
    regexp 是正则表达式
    idx是返回结果 取表达式的哪一部分  默认值为1。
    0表示把整个正则表达式对应的结果全部返回
    1表示返回正则表达式中第一个() 对应的结果 以此类推

select REGEXP_EXTRACT('axt','ag?[a-z]') -- ax    任何东西后面跟？表示这个字符可选可不选（0或1），所以后面的[a-z]匹配了x
       REGEXP_EXTRACT('axt','ag?[a-z]+'),     -- axt   
       REGEXP_EXTRACT('axt','ag?')       --a

select REGEXP_EXTRACT('sf12.34,df556','[0-9]+\.?[0-9]+')    -- 12.34匹配第一次出现的正整数和正浮点数

select regexp_extract_all('sf-12.34,+455df556','[-+]?[0-9]+\.?[0-9]+')[2]   --  -12.34 可匹配正负号   regexp_extract_all返回匹配出的符合条件的列表

select regexp_extract_all('sfsf8sdd', '[0-9]+\.?[0-9]+') [1]   -- 报错匹配不到，+表示1或多个，因为[0-9]已经匹配了一个数字了，加号必须至少匹配一个，8后面无数字所以不满足条件报错

select regexp_extract_all('sfsf8sdd', '[0-9]*\.?[0-9]+') [1]   -- 8      * 表示重复0或多次

```

#### 复杂类型
```sql
1.map
with tmp as (select  map('s1',8,1,2,3,4,5,'s2') mp )
select mp,size(mp),map_keys(mp),MAP_VALUES(mp) from tmp
-- map() 传入string或者int,按顺序构成字典，可获取key和value
-- 返回 {'s1':8,1:2,3:4,5:'s2'}，4，['s1',1,3,5]，[8,2,4,'s2']

select EXPLODE(mp) from tmp      
-- 返回 key,value 两列，如果还需返回其他字段，需要加lateral view

select tf.*,t.* from (select 0 xx)t lateral view POSEXPLODE(array(1,2,3,4)) tf as idx,value 
-- POSEXPLODE 多返回一列下标序号

with tmp as (
   select  map('start_uasge',array('0'),'end_usage',array('3'),'span_nums',array('2'), 'span_list',array('0~4', '4~12','12~20', '20~24')) span_info union all
   select  map('start_uasge',array('3'),'end_usage',array('5'),'span_nums',array('4'), 'span_list',array('0~4', '4~8','8~12', '12~16','16~20','20~24')) span_info union all
   select  map('start_uasge',array('5'),'end_usage',array('8'),'span_nums',array('6'), 'span_list',array('0~4', '4~8','8~10', '10~12','12~14','14~16','16~20','20~24')) span_info union all
   select  map('start_uasge',array('8'),'end_usage',array('999'),'span_nums',array('8'), 'span_list',array('0~4', '4~6','6~8', '8~10','10~12','12~14','14~16','16~18','18~20','20~24')) span_info          
 ),

parms as (
 select 
     span_nums,
     hour_span,
     cast(split(hour_span,'~')[0] as int) start_hour,
     cast(split(hour_span,'~')[1] as int) - 1 end_hour,
     concat(
         cast(cast(split(hour_span,'~')[0] as int)*2 as string) ,
         '~',
         cast(cast(split(hour_span,'~')[1] as int)*2 - 1 as string)) span
 from (
     select *
        from (
         select 
             cast(span_info['span_nums'][0] as int) span_nums,
             cast(span_info['start_uasge'][0] as int) start_uasge,
             cast(span_info['end_usage'][0] as int) end_usage,
             span_info['span_list'] span_list
         from tmp
         )t lateral view explode(t.span_list) tmptd as hour_span
  )x 
)
select * from parms
-- map array explode 的联合使用示例

2.array
select array(1,2,3),array(1,2,3)[0],array(1.2,'x')
-- 返回[1,2,3], 1, ['1.2','x']  

select array(1,'2','x',1.2) x
union 
select array(1,2)
-- 返回error union时字段类型需一致，前者array<string>,后者array<int>，把后者改一个string参数就可自动类型转换


3.struct
select
  struct(c1, c2) x
from
  (
    select
      1 c1, 2 c2
    union
    select
      3 c1, 4 c2
  )c
  order by x.col2 desc
-- struct() 传入多列
-- 返回 
x
{"col1":3,"col2":4}
{"col1":1,"col2":2}

4.inline
select inline(array(struct('A',10,date '2015-01-01'),struct('B',20,date '2016-02-02')));
select inline(array(struct('A',10,date '2015-01-01'),struct('B',20,date '2016-02-02'))) as (col1,col2,col3);
select tf.* from (select 0) t lateral view inline(array(struct('A',10,date '2015-01-01'),struct('B',20,date '2016-02-02'))) tf;
select tf.* from (select 0) t lateral view inline(array(struct('A',10,date '2015-01-01'),struct('B',20,date '2016-02-02'))) tf as col1,col2,col3;

 Return       input_format                                                      含义
T1,…,Tn  inline(ARRAY<STRUCT<f1:T1,...,fn:Tn>> a)      Explodes an array of structs to multiple rows. Returns a row-set with N columns (N = number of top level elements in the struct), one row per struct from the array. (As of Hive 0.10.)

5.stack
select stack(2,'A',10,date '2015-01-01','B',20,date '2016-01-01');
select stack(2,'A',10,date '2015-01-01','B',20,date '2016-01-01') as (col0,col1,col2);
select tf.* from (select 0) t lateral view stack(2,'A',10,date '2015-01-01','B',20,date '2016-01-01') tf;
select tf.* from (select 0) t lateral view stack(2,'A',10,date '2015-01-01','B',20,date '2016-01-01') tf as col0,col1,col2;

T1,...,Tn/r    stack(int r,T1 V1,...,Tn/r Vn)    Breaks up n values V1,...,Vn into r rows. Each row will have n/r columns. r must be constant.

```

#### 排序与窗口函数
```sql
1.row_number() 默认由小到大排序，返回顺序号
2.rank() 默认由小到大排序，同值共号，返回断层的顺序号（某列[5,7,7,10],返回1,2,2,4）
3.dense_rank() 默认由小到大排序，同值共号，返回不断层的顺序号（某列[5,7,7,10],返回1,2,2,3）
4.ntile() 默认由小到大排序，返回分层顺序号
5.FIRST_VALUE(col_a), LAST_VALUE(col_a) 返回截止到当前行的按某字段分组的col_a字段的最小与最大值
6.lead(col_a,1) 返回窗口内向下1行数据的该字段值
6.lag(col_a,1) 返回窗口内向上1行数据的该字段值
```

#### hive与presto语法区别
```sql
select * from  (select * from tb)  -- presto 
select * from  (select * from tb)x -- hive 需加别名

-- presto列表下表从1开始，hive列表下表标从0开始
```


### 工具sql

#### 计算两个经纬度之间距离 
```sql
-- hive & presto
with tmp as (
  select
    109.67249220637174 b_lon,
    27.449656246000266 b_lat,
    109.69888390656 s_lon,
    27.4442168870677 s_lat
)
select
  1000 * 6371.393 * acos(
    cos(radians(cast(s_lat as double))) * cos(radians(cast(b_lat as double))) * cos(
      radians(cast(b_lon as double)) - radians(cast(s_lon as double))
    ) + sin(radians(cast(b_lat as double))) * sin(radians(cast(s_lat as double)))
  )
from
  tmp
```

#### 计算当前日期对应星期几 
```sql
with tmp as (
  select
    '20220412' event_day
)
select
  case
    -- presto
      mod(
        date_diff(
          'day',
          cast('2021-02-01' as date),
          parse_datetime(event_day, 'yyyyMMdd')
        ),
        7
      )
    -- hive
    (
      - datediff(
        '2021-02-01',
        FROM_UNIXTIME(
          UNIX_TIMESTAMP(event_day, 'yyyymmdd'),
          'yyyy-mm-dd'
        )
      )
    ) % 7
    when 0 then 1
    when 1 then 2
    when 2 then 3
    when 3 then 4
    when 4 then 5
    when 5 then 6
    when 6 then 7
  end AS day_of_week
from
  tmp
```
#### 一行转多行 
```sql
-- hive
with tmp as (
   select '[["100605096","113.54398085147338","26.75429268784589","25","42"],["100605521","113.54396563814427","26.75428509464191","52","88"]]' res
)
select cid, res_new from (
    select 999 as cid, split(SUBSTR(res,3,LENGTH(res)-4),'\\\],\\\[') res from tmp
)t
  LATERAL VIEW EXPLODE(t.res) tmptable as res_new
--  hive中的']'等特殊字符转义需要三个反斜杠（根据情况可能需要多个，在pyspark中sql就需要4个了）
```

#### 生成序列数 
```sql
-- hive
select tmp.*,t.* from 
(select 1 a)t LATERAL view   posexplode(split(space(8),' ')) tmp as idx,value

-- presto
select  
id, s from (  select 2 as id )
cross join 
UNNEST(SEQUENCE(0,10, 2)) as t ( s )
```
#### json格式解析
```sql
-- hive
get_json_object(json_col,'$.xxx') 
-- presto mysql
json_extract(json_col, '$[*].xxx')
```

#### 威尔逊区间平滑
```sql
  with smooth_tmp as (
      select city_id,
      block_id,
      station_id,
      window_id,
      day_of_week,
      (a-c)/d* window_cnt/window_consumption  eff_1d
      from (
          select *,
          order_cnt*1.0/window_cnt + 1.96*1.96/(2 * window_cnt) a,
          1.96 * sqrt((order_cnt*1.0/window_cnt) * (1-(order_cnt*1.0/window_cnt))/window_cnt + 1.96*1.96/(4*window_cnt*window_cnt) ) c,
          1+ 1.96*1.96/window_cnt d
          from tmp
      )x
  )
  select * from smooth_tmp
```

#### 防止小文件产生
```python
# distribute by
    sql_str = f"""
        insert into table ai.dws_ai_jw_physics_move_car_da_v2 partition(event_day = '{event_day}')
        select
        city_name,
        station_id,
        station_type,
        cur_date,
        city_id
        from
        ai.dws_ai_jw_physics_move_car_da where event_day = '{event_day}' distribute by cast(rand()*500 as int)"""
    spark.sql(sql_str)
 
# /*+REPARTITION(100)*/
    sql_str = f"""
        insert into table ai.dws_ai_jw_physics_move_car_da_v2 partition(event_day = '{event_day}')
        select /*+REPARTITION(100)*/
        city_name,
        station_id,
        station_type,
        cur_date,
        city_id
        from
        ai.dws_ai_jw_physics_move_car_da where event_day = '{event_day}' """
    spark.sql(sql_str)
```
### 编辑器属性设置

[概述](https://cloud.tencent.com/developer/article/1530056)

[函数](https://zhuanlan.zhihu.com/p/102502175)

动态分区设置

    set hive.exec.dynamic.partition.mode=nonstrict;   
    set hive.exec.dynamic.partition=true; 

hive任务减少小文件，Map-only的任务结束时合并小文件：

    set hive.merge.mapfiles = true
在Map-Reduce的任务结束时合并小文件：、

    set hive.merge.mapredfiles= true
当输出文件的平均大小小于1GB时，启动一个独立的map-reduce任务进行文件merge：

    set hive.merge.smallfiles.avgsize=1024000000



### grafana 可视化画图
```sql
— 画多边形区域
SELECT
  now() as time,
  h3_edge as pos,
  'polygon' as type,
  concat(
    '{"content": "order_sn:',
    cast(order_sn as VARCHAR),
    '", "option":{"strokeColor":"#fa3300","fillColor":"#ee2c2c","strokeWeight":5,"fillOpacity":0.3},"isStroke":true}'
  ) as config
from
  ai.dws_ai_order_visual_yf 
where
city_id = $city_id and
  event_day='20220124'

— 画制定半径的圆
select  now() as time, 'circle' as type, 20 as  radius,  split(start_point,'|')[1] as longitude,   split(start_point,'|')[2] as latitude,   concat('{"option":{"fillColor":', cast( (ln(10000)+1)*10  as varchar), '}, "isStroke":false }') as config  from  ai.dws_ai_order_visual_yf where  city_id = $city_id and   event_day='20220124'

— 画点
select  now() as time, 'Point' as type, '112.778399,32.134825' as pos 

— content显示字段内容
select now() as time, 'circle' as type, 10 as radius, split(point,'|')[1] as longitude, split(point,'|')[2] as latitude, concat('{"content": "station_id:', cast(station_id as VARCHAR ), ',' , 'block_id:', cast(block_id as VARCHAR ),',','order_cnt:',cast(order_cnt as VARCHAR ), ' ", "option":{"fillColor":', '"#0000FF"', '}, "isStroke":false, "fillOpacity":0 }') as config from ai.dws_ai_jw_station_dispatch_detail_da where event_day = '$event_day' and city_id = $city_id and is_dispatch=1 and is_blind=1 and status=0

- 画线 polyline
```
