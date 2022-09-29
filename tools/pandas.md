[中文文档](https://www.gairuo.com/p/pandas-tutorial)

### 基础设置

    pd.set_option('display.max_columns', None)  # 列
    pd.set_option('display.max_rows', None)  # 行
    pd.set_option('display.float_format', lambda x: '%.2f' % x)  # 不用科学计数

### 基础语法
    
    列改名
    city_info_df.rename(columns={'id':'city_id'},inplace=True)   

    填充缺失值,以下填写不会马上生效，需要重新赋值给 df 或者 inplace=Ture
    values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    df.fillna(value=values,inplace=Ture)  

    如果上述填充还不行可以用替换
    df = df.replace({'col': {np.nan: 100}})      
    
    指定连接方式和连接字段合并两表（默认left 保留左表全部行列+右表对应on的右表字段）
    df_id_name=pd.merge(station_df_227,city_info_df,how='left',on=['city_id’])

    指定列去重，只针对subset的指定的字段去除重复的
    df= block_cell_df.drop_duplicates(subset='block_id')   
    
    去除重复行，所有字段都一样的行,drop=True删除原先的index
    df=data.drop_duplicates().reset_index(drop=True)                                     
    
    指定列排序
    block_cell_df.sort_values(by='cell_id').head()                                              
    
    apply函数
    order_df['in_cell_id']=order_df.apply(lambda x:generate_new_cell_id(x['in_cell_id'],new_block_version),axis=1)
     

    将分组前同一个out_cell_id对应的多个out_cell_cnt字段组成一个字段
    test_df2=test_df.groupby('out_cell_id').out_cell_cnt.apply(lambda x:list(set(x))).reset_index()       

    更改字段类型 
    order_df_pd['start_block']=order_df_pd['start_block'].astype('int')

    分组对不同列做不同聚合操作
    df.groupby(['in_num']).agg({'mae':'mean' ,'rmse':'mean', 'prediction':'mean', 'mse':'count'}).reset_index()

    皮尔逊相关系数
    df.corr()

    把浮点数取整，然后逗号分隔格式化输出
    df[col] = df[col].apply(lambda x:'{:,}'.format(int(x)))     


### 其余总结

一、Apply函数 

可以用函数对一行的多列进行运算（必须指定axis=1）下面三个是等价的

    1. def func1(x,y):
                return x+y
        d1['d']=d1.apply(lambda x:func1(x['a'],x['b']),axis=1)
    2. d1['d']=d1.apply(lambda x:x['a’]+x['b'],axis=1)
    3. d1['d']=d1['a']+d1['b']

apply生成多列，匿名函数可以换成自定义函数

    df[['lon','lat']] = df.apply(lambda x:(float(x['start_point'].split('|')[0]),float(x['start_point'].split('|')[1])), axis=1,result_type='expand')
    
二、Map函数

用于对一列变换

    d1['e']=d1.d.map(lambda x:x+2)

三、Cut & qcut

Cut 需要传入需要分组的值的边界，一个列表

    d1['ccut']=pd.cut(d1.a,[i for i in range(0,11,2)])

Qcut 保证每个分组的数量是一致的，传入一个正整数，边界根据分组的值的数量和范围均分

    d1['qqcut']=pd.qcut(d1.a,5)

    # 如果分界点值不唯一则报错：Bin edges must be unique，添加等值元素rank不同即可解决
    d1['qqcut']=pd.qcut(d1.a.rank(method='first'),5)

e.g.有列A, B，A的值在1-100（含），对A列每10步长，求对应的B的和

    df = pd.DataFrame({'A': [1,2,11,11,33,34,35,40,79,99], 
                    'B': [1,2,11,11,33,34,35,40,79,99]})
    df1 = df.groupby(pd.cut(df['A'], np.arange(0, 101, 10)))['B'].sum()

四、Concat 

axis=0 增加行；axis=1 增加列

    pd.concat((df,df),axis=0)

五、分组排序取topN

    df = pd.DataFrame({'A': 'a a b b b b'.split(),
                    'B': [1,2,3,2,2,4],
                    'C': [4,6,5,3,1,5],
                        'D’:[2,2,2,2,2,2]})

(1) 实现分组排序取前n行（所有的组取同样的行数）

    tmp = df.sort_values(['A','B'],ascending=[1,0])    # 先排序（A升序，B降序）
    tmp.groupby('A').head(1)
    等价于
    df.groupby('A').apply(lambda t: t[t.B==t.B.max()]).reset_index(drop=True)  #这是取了个最大值，任意取数量级还是得上面方法

(2) 实现分组排序取各自召回数量的数据（各组不同行数,D列为数量）

    from functools import reduce
    lis = []
    for col in list(df.A.unique()):
        recall_length = int(df.D.unique())
        tmp = df[df['A']==col].sort_values('B',ascending=False)[:recall_length]
        lis.append(tmp)
        df1=reduce(lambda x,y: pd.concat((x,y),axis=0),lis)

六、分组后将需要的列组成列表

    df = pd.DataFrame({'A': 'a a b b b b'.split(),
                    'B': [1,2,3,2,2,4],
                    'C': [4,6,5,3,1,5]})
    g = df.groupby('A’)
    g.B.apply(lambda x:list(x)).reset_index() 等价于 g.apply(lambda x:list(x['B'])).reset_index() 


七、赋值

（1）取某些行列赋新值，改变原先df的值

    df.loc[df['A']=='b','D']=3
（2）将animal列中的snake替换为python

    df['animal'] = df['animal'].replace('snake', 'python')

八、基础的数值异常值去除

    ds = tmp.describe()
    for col in tmp.columns[:]:
        q1=ds.loc['25%',col]
        q3=ds.loc['75%',col]
        iqr = q3-q1
        print(q1-iqr,q3+iqr)
        tmp = tmp[tmp[col].between(q1-iqr,,q3+iqr)]

九、判断某列是否有空值

    df.isnull().any()

十、category 类型数据转int

    1.pd.get_dummies  会多n列
    2.dfp["new_int_col"] = dfp["category_col"].cat.codes  只多一列

十一、删除

（1）插入一条索引及该行对应的值

    df.loc['k'] = [5.5, 'dog', 'no', 2]
（2）删除索引（删一条/行数据）

    df = df.drop('k')
（2）删除列

    df.drop('col1',axis=1,inplace=True)

十二、shift

函数可以把数据移动指定的位数，period参数指定移动的步幅,可以为正为负.axis指定移动的轴,1为行,0为列

    df.shift(axis=0) # 数据在第一行多了一行nan，其余数据平移向下 ，默认axis=0
    df.shift(period=-1，axis=0) # 数据在最下面一行多了一行nan，
    df.shift(axis=1) # 数据在第一列多了一列nan，其余数据平移向右
最多用处求某列的前后行的差值

    tmp['nums'].shift() !=tmp['nums'] #可以去重复！等价于drop_duplicates
    df1 = df.drop_duplicates(subset='A’)

十三、求多列数据哪一列和/平均/最值/    最小/最大

    tmp[['nums','ne']].sum().idxmin()/idxmax()     #返回和最小/最大的那一列名

十四、求A列每个值的前3大/小的B的和

    df = pd.DataFrame({'A': list('aaabbcaabcccbbc'), 
                    'B': [12,345,3,1,45,14,4,52,54,23,235,21,57,3,87]})
    df.groupby(['A'])['B'].nlargest/nsmallest(3).sum(level=0).reset_index()

十四、计算每个元素至左边最近的0（或者至开头）的距离，生成新列y

    df = pd.DataFrame({'X': [7, 2, 0, 3, 4, 2, 5, 0, 3, 4]})
    # 方法二
    # x = (df['X'] != 0).cumsum()
    # y = x != x.shift()
    # df['Y'] = y.groupby((y != y.shift()).cumsum()).cumsum()

    # 方法三
    # df['Y'] = df.groupby((df['X'] == 0).cumsum()).cumcount()
    #first_zero_idx = (df['X'] == 0).idxmax()
    # df['Y'].iloc[0:first_zero_idx] += 1

十五、一个全数值的DataFrame，返回最大得3个值的坐标

    df = pd.DataFrame(np.random.random(size=(5, 3)))
    df.unstack().sort_values()[-3:].index.tolist()  #坐标返回的是（列，行）

十六、transform函数

此函数参数必须是个函数，可以是自定义的，也可以是内置的如下所示:

(1) 将vals负值代替为同grps组的平均值

1.1 用transform

    df = pd.DataFrame({'grps': list('aaabbcaabcccbbc'), 
                    'vals': [-12,345,3,1,45,14,4,-52,54,23,-235,21,57,3,87]})
    def replace(group):
        mask = group<0
        group[mask] = group.mean()
        return group
    df['vals'] = df.groupby(['grps'])['vals'].transform(replace)
1.2 正常都是用merge

    df2 = df.merge(df.groupby('grps').vals.mean().reset_index(),on=['grps'],how='inner')
    df2['vals_x']=df2.apply(lambda x:x['vals_y'] if x['vals_x']<0 else x['vals_x'],axis=1)
(2)按grps分组，求每行vals数据在各组中占比

    df['xx'] = df['vals']/df.groupby('grps')['vals'].transform('sum')

十七、滑动窗口函数

(1)计算3个长度的滑动窗口平均值

    df = pd.DataFrame({'value': [1, 2, 3, np.nan, 2, 3, np.nan, 1, 7, 3, np.nan, 8]})
    df = df.fillna(0)
    df.rolling(3,min_periods=1).mean()  #包括自身向上滑动3个求平均，如果最小周期不设置为1，那么前两个就有nan值，就会为nan
    df.rolling(3,min_periods=1,center=True).mean() #上下各一个，再加自身求平均
(2)计算group分组后的value列的3个长度的滑动窗口平均值，并且不计入nan值的行

    df = pd.DataFrame({'group': list('aabbabbbabab'),
                        'value': [1, 2, 3, np.nan, 2, 3, np.nan, 1, 7, 3, np.nan, 8]})
    print(df)

    g1 = df.groupby(['group'])['value’]     # 用于rolling中不计入nan值
    g2 = df.fillna(0).groupby(['group'])['value’] # 用于每行计算，不影响nan值在的行

    s = g2.rolling(3, min_periods=1).sum() / g1.rolling(3, min_periods=1).count()

    s.reset_index(level=0, drop=True).sort_index() 

十八、日期数据分析

（1）创建df，将2015所有工作日作为随机值的索引 

    dti = pd.date_range(start='2015-01-01', end='2015-12-31', freq='B’) # ‘B’工作日、’D’全部日
    ss = pd.DataFrame(np.random.rand(len(dti)), index=dti)

(2)星期等于周三的工作日求平均

    ss[ss.index.weekday==3].mean()

(3)求每个自然月的平均数

    ss.resample('M').mean()

(4)每连续4个月为一组，求最大值所在的日期

    ss.groupby(pd.Grouper(freq='4M')).idxmax()

(5)创建2015-2016每月第三个星期四的序列

    pd.date_range('2015-01-01', '2016-12-31', freq='WOM-3THU’)

十九、数据清洗

    df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm', 
                                'Budapest_PaRis', 'Brussels_londOn','China'],
                'FlightNumber': [10045, np.nan, np.nan, np.nan, 10085,np.nan],
                'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32],[20]],
                    'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )', 
                                '12. Air France', '"Swiss Air"',np.nan]})
(1)FlightNumber列中有些值缺失了，他们本来应该是每一行增加10，填充缺失的数值，并且令数据类型为整数

    df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int) 
    # 插值函数，默认为线性，会在操作列中为上下离nan最近的非nan值进行等分插值，最后或者最前的位置就是离其最近的非nan值
    
    df['Airline'].bfill().astype(str) 
    # 非整型或浮点型可用 bfill(backword)和ffill(forward)   取前后的插值函数
(2)将From_To分成from 和 to两列,删除原先的列

    tmp=df.From_To.str.split('_',expand=True)
    tmp.columns = ['From','To’]
    df= df.join(tmp)
    df = df.drop('From_To', axis=1)
(3)首字母大写

    df['To'] = df['To'].str.capitalize()
(4)Airline列，有一些多余的标点符号，需要提取出正确的航司名称。举例：'(British Airways. )' 应该改为 'British Airways’

    f['Airline'] = df['Airline'].str.extract('([a-zA-Z\s]+)', expand=False).str.strip() # 抽取字母+去除首位空格
(5) 将上述列表列展开为三列,缺失值用nan ,再给新的df换个名字

    delays = df['RecentDelays'].apply(pd.Series) # 操作nb
    delays.columns=['delay_{}'.format(i) for i in range(1,len(delays.columns)+1)]
    df = df.drop('RecentDelays', axis=1).join(delays)

二十、层次化索引

（1）用 letters = ['A', 'B', 'C'] 和 numbers = list(range(10))的组合作为系列随机值的层次化索引

    letters = ['A', 'B', 'C']
    numbers = list(range(4))

    mi = pd.MultiIndex.from_product([letters, numbers])
    s = pd.Series(np.random.rand(12), index=mi)
（2）选择二级索引为1, 3的行

    s.loc[:, [1, 3]]  #这里的逗号分隔的一级和二级索引，如果是dataframe loc逗号分隔的是行和列
(3)计算每个一级索引的和（A, B, C每一个的和）

    s.sum(level=0)

    #方法二
    #s.unstack().sum(axis=0)

(4)交换索引等级，新的Series是字典顺序吗？不是的话请排序

    new_s = s.swaplevel(0, 1)
    print(new_s)
    print(new_s.index.is_lexsorted())
    new_s = new_s.sort_index()
    print(new_s)

二十一、在同一个图中可视化2组数据，共用X轴，但y轴不同

    df = pd.DataFrame({"revenue":[57,68,63,71,72,90,80,62,59,51,47,52],
                    "advertising":[2.1,1.9,2.7,3.0,3.6,3.2,2.7,2.4,1.8,1.6,1.3,1.9],
                    "month":range(12)})

    ax=df.plot.bar("month", "revenue", color = "green")
    df.plot.line("month", "advertising", secondary_y = True,ax=ax)
    ax.set_xlim((-1,12))

二十二、min\max归一化

    def normalization(df):
        numerator = df.sub(df.min())
        denominator = (df.max()).sub(df.min())
        Y = numerator.div(denominator)
        return Y
    df = pd.DataFrame(np.random.random(size=(5, 3)))
    print(df)
    normalization(df)