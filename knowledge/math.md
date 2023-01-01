### 中心极限定理
#### 定义

&emsp;&emsp;中心极限定理指的是给定一个任意分布的总体。我每次从这些总体中随机抽取 n 个抽样，一共抽 m 次。 然后把这m组抽样分别求出平均值。 这些平均值的分布接近正态分布。

#### 应用

&emsp;&emsp;实际应用在实际生活当中，我们不能知道我们想要研究的对象的平均值，标准差之类的统计参数。中心极限定理在理论上保证了我们可以用只抽样一部分的方法，达到推测研究对象统计参数的目的。掷骰子这一行为的理论平均值3.5是我们通过数学定理计算出来的。而我们在实际模拟中，计算出来的样本平均值的平均值（3.48494）确实已经和理论值非常接近了。
简单地说，连续投掷多次色子，取这几次平均值，会发现比单独一次投掷更加接近平均（3.5）。
请比较一下几种情况：

- 投掷了一次，得到1，概率1/6
- 投掷了两次，平均为1，概率1/36
- 投掷了三次，平均为1，概率1/216

&emsp;&emsp;发现取平均值以后，达到极端值的概率更小，所以即使原始分布为平均，其取样平均值（把几次放在一起算平均值）的结果也会趋向于正态分布。

### 假设检验

https://www.zhihu.com/question/20254932?sort=created

### 概率密度函数

https://www.zhihu.com/question/263467674

通俗来说就是以随机变量x的取值为横轴，纵轴为随机变量对应的概率密度取值，对随机变量x某个区间求积分即表示在该区间的发生的概率，可以理解趋于某一点时，其概率密度值越大，发生的概率越大，如标准的正态分布在0附近取值的概率最大


#### 概念
&emsp;&emsp;在数学中，连续型随机变量的概率密度函数是一个描述这个随机变量的输出值，在某个确定的取值点附近的可能性的函数。而随机变量的取值落在某个区域之内的概率则为概率密度函数在这个区域上的积分。当概率密度函数存在的时候，累积分布函数是概率密度函数的积分。概率密度函数一般以小写标记。

#### 定义
&emsp;&emsp;对于一维实随机变量 $X$, 设它的累积分布函数是 $F_{X}(x)$, 如果存在可测函数 $f_{X}(x)$, 满 足: $F_{X}(x)=\int_{-\infty}^{x} f_{X}(t) d t$, 那么 $X$ 是一个连续型随机变量, 并且 $f_{X}(x)$ 是它的概率密度函数。

#### 性质
 - $f(x) \geq 0$
 - $\int_{-\infty}^{+\infty} f(x) d x=1$
 - $P(a<x \leq b)=\int_{a}^{b} f(x) d x$

#### 举例
&emsp;&emsp;最简单的概率密度函数是均匀分布的密度函数。对于一个取值在区间 $[a, b]$ 上的均匀分 布函数 $I_{[a, b]}$ ，它的概率密度函数： $f_{I_{[a, b]}}(x)=\frac{1}{b-a} I_{[a, b]}$
也就是说, 当 $x$ 不在区间 $[a, b]$ 上的时候, 函数值等于 0 ; 而在区间 $[a, b]$ 上的时候, 函数 值等于这个函数 $\frac{1}{b-a}$ 。这个函数并不是完全的连续函数, 但是是可积函数。

<img align="center"  width='500' height='300' src="picture/math_basic1.png"  />

&emsp;&emsp;正态分布是重要的概率分布。随着参数 $\mu$ 和 $\sigma$ 变化，概率分布也产生变化。它的概率密度函数是:
$$
f(x)=\frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{(x-\mu)^{2}}{2 \sigma^{2}}}
$$

&emsp;&emsp;对概率密度函数作傅里叶变换可得特征函数。特征函数与概率密度函数有一对一的关系。因此知道一个分布的特征函数就等同于知道一个分布的概率密度函数:
$$
\varnothing_{x}(j w)=\int_{-\infty}^{+\infty} f(x) e^{j w x} d x
$$



&emsp;&emsp;概率密度并不直接表示概率，就是概率密度只是表示概率的分布程度，就像一个小球1kg ，有一小块密度大，有一小块密度小，这一块的密度只是表示这里材料的密集度，并不表示质量。概率密度没有范围就想小球某一块的密度大小没有范围一样。就像某一块的密度可以很大，那其他块的密度就很小，但是总质量还是那么多。

### 期望

随机变量 $x\in X$

连续随机变量的期望为
$$
\mathbb{E}[f(X)]=\int_x p(x) \cdot f(x) d x
$$
离散随机变量的期望为
$$
\mathbb{E}[f(X)]=\sum_{x \in X} p(x) \cdot f(x)
$$

### 随机抽样

给定10个球：2红，5黄，3绿，当我从箱子里抽一个球，并且观测到了颜色，那么一次随机抽样就完成了，且有了一次观测值

给定球颜色概率：0.2红，0.5黄，0.3绿，假如我随机抽样100次，得到多次观测值

    import numpy as np
    np.ramdom.choice(['r','y','g'], size=100, p=[0.2,0.5,0.3])


### 最大期望算法（EM）

&emsp;&emsp;https://zhuanlan.zhihu.com/p/85236423

#### 概念
&emsp;&emsp;是一种迭代算法，用于含有隐变量的概率模型参数的极大似然估计（或最大后验概率估计）

&emsp;&emsp;通常，当模型的变量都是观测变量时，可以直接通过极大似然估计法，或者贝叶斯估计法估计模型参数。但是当模型包含隐变量时，就不能简单的使用这些估计方法。


### 极大似然估计

&emsp;&emsp;https://zhuanlan.zhihu.com/p/32480810

#### 理解
&emsp;&emsp;利用已知的样本结果，反推最有可能（最大概率）导致这样结果的参数值。

$$\widehat{\theta}_{M L E}=\arg \max _{\theta} P(D \mid \theta)$$

#### 举例
&emsp;&emsp;例如：一个麻袋里有白球与黑球，但是我不知道它们之间的比例，那我就有放回的抽取10次，结果我发现我抽到了8次黑球2次白球，我要求最有可能的黑白球之间的比例时，就采取最大似然估计法：
 我假设我抽到黑球的概率为p,那得出8次黑球2次白球这个结果的概率为：P(黑=8)=p^8*（1-p）^2,现在我想要得出p是多少啊，很简单，使得P(黑=8)最大的p就是我要求的结果，接下来求导的的过程就是求极值的过程啦。可能你会有疑问，为什么要ln一下呢，这是因为ln把乘法变成加法了，且不会改变极值的位置（单调性保持一致嘛）这样求导会方便很多~同样，这样一道题：设总体X 的概率密度为 已知 X1,X2..Xn是样本观测值，求θ的极大似然估计这也一样啊，要得到 X1,X2..Xn这样一组样本观测值的概率是P{x1=X1,x2=X2,...xn=Xn}=
f(X1,θ)f(X2,θ)…f(Xn,θ)  然后我们就求使得P最大的θ就好啦，一样是求极值的过程，不再赘述。

&emsp;&emsp;假设我们的抽样是理想正确的；
概率大的事件在一次观测中更容易发生（打猎问题）；
在一次观测中发生了的事件其概率应该大（身高问题）。

#### 步骤：
- 1.已知某分布的概率密度函数
- 2.计算其似然函数
- 3.似然函数取对数
- 4.求最值对应的参数
- 5.得到估计的参数


### 最大后验概率(MAP)

&emsp;&emsp;https://zhuanlan.zhihu.com/p/32480810

&emsp;&emsp;在贝叶斯统计学中，“最大后验概率估计”是后验概率分布的众数。利用最大后验概率估计可以获得对实验数据中无法直接观察到的量的点估计。它与最大似然估计中的经典方法有密切关系，但是它使用了一个增广的优化目标，进一步考虑了被估计量的先验概率分布。所以最大后验概率估计可以看作是规则化（regularization）的最大似然估计。

$$\begin{aligned} \hat{\theta}_{M A P} &=\arg \max _{\theta} P(\theta \mid D) \\ &=\arg \max _{\theta} P(D \mid \theta) P(\theta) \end{aligned}$$

&emsp;&emsp;(1) 频率学派: 存在唯一真值 $\theta$ 。举一个简单直观的例子-抛硬币, 我们用 $P(h e a d)$ 来表示硬币 的bias。抛一枚硬币100次, 有20次正面朝上, 要估计抛硬币正面朝上的bias $P(h e a d)=\theta$ 。在 频率学派来看, $\theta=20 / 100=0.2$, 很直观。当数据量趋于无穷时, 这种方法能给出精准的估 计; 然而缺乏数据时则可能产生严重的偏差。例如, 对于一枚均匀硬币, 即 $\theta=0.5$, 抛郑 次, 出现5次正面 (这种情况出现的概率是 $1 / 2^{\wedge} 5=3.125 \%$ ), 频率学派会直接估计这枚硬币 $\theta=1$, 出现 严重错误。

&emsp;&emsp;(2) 贝叶斯学派: $\theta$ 是一个随机变量, 符合一定的概率分布。在贝叶斯学派里有两大输入和一大输 出, 输入是先验 (prior)和似然 (likelihood), 输出是后验 (posterior)。先验, 即 $P(\theta)$, 指的是在 没有观测到任何数据时对 $\theta$ 的预先判断, 例如给我一个硬币, 一种可行的先验是认为这个硬币有 很大的概率是均匀的, 有较小的概率是是不均匀的; 似然, 即 $P(X \mid \theta)$, 是假设 $\theta$ 已知后我们观 察到的数据应该是什么样子的; 后验, 即 $P(\theta \mid X)$, 是最终的参数分布。

### 线性回归


#### 最小二乘法及几何意义

<img align='center' width='1000' height='500'  src="picture/LeastSquares1.png" />

#### 最小二乘法概率角度为高斯噪声的MLE

<img  align='center' width='1000' height='500'  src="picture/LeastSquares2.png" />

#### 正则化 L2岭回归 频率角度
<img align='center' width='1000' height='500' src="picture/L2_1.png" />

- 图中N>>p表示正常情况，非正常情况 N<<p 就会造成过拟合
- 相比不带正则化项的线性回归的损失函数，带正则的损失函数多了$\lambda I$的对角矩阵
- $X^{\top} X$是半正定矩阵，加上对角矩阵一定是正定的，所以一定可逆

#### 正则化 L2岭回归 贝叶斯角度
<img align='center' width='1000' height='500' src="picture/L2_2.png" />

- 不带正则的最小二乘估计等价于频率角度的噪声为高斯分布的极大似然估计
- 带正则项的最小二乘估计等价于贝叶斯角度的先验和噪声都为高斯分布的最大后验概率估计

### 贝叶斯平滑

[基本概念](https://blog.csdn.net/jinping_shi/article/details/78334362)

[简要步骤](https://blog.csdn.net/Elaine_DWL/article/details/97525596?spm=1001.2101.3001.6650.9&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-9.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-9.pc_relevant_default&utm_relevant_index=15)

### 威尔逊区间平滑（Wilson CTR）

CTR（Click-Through-Rate）即点击通过率，指广告的点击到达率。CTR是广告推荐系统中一项重要的衡量算法好坏的指标。

计算公式： CTR = 点击数 / 曝光数

由于原始CTR计算方式只考虑了相对值，没有考虑绝对值。即，没有考虑曝光的数值大小，在曝光少的情况下，计算出的CTR其实不可靠，样本充足的情况下，才能反应真实情况。

举个例子，有三个广告：

A：点击数 5 曝光数 10

B：点击数 50 曝光数 100

C：点击数 500 曝光数 1000

此三个广告的CTR 都是 0.5 ，但是按照实际表现，从置信的角度分析，应该是C > B > A，因为C的样本数更多，可信度更高。

为了衡量样本数对于 CTR 置信区间的影响，科学家们引入"威尔逊（Wilson）区间"的概念。公式如下：

$\frac{\hat{p}+\frac{z^{2}}{2 n}}{1+\frac{z^{2}}{n}} \pm \frac{z}{1+\frac{z^{2}}{n}} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}+\frac{z^{2}}{4 n^{2}}}$

根据实际情况取区间上下界

p —— 概率，即点击的概率，也就是 CTR；

n —— 样本总数，即曝光数；

z —— 在正态分布里， $\mu+z \times \sigma$ 会有一定的置信度。例如 z=1.96 ，就有 95% 的置信度,可查表看对应置信度。
Wilson区间的含义就是，就是指在一定置信度下，真实的 CTR 范围是多少。Wilson CTR修正的源码如下：
```python
import numpy as np

def walson_ctr(num_click, num_pv, z=1.96):
    p = num_click * 1.0 / num_pv
    n = num_pv
    
    A = p + z**2 / (2*n)
    B = np.sqrt(p * (1-p) / n + z**2 / (4*(n**2)))
    C = z * B
    D = 1 + z**2 / n

    ctr = (A - C) / D #取了区间下界
    return ctr

def wilson_score(pos, total):
    """
    威尔逊得分计算函数
    参考：https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    :param pos: 正例数
    :param total: 总数
    :param p_z: 正太分布的分位数，这里取95%置信区间，z值为1.96
    :return: 威尔逊得分
    """
    threshold = 10  # 修正阈值，当out大于此值时不用修正

    if pos / total > 1:
        pos = total

    if total > threshold:
        return pos / total

    p_z = 1.96
    pos_rat = pos * 1. / total * 1.  # 正例比率
    score = (pos_rat + (np.square(p_z) / (2. * total))
                - ((p_z / (2. * total)) * np.sqrt(4. * total * (1. - pos_rat) * pos_rat + np.square(p_z)))) / \
            (1. + np.square(p_z) / total)
    if score < 0:
        score = 0
    return float(score)

```

在电单车场景中，定义[车站效率=订单数量/车辆等待时长]反应车站好坏，此时有三个车站场景:

A：订单数 1 等待时长 0.1h

B：订单数 10 等待时长 1h

C：订单数 100 等待时长 10h

此时三个车站ABC效率都是10,但是按照实际表现，从置信的角度分析，应该是C > B > A，因为C的样本数更多，可信度更高。

因此，可将上述方法推广至此场景，由于wilson平滑的是概率，所以将车站效率拆分成

车站效率=(订单数量/等待车辆数)*(等待车辆数/车辆等待时长),我们对前一项括号中内容进行修正

### 朴素贝叶斯

https://zhuanlan.zhihu.com/p/164619896

### 泊松分布

描述单位时间（或空间）内随机事件发生的次数。如某一服务设施在一定时间内到达的人数

?>Pagerank（节点重要性）

?>Epsilon-greedy

有一定的概率对目标值y进行探索
```python
import numpy as np
res = y
is_explore = True if np.random.uniform()>0.5 else False
if is_explore:
    res = int(y * (1+coef))
```

### UCB（多臂老虎机）


### 梯度和导数
[概念1](https://zhuanlan.zhihu.com/p/377666441)

[概念2](https://www.cnblogs.com/tangjunjun/p/11649356.html)

    导数:指的是一元函数中，函数y=f(x)在某一点处沿x轴正方向的变化率。
    偏导数:指的是多元函数中，函数y=f(x1,x2,…,xn)在某一点处沿某一坐标轴（x1,x2,…,xn）正方向的变化率。
    方向导数：我们不仅要知道函数在坐标轴正方向上的变化率（即偏导数），而且还要设法求得函数在其他特定方向上的变化率。而方向导数就是函数在其他特定方向上的变化率。

    梯度的提出只为回答一个问题：
    　函数在变量空间的某一点处，沿着哪一个方向有最大的变化率？
    　梯度定义如下：
    　函数在某一点的梯度是这样一个向量，它的方向与取得最大方向导数的方向一致，而它的模为方向导数的最大值。
    　这里注意三点：
    　1）梯度是一个向量，即有方向有大小；
    　2）梯度的方向是最大方向导数的方向；
    　3）梯度的值是最大方向导数的值。

    导数与向量
    　提问：导数与偏导数与方向导数是向量么？
    　向量的定义是有方向（direction）有大小（magnitude）的量。
    　从前面的定义可以这样看出，偏导数和方向导数表达的是函数在某一点沿某一方向的变化率，也是具有方向和大小的。因此从这个角度来理解，我们也可以把偏导数和方向导数看作是一个向量，向量的方向就是变化率的方向，向量的模，就是变化率的大小。
    　那么沿着这样一种思路，就可以如下理解梯度：
    　梯度即函数在某一点最大的方向导数，函数沿梯度方向函数有最大的变化率。
    　
    梯度下降法
    　既然在变量空间的某一点处，函数沿梯度方向具有最大的变化率，那么在优化目标函数的时候，自然是沿着负梯度方向去减小函数值，以此达到我们的优化目标。
    　如何沿着负梯度方向减小函数值呢？既然梯度是偏导数的集合，同时梯度和偏导数都是向量，那么参考向量运算法则，我们在每个变量轴上减小对应变量值即可。
    
    总结
    1.导数定义： 导数代表了在自变量变化趋于无穷小的时候，函数值的变化与自变量的变化的比值。几何意义是这个点的切线。物理意义是该时刻的（瞬时）变化率。

    注意：在一元函数中，只有一个自变量变动，也就是说只存在一个方向的变化率，这也就是为什么一元函数没有偏导数的原因。
    （derivative）

    2.偏导数： 既然谈到偏导数，那就至少涉及到两个自变量。以两个自变量为例，z=f（x,y），从导数到偏导数，也就是从曲线来到了曲面。曲线上的一点，其切线只有一条。但是曲面上的一点，切线有无数条。而偏导数就是指多元函数沿着坐标轴的变化率。
    注意：直观地说，偏导数也就是函数在某一点上沿坐标轴正方向的的变化率。
    （partial derivative）

    3.方向导数: 在某点沿着某个向量方向上的方向导数，描绘了该点附近沿着该向量方向变动时的瞬时变化率。这个向量方向可以是任一方向。

    方向导数的物理意义表示函数在某点沿着某一特定方向上的变化率。
    注意：导数、偏导数和方向导数表达的是函数在某一点沿某一方向的变化率，也是具有方向和大小的。
    （directional derivative）

    4.梯度: 函数在给定点处沿不同的方向，其方向导数一般是不相同的。那么沿着哪一个方向其方向导数最大，其最大值为多少，这是我们所关心的，为此引进一个很重要的概念: 梯度。

    5.梯度下降
    在机器学习中往往是最小化一个目标函数 L(Θ)，理解了上面的内容，便很容易理解在梯度下降法中常见的参数更新公式：

    Θ = Θ − γ ∂ L ∂ Θ
    通过算出目标函数的梯度（算出对于所有参数的偏导数）并在其反方向更新完参数 Θ ，在此过程完成后也便是达到了函数值减少最快的效果，那么在经过迭代以后目标函数即可很快地到达一个极小值。

    6.In summary:
    概念 　　物理意义
    导数 　　函数在该点的瞬时变化率
    偏导数 　函数在坐标轴方向上的变化率
    方向导数 函数在某点沿某个特定方向的变化率
    梯度 　　函数在该点沿所有方向变化率最大的那个方向


### 最优运输

（最优传输、线性规划、整数规划、混合整数规划、车辆路径规划（VRP）等问题、清华大学运筹学第四版蓝皮书）

[OR-Tools](https://developers.google.com/optimization/lp)

[POT-doc](https://pythonot.github.io/index.html)
[POT-demo](https://towardsdatascience.com/hands-on-guide-to-python-optimal-transport-toolbox-part-1-922a2e82e621)

#### 背景
&emsp;&emsp;最优运输问题最早是由法国数学家加斯帕德·蒙日(Gaspard Monge)在19世纪中期提出，它是一种将给定质量的泥土运输到给定洞里的最小成本解决方案。这个问题在20世纪中期重新出现在坎托罗维奇的著作中，并在近些年的研究中发现了一些令人惊讶的新进展，比如Sinkhorn算法。最优运输被广泛应用于多个领域，包括计算流体力学，多幅图像之间的颜色转移或图像处理背景下的变形，计算机图形学中的插值方案，以及经济学、通过匹配和均衡问题等。此外，最优传输最近也引起了生物医学相关学者的关注，并被广泛用于单细胞RNA发育过程中指导分化以及提高细胞观测数据的数据增强工具，从而提高各种下游分任务的准确性和稳定性。

&emsp;&emsp;当前，许多现代统计和机器学习问题可以被重新描述为在两个概率分布之间寻找最优运输图。例如，领域适应旨在从源数据分布中学习一个训练良好的模型，并将该模型转换为采用目标数据分布。另一个例子是深度生成模型，其目标是将一个固定的分布，例如标准高斯或均匀分布，映射到真实样本的潜在总体分布。在最近几十年里，OT方法在现代数据科学应用的显著增殖中重新焕发了活力，包括机器学习、统计和计算机视觉。
#### Wasserstein距离
[含义](https://zhuanlan.zhihu.com/p/58506295)

&emsp;&emsp;距离度量是机器学习任务中最重要的一环。比如，常见的人工神经网络的均方误差损失函数采用的就是熟知的欧式距离。然而，在最优运输过程中，优于不同两点之间均对应不同的概率，如果直接采用欧式距离来计算运输的损失（或者说对运输的过程进行度量和评估），则会导致最终的评估结果出现较大的偏差(即忽略了原始不同点直接的概率向量定义)。

&emsp;&emsp;Wasserstein distance 是在度量空间 M，定义概率分布之间距离的距离函数。

&emsp;&emsp;KL散度用于衡量分布间的差异程度，又称为相对熵，信息增益。KL散度不满足距离的定义：因为其不对称、不满足三角不等式。如果在高维空间两个分布不重叠或者重叠部分可忽略，则KL和JS散度反应不了远近，只是一个常量。

#### 定义
#### 1.Optimal Transport
$$
\begin{array}{r}
\gamma^{*}=\arg \min _{\gamma \in \mathbb{R}_{+}^{m \times n}} \sum_{i, j} \gamma_{i, j} M_{i, j} \\
\text { s.t. } \gamma 1=a ; \gamma^{T} 1=b ; \gamma \geq 0
\end{array}
$$
#### 2.Regularized Optimal Transport
$$
\begin{array}{r}
\gamma^{*}=\arg \min _{\gamma \in \mathbb{R}_{+}^{m \times n}} \sum_{i, j} \gamma_{i, j} M_{i, j}+\lambda \Omega(\gamma) \\
\text { s.t. } \gamma 1=a ; \gamma^{T} 1=b ; \gamma \geq 0
\end{array}
$$
#### 3.Entropic regularized OT
$$
\Omega(\gamma)=\sum_{i, j} \gamma_{i, j} \log \left(\gamma_{i, j}\right)
$$
著名的Sinkhorn算法用的就是熵正则，此形式使得问题严格凸，因此有唯一解，最终优化问题可被形式化为：
$$
\gamma_{\lambda}^{*}=\operatorname{diag}(u) K \operatorname{diag}(v)
$$
其中u和v为向量，$K=\exp (-M / \lambda)$

#### 4.Partial optimal transport

$$\begin{aligned} \gamma=\arg \min _{\gamma}<\gamma, M &>_{F} \\ \text { s. } t . \gamma & \geq 0 \\ \gamma 1 & \leq a \\ \gamma^{T} 1 & \leq b \\ 1^{T} \gamma^{T} 1=m \leq \min \left\{\|a\|_{1},\|b\|_{1}\right\} \end{aligned}$$



### A/B测试

[ABTest流量分层机制](https://zhuanlan.zhihu.com/p/359668457)

[ABTest流量分层机制2](https://blog.csdn.net/weixin_39925350/article/details/111373181)

**假设检验：假设和检验**

假设：
H0:原假设：A=B 
H1:备择（对立）假设:A!=B

检验：从总体中抽样部分观测结果。会有两种情况：抽样误差（偶然）、本质差异（必然）

比如我要计算全世界男生和女生平均身高是否相等？但是我们不可能实现统计全部男女身高，就需要抽样统计了，通过样本反应总体
原假设：男高=女高;对立假设：男高!=女高

样本与总体可能是不一样的。
假如我的抽样情况是：男高!=女高，就会有上述两种情况：
（1）实际上原假设是成立的，而我们由于抽样误差导致男女平均身高不一致
（2）实际上原假设是错误的，男女平均身高就是不一样，必然发生的

所以我们需要评估用样本估计总体这一操作的可信程度，引入p值
p:H0原假设成立的情况下，拒绝H0的概率（个人理解为冤枉H0的概率、拒绝原假设犯第一类错误的最小概率），也就是因为抽样误差导致的

$\alpha$：叫做显著性（差异）水平，使得犯第一类错误的概率控制在一定水平下，常取一个较小的数0.05，0.01
当取0.05时，我们计算的p<=0.05时候，说明我们没冤枉H0原假设，说明没犯第一类错误，说明这不是抽样误差，说明男女平均身高就是不一样，说明原假设H0错误，对立假设H1成立(落在了显著性差异的那部分了)
当p>0.05时候，说明我们冤枉了H0,说明男女身高实际是一样的，但犯了一类错误，这是抽样误差导致的，说明原假设H0成立(落在了置信区间的那部分了)

**总结**
P是抽样抽取的极端，而不是事件发生的极端。所以p越小，抽样越不极端，差异就越必然，则拒绝原假设H0
P值就是抽样误差（偶然）发生的概率，如果小于显著性水平（0.05）的话则认定为小概率事件，不会发生，则是本质差异（必然），而样本的发生与原假设是相反的，所以原假设不成立

P值是通过Z检验或者T检验或者其他表格找出来的概率，要得到这个概率就得先算z值或者t值，然后在对应的表里找

    Φ（1.96）=P{（X-μ)/σ<1.96}=0.975
    P(X<=x)=Φ[（X-μ)/σ]
    P(X>x)=1-Φ[（X-μ)/σ]
    P(|X|>x)=2*{1-Φ[（X-μ)/σ]}
    P(|X-μ|/σ>1.96)=2*(1-0.975)=2*0.025=0.05
    比如μ=0,σ=10时，P（|X|>19.6）=P（|X-0|/10>1.96）=2*[1-Φ（1.96）]=0.05



[详细](https://zhuanlan.zhihu.com/p/346602966)

[面试](https://zhuanlan.zhihu.com/p/487824153)

[面试2](https://blog.csdn.net/garbageSystem/article/details/122603832?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4-122603832-blog-108114448.pc_relevant_3mothn_strategy_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4-122603832-blog-108114448.pc_relevant_3mothn_strategy_recovery&utm_relevant_index=5)

[hash分流](https://blog.csdn.net/weixin_38753213/article/details/108114448)

[python-hash](https://blog.csdn.net/weixin_39588206/article/details/110909007?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-110909007-blog-88015355.pc_relevant_aa&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-110909007-blog-88015355.pc_relevant_aa&utm_relevant_index=4)

### embedding
个人理解：是一个lookup table 查表的过程，随机初始化表，最后梯度下降反传的时候的时候就优化更新了参数表，使得能够表达相似的样本间的向量

作用：
- 高维稀疏特征向量到低维稠密特征向量的转换；
- 嵌入层将正整数（下标）转换为具有固定大小的向量；
- 把一个one hot向量变为一个稠密向量

Embedding主要的三个应用方向：
- 1、在深度学习网络中作为Embedding层，完成从高维稀疏特征向量到低维稠密特征向量的转换；
- 2、作为预训练的Embedding特征向量，与其他特征向量连接后一同输入深度学习网络进行训练；
- 3、通过计算用户和物品的Embedding相似度，Embedding可以直接作为推荐系统或计算广告系统的召回层或者召回方法之一。

[代码实践](https://blog.csdn.net/weixin_42357472/article/details/120886559?spm=1001.2101.3001.6650.9&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-9-120886559-blog-124574439.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-9-120886559-blog-124574439.pc_relevant_default&utm_relevant_index=15)

### 交叉熵

[简要](https://blog.csdn.net/weixin_42078618/article/details/81736329#comments_24006134)

[详细](https://blog.csdn.net/rtygbwwwerr/article/details/50778098?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-50778098-blog-81736329.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-50778098-blog-81736329.pc_relevant_default&utm_relevant_index=4)

1.熵

$H(X)=-\sum_{i=1}^n p\left(x_i\right) \log \left(p\left(x_i\right)\right)$

- 指的就是信息量的大小，越有可能发生的事情，信息量越小，熵值越小，越不可能发生的事情，信息量越大，熵值越大。
- 本质上熵是信息量的期望，他的大小跟信息量的大小一定程度上成正相关。
- 公式表示为事件所有发生的概率乘以对应的概率的对数的总和，再取负数
- 为什么公式这样表达呢？因为取对数防止造成熵值的不必要波动，比如1和1亿这么一比较方差很大，但是取对数，方差很小
- 那为什么取负数呢？因为概率恒小于1,取对数恒小于0，为了方便观察对终值取负

2.相对熵(KL散度)

相对熵及KL散度，KL距离，是两个随机分布间距离的度量，它度量当真实分布为p时，假设分布q的无效性
一个算法中我们定义的有钱人和穷人的标准，跟机器定义的有钱人和穷人的标准，就有了一个gap，这个gap就是广义上的相对熵，即：预测值跟实际值的差距。如下是相对熵的数学定义

$D_{K L}(p \| q)=\sum_{i=1}^n p\left(x_i\right) \log \left(\frac{p\left(x_i\right)}{q\left(x_i\right)}\right)$

可以推导为

$\begin{aligned} D_{K L}(p \| q) &=\sum_{i=1}^n p\left(x_i\right) \log \left(p\left(x_i\right)\right)-\sum_{i=1}^n p\left(x_i\right) \log \left(q\left(x_i\right)\right) \\ &=-H(p(x))+\left[-\sum_{i=1}^n p\left(x_i\right) \log \left(q\left(x_i\right)\right)\right] \end{aligned}$

发现第一项是恒定不变的，第二项就被定义为交叉熵


### 运筹优化

[blog](https://blog.csdn.net/kittyzc/category_7940150_2.html)

### 因果推断

- [综述及基础方法](https://zhuanlan.zhihu.com/p/258562953)
- [综述](https://zhuanlan.zhihu.com/p/362788755)
- [双重差分](https://zhuanlan.zhihu.com/p/400085535)
- [概念](https://zhuanlan.zhihu.com/p/449976773)
- [框架](https://zhuanlan.zhihu.com/p/410053669)
- [滴滴](https://blog.csdn.net/DiDi_Tech/article/details/117137317?spm=1001.2014.3001.5501)
- [因果发展](http://www.360doc.com/content/21/0831/23/60669552_993570605.shtml)
- [概念](https://www.cnblogs.com/caoyusang/p/13518354.html)
  
因果推断在因果效应（Causal Effect）的应用层面通常涉及三个主题：
- 群体因果效应估计
- 个体因果效应估计
- 因果关系的发现
  
#### 双重差分法DID
双重差分法DID是群体因果效应估计的一个重要方法，用于协助我们找到干预价值高的群体（并进行后续定向干预）

一.相关名词：

- 1.观察学习（Observational Study）：因为伦理问题/客观限制，样本中的因变量（Treatment）不受研究者的控制，基于这样的样本进行的推导分析
- 2.自然实验（Natural experiment）：个体被（非观察者可控制的因素）暴露在试验或控制条件下的一种实验研究方法，本质上是一种观察实验。
- 3.面板数据（Panel Data）：截面数据与时间序列数据综合起来的一种数据类型，如下图所示：一条线（一年）为时序数据、横切面(具体某一天)为截面数据。

二、概念

双重差分法（Difference in Differences）:
通过利用观察学习的数据，计算自然实验中“实验组”与“对照组”在干预下增量的差距。即：实验组干预前后的均值的差减去对照组干预前后均值的差

三、步骤

- 分组：对于一个自然实验，其将全部的样本数据分为两组：一组是受到干预影响，即实验组；另一组是没有受到同一干预影响，即对照组；
- 目标选定：选定一个需要观测的目标指标，如购买转化率、留存率，一般是希望提升的KPI；
- 第一次差分：分别对在干预前后进行两次差分（相减）得到两组差值，代表实验组与对照组在干预前后分别的相对关系；
- 第二次差分：对两组差值进行第二次差分，从而消除实验组与对照组原生的差异，最终得到干预带来的净效应。

四、需满足的假设

前两个假设使用时通常会满足、无需专门验证，需要重点验证第三个假设

4.1线性关系假设
- 该假设来自于线性回归，认为因变量（Treatment）与结果变量存在线性关系
  
4.2个体处理稳定性假设（The Stable Unit Treatment Value Assumption，SUTVA）

- 个体的outcome是取决于个体干预变量treatment的一个函数，该假设由两部分组成
- 一致性（Consistency）：个体接受处理后所导致的潜在结果是唯一的。
例：我养狗了会变开心，无论是什么狗、不存在因为狗是黑的就不开心
- 互不干预（No interference）：个体接受处理后导致的潜在结果不受其他个体处理的影
例：我在淘宝上领到了红包之后会更愿意买东西，不因为我同事也领了红包就意愿降低了

4.3平行趋势假设（Parallel Trend Assumption）
- 实验组和对照组在没有干预的情况下，结果的趋势是一样的。即在不干预的情况下，前后两个时间点实验组与对照组的差值一致。
- 检验方式：通常情况下我们可以通过画图或者按照定义计算的方式验证样本是否满足假设
- 这个假设在随机实验下，通常是满足的，因为两批用户是很近似且同质的用户。但在观察实验的情景下，有可能会不满足，此时不能简单粗暴、不加处理的直接使用DID，需要对数据进行处理。当前常用的处理方式由如下三种：倾向得分匹配（Propensity Score Matching，PSM）、三重差分法（Difference-in-differences-in-differences, DDD）、合成控制法（Synthetic Control Method）。可以理解为人工构建相对同质的实验组和对照组的方法。

五、同质化人/物群构建方法

通过构建相对的同质人群使满足平行趋势假设

5.1倾向得分匹配（Propensity Score Matching，PSM）

相对于后两种方式，该方法在工业界更常用。
- 目的：从干预的人群和未干预的人群里找到两批人符合平行趋势假设
- 业务理解：在这两个人群里找个两批同质的人（该场景下的同质：在treatment维度上有近似表现的人）
- 例子：在探究领取红包对用户购买行为影响的场景下，对用户领取红包的倾向做预测（打分），认为分数相近的用户是matching、即同质的。圈选出分数相同的用户之后再验证平行趋势假设。

完成PSM后数据会呈现一些规律（如图所示）：

- 干预人群与非干预人群的score分布 —— 匹配后分布一致
- 抽样后人群在一些画像（如年龄、性别、职业）上的分布会更接近

#### 弹性模型评估指标
[1](https://zhuanlan.zhihu.com/p/363082639)

[2](https://zhuanlan.zhihu.com/p/457689388)

[3](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247552286&idx=5&sn=f5263dcc1cbc4b63acc47cf46cbe81c2&chksm=ebb735cadcc0bcdc998f647cc3cd61e74df9820cb68de5fd199785053e271763de0e28a5f2e5&scene=27)

[4](https://zhuanlan.zhihu.com/p/399322196)

### 临时写公式

$y=\omega_{0}+\sum_{i=1}^{n} \omega_{i} x_{i}+\sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \omega_{ij} x_{i} x_{j}$

$obj=max(\sum_{p=1}^{n}heat_{p})$

s.t. 
$distance(p_{ij}>50)$


$max \sum_{i=1}^m \sum_{j=1}^{n} f(s_i,t_j,target\_bike\_cnt_i\\_j) * target\_bike\_cnt_i\\_j $

$s.t.\quad  min\_cnt_i\\_j <= target\_bike\_cnt_i\\_j <= max\_cnt_i\\_j $

$s.t.\quad  \sum target\_bike\_cnt_i\\_j < usable\_bike\_cnt$


$$
Z=\frac{\bar{X}-\mu}{\sqrt{\sigma^2 / n}}=\frac{\bar{X}-\mu}{\sigma / \sqrt{n}}
$$

$$
Z=\frac{\left(\bar{x}_1-\bar{x}_2\right)-\left(u_1-u_2\right)}{\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}} \sim N(0,1)
$$