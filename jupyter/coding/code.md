### 基于快排的topK问题模版


```python
def partition(nums, left, right):
    pivot = nums[left]#初始化一个待比较数据
    i,j = left, right
    while(i < j):
        while(i<j and nums[j]>=pivot): #从后往前查找，直到找到一个比pivot更小的数
            j-=1
        nums[i] = nums[j] #将更小的数放入左边
        while(i<j and nums[i]<=pivot): #从前往后找，直到找到一个比pivot更大的数
            i+=1
        nums[j] = nums[i] #将更大的数放入右边
    #循环结束，i与j相等
    nums[i] = pivot #待比较数据放入最终位置 
    return i #返回待比较数据最终位置

#快速排序
def quicksort(nums, left, right):
    if left < right:
        index = partition(nums, left, right)
        quicksort(nums, left, index-1)
        quicksort(nums, index+1, right)
        
print('===快排===')
arr = [1,3,2,2,0]
quicksort(arr, 0, len(arr)-1)
print(arr) 


def topk_split(nums, k, left, right):
    # 寻找下标为k的位置，找到之后停止递归，下标k实际是确定了第k+1个位置，因为下标从0开始（如k=2,找前2个小的数，那么相当于确定了第2+1个数的位置，返回nums[:k]即可）
    #寻找到下标k的位置停止递归，使得nums数组中下标k左边是前k个小的数，index右边是后面n-k-1个大的数,k为第k+1小数字
    if (left<right):
        index = partition(nums, left, right)
        if index==k:
            return 
        elif index < k:
            topk_split(nums, k, index+1, right)
        else:
            topk_split(nums, k, left, index-1)
            
#获得前k小的数
def topk_smalls(nums, k):
    topk_split(nums, k, 0, len(nums)-1)
    return nums[:k]
print('===获得前k小的数,k=2===')
arr = [1,3,2,3,0,-19]
k = 2
print(topk_smalls(arr, k))
print(arr)


#获得第k小的数
def topk_small(nums, k):
    topk_split(nums, k, 0, len(nums)-1)
    return nums[k] 

print('===获得第k小的数,k=3===')
arr = [1,3,2,3,0,-19]
k = 3
print(topk_small(arr, k-1))
print(arr)


#获得前k大的数 
def topk_larges(nums, k):
    #parttion是按从小到大划分的，如果让index左边为前n-k个小的数，则index右边为前k个大的数
    topk_split(nums, len(nums)-k, 0, len(nums)-1) #把k换成len(nums)-k
    return nums[len(nums)-k:] 

print('===获得前k大的数,k=3===')
arr = [1,3,-2,3,0,-19]
k = 3
print(topk_larges(arr, k))
print(arr)

#获得第k大的数 
def topk_large(nums, k):
    #parttion是按从小到大划分的，如果让index左边为前n-k个小的数，则index右边为前k个大的数
    topk_split(nums, len(nums)-k, 0, len(nums)-1) #把k换成len(nums)-k
    return nums[len(nums)-k] 

print('===获得第k大的数,k=3===')
arr = [1,3,-2,3,0,-19]
k = 3
print(topk_large(arr, k))
print(arr)

#只排序前k个小的数
#获得前k小的数O(n)，进行快排O(klogk)
def topk_sort_left(nums, k):
    topk_split(nums, k, 0, len(nums)-1) 
    topk = nums[:k]
    quicksort(topk, 0, len(topk)-1)
    return topk+nums[k:] #只排序前k个数字

print('===只排序前k个小的数,k=4===')
arr = [0,0,1,3,4,5,0,7,6,7]
k = 4
print(topk_sort_left(arr, k))

#只排序后k个大的数
#获得前n-k小的数O(n)，进行快排O(klogk)
def topk_sort_right(nums, k):
    topk_split(nums, len(nums)-k, 0, len(nums)-1) 
    topk = nums[len(nums)-k:]
    quicksort(topk, 0, len(topk)-1)
    return nums[:len(nums)-k]+topk #只排序后k个数字

print('===只排序后k个大的数,k=4===')
arr = [0,0,1,3,4,5,0,-7,6,7]
k = 4
print(topk_sort_right(arr, k))

```

    ===快排===
    [0, 1, 2, 2, 3]
    ===获得前k小的数,k=2===
    [-19, 0]
    [-19, 0, 1, 3, 2, 3]
    ===获得第k小的数,k=3===
    1
    [-19, 0, 1, 3, 2, 3]
    ===获得前k大的数,k=3===
    [1, 3, 3]
    [-19, 0, -2, 1, 3, 3]
    ===获得第k大的数,k=3===
    1
    [-19, 0, -2, 1, 3, 3]
    ===只排序前k个小的数,k=4===
    [0, 0, 0, 1, 3, 4, 5, 7, 6, 7]
    ===只排序后k个大的数,k=4===
    [-7, 0, 0, 1, 0, 3, 4, 5, 6, 7]


#### 215 数组中的第K个最大元素


```python
import random
def findKthLargest(self, nums: List[int], k: int) -> int:
    def partition(nums,left,right):
        pivot = nums[left]
        l,r = left,right 
        while l<r:
            while l<r and nums[r]>=pivot:
                r-=1
            nums[l]=nums[r]
            while l<r and nums[l]<=pivot:
                l+=1
            nums[r] = nums[l]
        nums[l] = pivot
        return l 
    def randomchoice(nums,left,right): # 渐进O(n)
        ridx = random.randint(left,right)
        nums[left],nums[ridx] = nums[ridx],nums[left]
        return partition(nums,left,right)
    def topk_split(nums,left,right,k):
        if left<right:
            # idx = partition(nums,left,right)
            idx = randomchoice(nums,left,right)
            if idx==k:
                return 
            elif idx<k:
                topk_split(nums,idx+1,right,k)
            else:
                topk_split(nums,left,idx-1,k)
    topk_split(nums,0,len(nums)-1,len(nums)-k)
    print(nums[len(nums)-k])
    return nums[len(nums)-k]
```

### 二分搜索问题模板


```python
# 数组非递减有序排列

# 找目标值下标
def findtarget(nums,target):
    l,r = 0,len(nums)-1
    while l<=r:
        mid = l+(r-l)//2
        if nums[mid]>target:
            r = mid-1
        elif nums[mid]<target:
            l = mid+1
        elif nums[mid]==target:
            return mid
    return -1

# 找目标值左边界
def findl(nums,target):
    l,r = 0,len(nums)-1
    while l<=r:
        mid = l+(r-l)//2
        if nums[mid]>target:
            r = mid-1
        elif nums[mid]<target:
            l = mid+1
        elif nums[mid]==target: # 找左，收缩右边界
            r = mid -1
    if l>=len(nums) or nums[l]!=target: # 找左，判断左是否在最右边越界
        return -1
    return l 

# 找目标值右边界
def findr(nums,target):
    l,r = 0,len(nums)-1
    while l<=r:
        mid = l+(r-l)//2
        if nums[mid]>target:
            r = mid-1
        elif nums[mid]<target:
            l = mid+1
        elif nums[mid]==target: # 找右，收缩左边界
            l = mid +1
    if r<0 or nums[r]!=target:  # 找右，判断右是否在最左边越界
        return -1
    return r 
```

#### 34 在排序数组中查找元素的第一个和最后一个位置


```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def findl(nums,target):
            l,r = 0,len(nums)-1
            while l<=r:
                mid = l+(r-l)//2
                if nums[mid]>target:
                    r = mid-1
                elif nums[mid]<target:
                    l = mid+1
                elif nums[mid]==target:
                    r = mid -1
            if l>=len(nums) or nums[l]!=target:
                return -1
            return l 
        def findr(nums,target):
            l,r = 0,len(nums)-1
            while l<=r:
                mid = l+(r-l)//2
                if nums[mid]>target:
                    r = mid-1
                elif nums[mid]<target:
                    l = mid+1
                elif nums[mid]==target:
                    l = mid +1
            if r<0 or nums[r]!=target:
                return -1
            return r 
        return [findl(nums,target),findr(nums,target)]
```

#### 35 插入位置搜索


```python
# 正常二分，返回左边界l即可

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        l,r = 0,len(nums)-1

        while l<=r:
            mid = l+(r-l)//2
            if nums[mid]>target:
                r = mid-1
            elif nums[mid]<target:
                l = mid+1
            else:
                return mid
        return l

   
```

#### 153 寻找旋转排序数组的最小值


```python

def findMin( nums) -> int:

    l,r = 0,len(nums)-1
    while l<=r:
        mid = l+(r-l)//2
        # 右边界比中间大说明断点在左侧，收缩右
        if nums[mid]<nums[r]:
            r = mid 
        # 其他之外收缩左
        else:
            l=mid+1
    return nums[mid]

nums = [3,4,5,1,2]
findMin(nums)
```




    1



#### 搜索旋转排序数组


```python
#  先用上面的找到最小值的下标，分开两数组判断target,最后右数组找到了加上刚找到前面的下标即为所求
def search(self, nums: List[int], target: int) -> int:


    l,r = 0,len(nums)-1
    while l<=r:
        mid = l+(r-l)//2
        if nums[mid]<nums[r]:
            r = mid 
        else:
            l=mid+1
    idx = mid 
    left = nums[:idx]
    right = nums[idx:]
    print(left,right,idx)

    def binarysearch(nums,target):
        l,r = 0,len(nums)-1
        while l<=r:
            mid = l+(r-l)//2
            if nums[mid]<target:
                l = mid+1
            elif nums[mid]>target:
                r =mid-1
            else:
                return mid 
        return -1 
    r1 = binarysearch(left,target)
    r2 = binarysearch(right,target)
    if r1 !=-1:return r1 
    if r2 !=-1:return r2+idx 

    return r1

```

### 双指针问题
#### 283 移动0


```python
def moveZeroes(nums) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """

    l,r= 0,0
    n=len(nums)

    while r<n:
        # 当前不为0就把它移到前面，l+1
        if nums[r] !=0:
            nums[r],nums[l] = nums[l],nums[r]
            l+=1
        # 遍历r
        r+=1
nums=[0,1,2,4,0,7]
moveZeroes(nums)
nums
```




    [1, 2, 4, 7, 0, 0]



#### 11 盛最多水的容器


```python
def maxArea( height) -> int:
    n = len(height)
    l,r = 0,n-1
    res = 0
    while l<r:
        if height[l]>height[r]:
            res=max(res,height[r]*(r-l))
            r-=1
        else:
            res=max(res,height[l]*(r-l))
            l+=1
    return res
h = [1,8,6,2,5,4,8,3,7]
maxArea(h)
```




    49



#### 15 三数之和


```python
def threeSum(nums):
    n =len(nums)
    if n<3:return
    nums.sort()
    res = []
    for i in range(n-2):
        if nums[i]+nums[i+1]+nums[i+2]>0:break
        if nums[i]+nums[n-1]+nums[n-2]<0:continue
        if i>0 and nums[i]==nums[i-1]:continue
        l,r = i+1,n-1
        while l<r:
            tmp = nums[i]+nums[l]+nums[r]
            if tmp==0:
                res.append([nums[i],nums[l],nums[r]])
                while l+1<r and nums[l]==nums[l+1]:
                    l+=1
                while l<r-1 and nums[r]==nums[r-1]:
                    r-=1
                l+=1
                r-=1
            elif tmp<0:
                l+=1
            else:
                r-=1
    return res
nums = [-1,0,1,2,-1,-4]   
threeSum(nums)   
```




    [[-1, -1, 2], [-1, 0, 1]]



### 单调栈
#### 42 接雨水


```python
def trap(height) -> int:
    res= 0 
    # 单调递减栈
    stack =[]
    # 遍历柱子高度列表
    for i in range(len(height)):
        # 栈不空且当前遍历的高度比栈最后一个高，说明能存水了
        while stack and height[stack[-1]]<height[i]:
            # 出栈
            cur = stack.pop()
            # 如果栈里没数了，说明左边没边界了，续不了水
            if not stack:
                break
            # 左右边界找个矮的高- 底部的高度= 存水高度
            h = min(height[i],height[stack[-1]])-height[cur]
            # 右边界下标-左下标-1 = 存水宽度
            w = i-stack[-1]-1
            # 累计
            res+=h*w
        stack.append(i)
    return res
height = [4,2,0,3,2,5]
trap(height)
```




    9



#### 84 柱状图中最大矩形


```python
def largestRectangleArea( heights) -> int:

    res = 0
    stack =[]
    heights.insert(0,0)
    heights.append(0)
    # 递增栈，遇到比栈里小的弹出计算最大面积
    for i in range(len(heights)):
        while stack and heights[stack[-1]]>heights[i]:
            cur = stack.pop()
            l = stack[-1]+1
            r = i-1
            res = max(res,(r-l+1)*heights[cur])
        stack.append(i)
    return res
heights = [2,1,5,6,2,3]
largestRectangleArea(heights)
```




    10



### 矩阵
#### 54. 螺旋矩阵


```python
def spiralOrder(matrix):

    res=[]
    while matrix:
        res+= matrix[0]
        matrix = list(zip(*matrix[1:]))[::-1]

    return res
matrix = [[1,2,3],[4,5,6],[7,8,9]]
spiralOrder(matrix)
```




    [1, 2, 3, 6, 9, 8, 7, 4, 5]



#### 240 搜索二维矩阵


```python
def searchMatrix(matrix, target: int) -> bool:
    i,j = len(matrix)-1,0
    
    # 从左下角查找，注意指针越界条件
    while i>=0 and j<len(matrix[0]):
        if matrix[i][j]==target:
            return True
        elif matrix[i][j]<target:
            j+=1
        else:
            i-=1
    return False
matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
target = 5
searchMatrix(matrix,target)
```




    True



#### 48 旋转图像



```python
def rotate(matrix) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """

    m = list(zip(*matrix))
    for r in range(len(matrix)):
        matrix[r] = m[r][::-1]
```




    [(1, 4, 7), (2, 5, 8), (3, 6, 9)]



### 技巧
#### 136 只出现一次的元素-位运算


```python
from functools import reduce 
nums = [1,1,2,2,6,7,7]
reduce(lambda x,y:x^y,nums)
# ^异或运算，同则0，异则1
            
            
```




    6



#### 169 多数元素-摩尔投票



```python
# 初始化： 票数统计 votes = 0 ， 众数 x。
# 循环： 遍历数组 nums 中的每个数字 num 。
# 当 票数 votes 等于 0 ，则假设当前数字 num 是众数。
# 当 num = x 时，票数 votes 自增 1 ；当 num != x 时，票数 votes 自减 1 。
# 返回值： 返回 x 即可。

def majorityElement( nums) -> int:

    vote = 0

    for i in nums:
        if vote==0:
            x=i 
        vote += 1 if i==x else -1
    return x

nums = [2,2,1,1,1,2,2]
majorityElement(nums)
```




    2



#### 75 颜色分类


```python
def sortColors( nums) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """


    n = len(nums)
    if n<2:return 

    zero = 0
    one = 0
    two = n 

    def swap(nums,id1,id2):
        nums[id1],nums[id2] = nums[id2],nums[id1]
    

    while one<two:
        if nums[one]==0:
            swap(nums,one,zero)
            one+=1
            zero+=1
        elif nums[one]==1:
            one+=1
        else:
            two-=1
            swap(nums,one,two)
nums = [2,0,2,1,1,0]   
sortColors(nums)   
nums   
```




    [0, 0, 1, 1, 2, 2]



#### 31 下一个排列


```python
# 设计思路： 
# 1.从数组右侧向左开始遍历，找是否存在nums[i]>nums[i-1]的情况， 
# 2.如果不存在这种nums[i]>nums[i-1]情况 ，for循环会遍历到i==0（也就是没有下一个排列），此时按题意排成有序Arrays.sort() 
# 3.如果存在，则将从下标i到nums.length()的部分排序，然后在排过序的这部分中遍历找到第一个大于nums[i-1]的数，并与nums[i-1]交换位置


def nextPermutation( nums) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    print(nums) 
    for i in range(len(nums)-1, 0, -1):
        if nums[i] > nums[i-1]:
            nums[i:] = sorted(nums[i:])
            for j in range(i, len(nums)):
                if nums[j] > nums[i-1]:
                    nums[j], nums[i-1] = nums[i-1], nums[j]
                    break
            return
    nums.sort()


nums = [1,2,3]
nextPermutation(nums)
nums
```

    [1, 2, 3]





    [1, 3, 2]



#### 287 寻找重复数


```python

def findDuplicate(nums) -> int:
    slow = 0
    fast = 0
    
    # 使用快慢指针找到相遇点
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    
    # 将一个指针重置为起点，然后两个指针以相同的速度移动，直到它们再次相遇
    slow = 0
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    
    return slow

nums = [1,3,4,2,2]
```

### 动态规划

#### 300 最长递增子序列


```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # dp[i]表示以nums[i]这个数字结尾的最长递增子序列的长度
        # dp[3]如何由前面得到，既然递增子序列，只需找到前面比nums[3]小的数，在其dp长度上+1即可
        # 由于比dp[i]小的数可能有多个，所以遍历取数值最大的即为dp[i]
        dp = [1]*len(nums)
        for i in range(len(nums)):
            for j in range(i):  # 遍历
                if nums[i]>nums[j]: # 比当前位置小的数
                # 如果求非递减子序列，nums[i]>=nums[j]，加个等号即可
                    dp[i] = max(dp[i],dp[j]+1)
        return max(dp)
```

#### 53 最大子序和（最大子数组和）


```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # dp: 以nums[i]为结尾的最大子数组和为dp[i]
        # dp[i-1]到dp[i]: dp[i]=max(nums[i],dp[i-1]+nums[i])
        n = len(nums)
        if n==0:return 0
        dp=[0]*n # 第一个元素前没元素，dp[0]=nums[0]
        dp[0] = nums[0]
        for i in range(1,n):
            dp[i] = max(nums[i],nums[i]+dp[i-1])
        return max(dp)
```

#### 1143 最长公共子序列


```python
# 设字符串 text1,text2 的长度分别为 m+1行和 n+1列二维dp数组，dp[i][j]表示text1[0:i],text2[0:j]的最长公共子序列的长度。


def longestCommonSubsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0 for i in  range(n + 1)] for j in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]
s1 ="abcde"
s2 = "ace"
longestCommonSubsequence(s1,s2)
```




    3



#### 最长公共子串


```python
def find_lcsubstr(s1, s2):   
    m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]  #生成0矩阵，为方便后续计算，比字符串长度多了一列  
    mmax=0   #最长匹配的长度  
    p=0  #最长匹配对应在s1中的最后一位  
    for i in range(len(s1)):   # 和子序列不同，这是从初始0下标开始，如果字符相等就更新斜对角下一个位置最大值长度
        for j in range(len(s2)):  
            if s1[i]==s2[j]:  
                m[i+1][j+1]=m[i][j]+1  
                if m[i+1][j+1]>mmax:  # 匹配到连续的最大长度就更新最大值记录变量
                    mmax=m[i+1][j+1]  
                    p=i+1   # 最大长度所在下标1，因为[:p]取不到p
    return m,s1[p-mmax:p],mmax   #返回最长子串及其长度  

print(find_lcsubstr('abcdfg','abdfg'))
```

    ([[0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 3]], 'dfg', 3)


#### 72 编辑距离


```python
def minDistance(self, word1: str, word2: str) -> int:
    m,n = len(word1),len(word2)
    dp=[[0 for i in range(n+1)]for j in range(m+1)]
    for i in range(1,m+1):
        dp[i][0]=i 
    for j in range(1,n+1):
        dp[0][j]=j 
    for i in range(1,m+1):
        for j in range(1,n+1):
            if word1[i-1]==word2[j-1]:
                dp[i][j] = dp[i-1][j-1] # 相等不用加操作
            else:
                dp[i][j] = min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+1) # 不相等则需要增删插入，多一次操作
    return dp[m][n]
```

#### 516 最长回文子序列



```python
def longestPalindromeSubseq(self, s: str) -> int:
    """
    dp[i][j] 表示s[i:j]的最长回文子序列距离
    base:
    i>j的话 dp[i][j]=0   即左下三角为0
    i=j的话 dp[i][j]=1   即对角线为1
    要想求dp[i][j],那么首先得知道比他小一个字符距离的最长序列多少,即dp[i+1][j-1](中心向外扩展的，即i+1-1,j-1+1),如果新扩展的俩字符相等，
    那么直接加2，如果不相等，那就看单边，只往左移i,或者往右边移动j的那个dp位置，谁的值大就是谁

    """
    n = len(s)
    dp=[[0 for i in range(n)]for j in range(n)]
    for i in range(n):
        dp[i][i]=1
    for i in range(n-2,-1,-1):
        for j in range(i+1,n):
            if s[i]==s[j]:
                dp[i][j]=dp[i+1][j-1]+2
            else:
                dp[i][j] = max(dp[i][j-1],dp[i+1][j])
    return dp[0][n-1]
```

#### 10 正则表达式匹配


```python
def isMatch(s: str, p: str) -> bool:
    """
    初始化二维dp数组
    dp[i][j]代表s[0:i-1]和p[0:j-1]是否匹配 
    1.base:dp[0][0]为代表空，则为True
    2.s为空，p不空，遍历p，如果为*则把前面的字母消除
    3.s和p都不空
        3.1.如果s末尾和p末尾一样，或者p末尾是点，则dp[i][j]=dp[i-1][j-1]
        3.2.如果p末尾是*
            3.2.1.如果s末尾和p末尾前一个一样，或者p末尾前一个是点，则a*/.*可匹配0，1，多个s末尾字符
            3.2.2.如果不一样，用*消灭前面的字符
        3.3.如果不一样，直接不需要操作，默认赋值都是Flase了

    """
    dp = [[False]*(len(p)+1) for _ in range(len(s)+1)]
    dp[0][0]=True

    for j in range(1,len(p)+1):
        if p[j-1]=='*': # p第一个不可能为*，用任意字符+* 消除
            dp[0][j] = dp[0][j-2]
    for i in range(1,len(s)+1):
        for j in range(1,len(p)+1):
            if s[i-1]==p[j-1] or p[j-1]=='.':
                dp[i][j] = dp[i-1][j-1]
            elif p[j-1]=='*':
                if s[i-1]==p[j-2] or p[j-2]=='.':
                    dp[i][j] = dp[i][j-2] or dp[i-1][j-2] or dp[i-1][j]
                else:
                    dp[i][j] = dp[i][j-2]
    return dp[-1][-1]
s = "ab" 
p = ".*"
isMatch(s,p)
```




    True



### 回文子串问题

#### 5 最长回文子串


```python
def longestPalindrome(self, s: str) -> str:
    """
    从中间扩散到两边判断回文串
    遍历0～n-1,判断以s[i]为中心的回文、判断以s[i]和s[i+1]为中心的回文，
    小心s[i+1]越界
    """
    def palindorme(s,l,r):
        while l>=0 and r<len(s) and s[l]==s[r]:
            l-=1
            r+=1
        return s[l+1:r]
    
    res = ''
    for i in range(len(s)):
        s1 = palindorme(s,i,i)
        s2 = palindorme(s,i,i+1)
        res = res if len(s1)<len(res) else s1 
        res = res if len(s2)<len(res) else s2 
    return res
```

#### 647 回文子串


```python
def countSubstrings(self, s: str) -> int:
    """
    统计回文子串数量，
    具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串，
    在判断回文函数里记录回文数量即可
    """
    def palindorme(s,l,r):
        res = 0
        while l>=0 and r<len(s) and s[l]==s[r]:
            l-=1
            r+=1
            res+=1
        return res
    
    result = 0
    for i in range(len(s)):
        result += palindorme(s,i,i)
        result += palindorme(s,i,i+1)
    print(result)
    return result
```


```python
class Node(object):
    def __init__(self,val):
        self.next = None
        self.val = val

class Linklist(object):
    def __init__(self,):
        self.head= None
    def create(self,ls):
        self.head = Node(ls[0])
        cur = self.head
        tmp = self.head
        for h in ls[1:]:
            cur.next = Node(h)
            cur = cur.next
        return tmp
    def printf(self):
        head = self.head
        while head!=None:
            print(head.val)
            head = head.next
# init
data = [1,2,4,8]
l1 = Linklist()
l1.create(data)
l1.printf()

            
            


```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[23], line 25
         23 data = [1,2,4,8]
         24 l1 = Linklist()
    ---> 25 l1.create(data)
         26 l1.printf()


    Cell In[23], line 10, in Linklist.create(self, ls)
          9 def create(self,ls):
    ---> 10     self.head = Linklist(ls[0])
         11     cur = self.head
         12     tmp = self.head


    TypeError: __init__() takes 1 positional argument but 2 were given


### 链表问题总结

#### 链表初始化


```python
class LinkNode(object):
    def __init__(self,x):
        self.val = x
        self.next = None
class LinkList(object):
    def __init__(self):
        self.head = None
    def create(self,data):
        self.head = LinkNode(data[0])
        p = self.head
        q = self.head
        for i in range(1,len(data)):
            q.next = LinkNode(data[i])
            q = q.next
        return p
    def printf(self):
        head = self.head
        while head!=None:
            print(head.val)
            head = head.next
# init
data = [1,2,4,8]
l1 = LinkList()
l1.create(data)
l1.printf()
```

    1
    2
    4
    8


#### 找到倒数第k个节点


```python
# 倒数第k个节点
fast = l1.head
slow = l1.head
k=1
while k>0:
    k-=1
    fast = fast.next
while fast!=None:
    fast = fast.next
    slow = slow.next
print(slow.val)
```

#### 删除链表倒数第n个节点


```python
def removeNthFromEnd(head, n):
    dummy = LinkNode(-1)
    dummy.next = head
    fast = dummy
    slow = dummy
    while n:
        fast = fast.next 
        n-=1
    while fast.next:
        fast=fast.next
        slow=slow.next
    slow.next = slow.next.next
    return dummy.next

data = [1,2,4,8]
l1 = LinkList()
l1.create(data)
l1.printf()
rl1 = removeNthFromEnd(l1.head,2)

while rl1:
    print(rl1.val)
    rl1 =rl1.next
```

    1
    2
    4
    8
    1
    2
    8


#### 链表中点


```python
# 中点
# 奇数为正中间，偶数中间偏右
fast = l1.head
slow = l1.head
while fast !=None and fast.next!= None:
    fast = fast.next.next
    slow = slow.next
print(slow.val)
```

    4


#### 反转链表


```python
# 反转
pre = None
head = l1.head
while head !=None:
    tmp = head.next
    head.next = pre
    pre = head
    head = tmp
while pre!=None:
    print(pre.val)
    pre = pre.next
```

    8
    4
    2
    1


#### 链表是否有环


```python
# 初始环
n1 = LinkNode(8)
n2 = LinkNode(7)
n3 = LinkNode(0)
n4 = LinkNode(3)
n1.next = n2
n2.next = n3
n3.next = n4
n4.next = n3

# 有环?
slow = n1
fast = n1
while slow!=None and fast.next!=None:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
        print(1)
        break
print(0)

```

    1
    0


#### 链表环起点


```python
# 环起点
slow = n1
fast = n1
while slow !=None and fast.next !=None:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
        fast = n1
        break
while slow!=fast:
    slow = slow.next
    fast = fast.next
print(slow.val)
```

    0


#### 25 k个一组反转链表
#### 24 两两交换链表中的节点



```python
a =[1,3,2,6]
obj = LinkList()
obj.create(a)
obj.printf()

class solution:
    def reverse(self,head,tail):
        pre = None
        cur = head
        while pre!=tail:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return tail,head
    def reverseKgroup(self,head,k):
        hair = LinkNode(0)
        hair.next = head
        pre = hair
        
        while head:
            tail = pre
            for i in range(k):
                tail = tail.next
                if not tail:
                    return hair.next
            nxt = tail.next
            head,tail = self.reverse(head,tail)
            pre.next = head
            tail.next = nxt
            pre = tail
            head = tail.next
        return hair.next
    
s = solution()
res = s.reverseKgroup(obj.head,2)
        
while res:
    print(res.val)
    res = res.next
```

    1
    3
    2
    6
    3
    1
    6
    2


#### 指定m,n区间反转链表(92 反转链表 II)


```python
def reverseBetween(head, left: int, right: int):
    dummy = LinkNode(-1)
    dummy.next = head 
    p = dummy
    for i in range(left-1):
        p = p.next 
    # 反转指定区间
    m = p.next
    n = m.next 
    for i in range(right-left):
        temp = n.next 
        n.next = m
        m = n 
        n = temp 
    # 注意链接顺序
    p.next.next = n 
    p.next = m 
    return dummy.next

data = [1,2,3,4,5]
l1 = LinkList()
l1.create(data)
l1.printf()

rb_l1 = reverseBetween(l1.head,2,4)
while rb_l1:
    print(rb_l1.val)
    rb_l1 =rb_l1.next
```

    1
    2
    3
    4
    5
    1
    4
    3
    2
    5


#### 2 两数相加


```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        if not l1:  # 如果l1为空链表，直接返回l2
            return l2
        if not l2:  # 如果l2为空链表，直接返回l1
            return l1
        dummy = ListNode(-1)  # 创建一个虚拟头节点，初始值为-1
        p = dummy  # 创建一个指针p，指向虚拟头节点，用于构建结果链表
        carry = 0  # 初始化进位值为0
        while l1 and l2:  # 当l1和l2都不为空时进行循环
            p.next = ListNode((l1.val + l2.val + carry) % 10)  # 计算当前位的值，考虑进位
            carry = (l1.val + l2.val + carry) // 10  # 计算新的进位值
            l1 = l1.next  # 移动l1指针到下一位
            l2 = l2.next  # 移动l2指针到下一位
            p = p.next  # 移动p指针到刚刚创建的节点
        while l1:  # 当l1不为空时，继续循环
            p.next = ListNode((l1.val + carry) % 10)  # 计算当前位的值，考虑进位
            carry = (l1.val + carry) // 10  # 计算新的进位值
            l1 = l1.next  # 移动l1指针到下一位
            p = p.next  # 移动p指针到刚刚创建的节点
        while l2:  # 当l2不为空时，继续循环
            p.next = ListNode((l2.val + carry) % 10)  # 计算当前位的值，考虑进位
            carry = (l2.val + carry) // 10  # 计算新的进位值
            l2 = l2.next  # 移动l2指针到下一位
            p = p.next  # 移动p指针到刚刚创建的节点
        if carry == 1:  # 如果最后还有进位，需要额外添加一个值为1的节点
            p.next = ListNode(1)
        return dummy.next  # 返回虚拟头节点的下一个节点，即结果链表的头节点

```

### 滑动窗口模板
#### 76 最小覆盖子串


```python
"""
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
"""
def minWindow(s: str, t: str) -> str:
    need,window = dict(),dict()
    for i in t:
        need[i] = 1 if i not in need else need[i]+1
    l,r,valid = 0,0,0
    start,lens = 0,float('inf')
    while r<len(s):
        c = s[r]
        # 右移
        r+=1
        if need.get(c):
            window[c] = 1 if c not in window else window[c]+1
            if window[c]==need[c]:
                valid+=1
        while valid==len(need):
            # 更新最小覆盖子串
            if r-l<lens:
                start = l 
                lens = r-l 
            move=s[l]
            l+=1
            if need.get(move):
                if need[move]==window[move]:
                    valid-=1
                window[move]-=1
    return "" if lens==float('inf') else s[start:start+lens]
s = "ADOBECODEBANC"
t = "ABC"
minWindow(s,t)
```




    'BANC'



#### 567 字符串的排列


```python
"""
输入：s1 = "ab" s2 = "eidbaooo"
输出：true
解释：s2 包含 s1 的排列之一 ("ba").
"""
def checkInclusion( s1: str, s2: str) -> bool:
    need,window = dict(),dict()
    for c in s1:
        need[c] = 1 if c not in need else need[c]+1
    l,r = 0,0
    valid = 0
    while r<len(s2):
        c = s2[r]
        r+=1
        if need.get(c):
            window[c] = 1 if c not in window else window[c]+1
            if need[c]==window[c]:
                valid +=1
        # 判断左侧窗口是否收敛
        while r-l==len(s1):
            # 判断是否找到了合法子串
            if valid==len(need):
                return True
            move = s2[l]
            l+=1
            if need.get(move):
                if need[move]==window[move]:
                    valid-=1
                window[move]-=1
    return False
s1 = "ab" 
s2 = "eidbaooo"
checkInclusion(s1,s2)
```




    True



#### 438 找到所有字母异位词


```python
"""
输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。

"""
# 在上一个的基础上把左边界纪录下就可
def findAnagrams(s: str, p: str):
    need,window = dict(),dict()
    for c in p:
        need[c] = 1 if c not in need else need[c]+1
    l,r = 0,0
    valid = 0
    res = []
    while r<len(s):
        c = s[r]
        r+=1
        if need.get(c):
            window[c]=1 if c not in window else window[c]+1
            if need[c]==window[c]:
                valid+=1
        while r-l==len(p):
            if valid==len(need):
                res.append(l)
            move = s[l]
            l+=1
            if need.get(move):
                if need[move]==window[move]:
                    valid-=1
                window[move]-=1
    return res
s = "cbaebabacd"
p = "abc"
findAnagrams(s,p)
```




    [0, 6]



#### 3 无重复字符的最长子串


```python
"""
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

"""

def lengthOfLongestSubstring( s: str) -> int:
    window = dict()
    l,r = 0,0
    res = 0
    while r<len(s):
        c = s[r]
        r+=1
        window[c] = 1 if c not in window else window[c]+1
        # 只要窗口里有字符数量>1就缩左1位
        while window[c]>1:
            m = s[l]
            l+=1
            window[m]-=1
        # 更新
        res = max(res,r-l)
    return res
s = "abcabcbb"
lengthOfLongestSubstring(s)
```




    3



### 回溯问题

#### 17 电话号码的字母组合


```python
def letterCombinations(digits: str):
    if not digits:return []
    dic = {'2':['a','b','c'],
    '3':['d','e','f'],
    '4':['g','h','i'],
    '5':['j','k','l'],
    '6':['m','n','o'],
    '7':['p','q','r','s'],
    '8':['t','u','v'],
    '9':['w','x','y','z']
    }

    
    def back(path,id): #  数字下标和路径
        # 达到数字长度就加入结果
        if id==len(digits):
            res.append(''.join(path))
        
        # 没有就拿到当前数字，获取数字对应的字符列表，遍历列表添加结果，数字下标加1回溯递归
        else:
            di = digits[id]
            for c in dic[di]:
                path.append(c)
                back(path,id+1)
                path.pop()

    res,path =[],[]
    back(path,0)
    return res
letterCombinations('23')
```




    ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']



#### 22 括号生成


```python
def generateParenthesis(n: int):

    s = ''
    res = []
    def dfs(s,l,r):
        # 都用完加入结果
        if l==0 and r==0:
            res.append(s)

        # 左剩的比右多
        if l>r:
            return
        # 用左
        if l:
            dfs(s+'(',l-1,r)
        # 用右
        if r:
            dfs(s+')',l,r-1)
    
    dfs(s,n,n)
    return res
generateParenthesis(3)

```




    ['((()))', '(()())', '(())()', '()(())', '()()()']



#### 46 全排列


```python
def permute(nums):

    def back(path,nums):
        if len(path)==len(nums):
            end.append(path[:])
        
        for i in range(len(nums)):
            if nums[i] not in path:
                path.append(nums[i])
                back(path,nums)
                path.pop()
    
    path ,end = [],[]
    back(path,nums)
    return end
permute([1,2,3])
```




    [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]



#### 47 全排列2


```python
def permuteUnique(nums):

    def back(path,nums,used):
        if len(path)==len(nums):
            end.append(path[:])
        
        for i in range(len(nums)):
            if used[i]==1:
                continue
            # 两个数一致，前一个是否被用过都无所谓，任选一个0/1跳过即可
            if i>0 and nums[i]==nums[i-1] and used[i-1]==0:
                continue
            path.append(nums[i])
            used[i]=1
            back(path,nums,used)
            path.pop()
            used[i]=0
    # 记得排好序
    nums.sort()
    path ,end = [],[]
    used=[0] * len(nums)
    back(path,nums,used)
    return end

permuteUnique([1,1,2])
```




    [[1, 1, 2], [1, 2, 1], [2, 1, 1]]



#### 78 子集


```python
def subsets(nums):

    def back(nums,start):
        # 不用判断长度全都添加
        res.append(path[:])
        for i in range(start,len(nums)):
            path.append(nums[i])
            # 下标增加
            back(nums,i+1)
            path.pop()
    
    path = []
    res = []
    back(nums,0)
    return res

subsets([1,2,3])
```




    [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]



#### 77组合


```python

def combine( n: int, k: int):
    nums = list(range(1,n+1))

    def back(path,nums,start):
        # 长度判断
        if len(path)==k:
            res.append(path[:])
            return
        for i in range(start,n): 
            # 使用判断
            if nums[i] not in path:
                path.append(nums[i])
                # 下标增加判断
                back(path,nums,i+1)
                path.pop()

    path,res=[],[]
    back(path,nums,0)
    return res

n = 4
k = 2
combine(n,k)

```




    [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]



#### 39 组合总和


```python
def combinationSum( candidates, target: int):
    
    def back(path,target,candidates,start):
        if target<0:
            return
        if target==0:
            res.append(path[:])
            return
        
        for i in range(start,n):
            path.append(candidates[i])
            target = target-candidates[i]
            # 可重复因此需要重复多次减去每个下标为i的值，判断target是否为0
            back(path,target,candidates,i)
            path.pop()
            target = target+candidates[i]

    
    path,res=[],[]
    if not candidates:return []
    n = len(candidates)
    back(path,target,candidates,0)
    return res

    

combinationSum([2,3,6,7],7)
```




    [[2, 2, 3], [7]]



#### 79 单词搜索


```python
def exist(board, word: str):
    # i行j列字符 和 第k个单词字符 相等则真
    def dfs(i,j,k):
        if not 0<=i<len(board) or not 0<=j<len(board[0]) or board[i][j]!=word[k]:
            return False
        if len(word)-1==k:return True
        # 置为空字符串表示已经使用过
        board[i][j]=''
        res = dfs(i+1,j,k+1) or dfs(i-1,j,k+1) or dfs(i,j+1,k+1) or dfs(i,j-1,k+1)
        # 恢复可用
        board[i][j]=word[k]
        return res 
    
    for i in range(len(board)):
        for j in range(len(board[0])):
            # 如果递归为真则能找到
            if dfs(i,j,0):
                return True
    return False

board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
word = "ABCCED"
exist(board,word)
```




    True



#### 131 分割回文串


```python
def partition(s: str):
    
    # 遍历子串结束位置，也就是逗号的位置，从0到len(s)
    def back(path,i):
        # 遍历到==len(s)写入答案
        if i==n:
            res.append(path[:])
        
        # 从当前逗号位置递归，取不同长度字符串判断是否回文
        for j in range(i,n):
            t = s[i:j+1]
            if t==t[::-1]:
                path.append(t)
                back(path,j+1)
                path.pop()
    
    path =[]
    res = []
    n = len(s)
    back(path,0)
    return res
s = "aab"
partition(s)
```




    [['a', 'a', 'b'], ['aa', 'b']]



#### 51 N皇后


```python
def solveNQueens(n: int):
    
    def valid(r,c):
        # 遍历0～r行的每个R
        for R in range(r):
            #取出每行对应的列位置
            C = col[R]
            # 如果斜对角相等表示皇后互相攻击到了，返回F
            if r+c==R+C or r-c==R-C:
                return False
        return True



    def dfs(r,s):
        # 遍历完所有行了写入结果
        if r==n:
            # 
            res.append(['.'*c + 'Q'+'.'*(n-1-c) for c in col])
            return 
        for c in s:
            # 当前列可以
            if valid(r,c):
                # 行下标位置写入列
                col[r]=c
                # 递归下一行，去掉可用的列位置
                dfs(r+1,s-{c})

    # col列表每个下标表示对应行，表元素表示皇后的位置，初始化为0，即（下标，元素）代表（行，皇后位置）
    col = [0]*n
    # s集合表示可以遍历的列
    s = set(range(n))
    res = []
    dfs(0,s)
    return res
solveNQueens(4)
```




    [['.Q..', '...Q', 'Q...', '..Q.'], ['..Q.', 'Q...', '...Q', '.Q..']]



#### 200 岛屿数量

针对这道题，我们只需要对矩阵进行依次遍历，如果当前grid[x][y] == "1",则启动DFS模式。 找到与之相连的所有1，将其置为0。搜索结束后，我们找到了一个岛屿，岛屿数量+=1。 如此循环，最终返回岛屿数量即可。



```python
def numIslands(grid) -> int:
    def dfs(grid,i,j):
        if not 0<=i<len(grid) or not 0<=j<len(grid[0]) or grid[i][j]=='0':
            return 
        grid[i][j]='0'
        dfs(grid,i-1,j)
        dfs(grid,i+1,j)
        dfs(grid,i,j-1)
        dfs(grid,i,j+1)
        
    cnt =0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j]=='1':
                dfs(grid,i,j)
                cnt+=1
    return cnt
grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
numIslands(grid)
```




    3




```python
matrix = [[1,2,3],[4,5,6],[7,8,9]]
list(zip(*matrix[1:]))
```




    [(4, 7), (5, 8), (6, 9)]



### 二叉树

- 前中后序遍历：递归法、迭代法
- 队列迭代：层序遍历、z字遍历、二叉树最大/最小深度、二叉树右视图、翻转二叉树、对称二叉树
- 递归回溯：二叉树的直径、二叉树的最大路径和
- 构造二叉树：从前中序建二叉树、从中后序建二叉树
- 二叉搜索树：验证二叉搜索树、二叉搜索树的搜索

#### 144. 二叉树的前序遍历



```python

class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        
        # 迭代法
        # 根入栈出栈，出栈的根按右左顺序入栈，这样弹出栈时左节点才会被先记录
        if not root:
            return []
        
        stack = [root]
        res = []

        while stack:
            cur = stack.pop()
            res.append(cur.val)

            if cur.right:
                stack.append(cur.right)
            if cur.left:
                stack.append(cur.left)
        
        return res
        
        # 递归法
        # def pre(root):
        #     if not root:
        #         return
        #     res.append(root.val)
        #     pre(root.left)
        #     pre(root.right)
        # res=[]
        # pre(root)
        # return res


```

#### 94二叉树中序遍历


```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        stack = []  # 不能提前将root结点加入stack中
        result = []
        cur = root
        while cur or stack:
            # 先迭代访问最底层的左子树结点
            if cur:     
                stack.append(cur)
                cur = cur.left		
            # 到达最左结点后处理栈顶结点    
            else:		
                cur = stack.pop()
                result.append(cur.val)
                # 取栈顶元素右结点
                cur = cur.right	
        return result
```

#### 105 从前序与中序遍历序列构造二叉树



```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder:
            return
        # 找根节点
        val = preorder[0]
        root = TreeNode(val)

        # 先中序切割，因为中序可以确定左右长度
        index = inorder.index(val)
        inorder_left = inorder[:index]
        inorder_right = inorder[index+1:]

        # 前序切割长度和中序保持一致
        preorder_left = preorder[1:len(inorder_left)+1]
        preorder_right = preorder[len(inorder_left)+1:]

        # 递归调前序和中序左右子树
        root.left = self.buildTree(preorder_left,inorder_left)
        root.right = self.buildTree(preorder_right,inorder_right)

        return root
```

#### 106 从中序与后序遍历序列构造二叉树



```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not postorder:
            return 
        # 找根节点
        val = postorder[-1]
        root = TreeNode(val)

        # 先中序切割，因为中序可以确定左右长度
        index = inorder.index(val)
        inorder_left = inorder[:index]
        inorder_right = inorder[index+1:]

        # 后序切割长度和中序保持一致
        postorder_left = postorder[:len(inorder_left)]
        postorder_right = postorder[len(inorder_left):len(postorder)-1]

        # 递归调后序和中序左右子树
        root.left = self.buildTree(inorder_left,postorder_left)
        root.right = self.buildTree(inorder_right,postorder_right)

        return root
```

#### 700.二叉搜索树的搜索


```python
# 给定二叉搜索树（BST）的根节点 root 和一个整数值 val。

# 你需要在 BST 中找到节点值等于 val 的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 null 。 

class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        while root:
            if val<root.val:
                root = root.left
            elif val>root.val:
                root = root.right
            else:
                return root 
        return 
```

####  98 验证二叉搜索树


```python
# 给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。

# 有效 二叉搜索树定义如下：

# 节点的左子树只包含 小于 当前节点的数。
# 节点的右子树只包含 大于 当前节点的数。
# 所有左子树和右子树自身必须也是二叉搜索树。 

# 思路：中序遍历迭代法模版增加pre，比较当前节点（父节点）和pre前节点（左孩子）大小，左孩子>=父节点就不是二叉搜索树
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:

        stack = []
        cur = root
        pre = None 
        while cur or stack:
            if cur:
                stack.append(cur)
                cur = cur.left 
            else:
                cur = stack.pop()
                if pre and cur.val<=pre.val:
                    return False
                pre = cur 
                cur = cur.right 
        return True
```

#### 108 将有序数组转换为二叉搜索树



```python
# 给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。

# 高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。

 
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:

        def st(nums,left,right):
            if left>right:
                return
            
            mid = left+(right-left)//2
            root = TreeNode(nums[mid])
            root.left = st(nums,left,mid-1)
            root.right = st(nums,mid+1,right)
            return root
        
        root = st(nums,0,len(nums)-1)
        return root


```

#### 235 二叉搜索树的最近公共祖先
#### 236 二叉树的最近公共祖先



```python
# 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

# 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，
# 满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

# 这道题目刷过的同学未必真正了解这里面回溯的过程，以及结果是如何一层一层传上去的。
# 那么我给大家归纳如下三点：
# 求最小公共祖先，需要从底向上遍历，那么二叉树，只能通过后序遍历（即：回溯）实现从底向上的遍历方式。
# 在回溯的过程中，必然要遍历整棵二叉树，即使已经找到结果了，依然要把其他节点遍历完，因为要使用递归函数的返回值（也就是代码中的left和right）做逻辑判断。
# 要理解如果返回值left为空，right不为空为什么要返回right，为什么可以用返回right传给上一层结果。
# 可以说这里每一步，都是有难度的，都需要对二叉树，递归和回溯有一定的理解。
# 本题没有给出迭代法，因为迭代法不适合模拟回溯的过程。理解递归的解法就够了。

#
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        if not root or root==p or root==q:
            return root
        
        left = self.lowestCommonAncestor(root.left,p,q)
        right = self.lowestCommonAncestor(root.right,p,q)

        if left and right:return root
        if not left:return right 

        return left
```
