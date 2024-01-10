def largestRectangleArea( heights) -> int:

    res = 0
    stack =[]
    heights.insert(0,0)
    heights.append(0)
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