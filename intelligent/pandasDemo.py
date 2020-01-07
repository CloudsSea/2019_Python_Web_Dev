import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

def demo1():
    s = pd.Series([1,3,5,np.nan,6,8])
    print(s)
def demo2():
    dates = pd.date_range('20130101',periods=6)
    print(dates)
    df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
    print(df)

    #demo4

def demo3():
    df = pd.DataFrame({ 'A' : 1.,
                        'B' : pd.Timestamp('20130102'),
                        'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                        'D' : np.array([3]*4,dtype='int32'),
                        'E' : pd.Categorical(['test','train','test','train']),
                        'F' :'foo'})
    print(df)
    print(df.dtypes)

def demo4():
    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

    # print(df.head())
    # print(df.tail(3))
    # print(df.index)
    # print(df.columns)
    # print(df.values)
    # print("describe对于数据的快速统计汇总")
    # print(df.describe())
    # # #转置
    # print(df.T)
    # # #按轴排序
    # print(df.sort_index(axis=1,ascending=False))
    # # #按值排序
    # print(df.sort_values(by='B'))

    # 三 选择
    #3.1 获取
    # print(df['A'])
    # print(df.A)
    # print(df[0:3])
    # print(df['20130102':'20130104'])
    #3.2 通过标签选择
    #使用标签来获取一个交叉的区域
    # print(df.loc[dates[0]])
    #通过标签来在多个轴上进行选择
    # print(df.loc[:,['A','B']])
    # 对于返回的对象进行维度缩减
    print(df.loc['20130102',['A','B']])








if __name__ == '__main__':
    # demo1()
    # demo2()
    #demo3()
    demo4()


