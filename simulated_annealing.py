import math
import matplotlib.pyplot as plt
from random import random

#

def func(x,y):
    # 定义方程
    res= 4*x*y**2+2*x**4-1.4*x*y+0.5*x**2*y**0.5-x+6*y+7
    return res
#定义退火法计算方法
class SA:
    def __init__(self,func,iter=100 ,t0=100, tf=0.01,alpha=0.99):
        self.func=func
        self.iter=iter
        self.t0=t0
        self.tf=tf
        self.alpha=alpha
        self.x = [random() * 11 - 5 for i in range(iter)]
        self.y = [random() * 11 - 5 for i in range(iter)]



if __name__ == '__main__':
    pass