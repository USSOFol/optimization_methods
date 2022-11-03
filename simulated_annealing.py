import math
import matplotlib.pyplot as plt
from random import random

#

def func(x,y):
    # 定义方程
    res= 4*x*y**2+2*x**4-1.4*x*y+0.5*x**2*y**3-x+6*y+7
    return res

#定义退火法计算方法
class SA:
    def __init__(self,func,iter=100 ,t0=100, tf=0.01,alpha=0.99):
        self.func=func
        self.iter=iter
        self.t0=t0#退火的速率
        self.T=t0
        #当前温度
        self.tf=tf
        #终止温度
        self.alpha=alpha
        self.x = [random() * 11 - 5 for i in range(iter)]
        self.y = [random() * 11 - 5 for i in range(iter)]
        #生成-5到5的随机数，iter个
        self.most_best=[]
        self.history={"F": [],"T": []}
        #放一个空字典在此，存储过程
    def generation(self,x,y):
        #进行数据扰动
        while True:
            x_new=x+self.T*(random()-random())
            y_new=y+self.T*(random()-random())
            if (-5 <= x_new <= 5) & (-5 <= y_new <= 5):
                break
        return x_new, y_new

    def Metropolis(self, f, f_new):
        # metropolis准则,判断当前数据是否保留,或者用于置换
        if f >= f_new:
            return 1
        else:
            p=abs(math.exp(-(f_new-f)/self.T))
            if random()< p:
                return 1
            else:
                return 0
    def best(self):
        #存储最优结果
        f_list=[]
        for i in range(self.iter):
            f=self.func(self.x[i],self.y[i])
            f_list.append(f)
        f_best = min(f_list)
        idx=f_list.index(f_best)
        return f_best,idx
    def run(self):
        """运行"""
        count=0
        while self.T>self.tf:
            #内循环100次
            for i in range(self.iter):
                f=self.func(self.x[i],self.y[i])
                x_new,y_new=self.generation(self.x[i],self.y[i])
                f_new=self.func(x_new,y_new)
                if self.Metropolis(f,f_new):
                    self.x[i]=x_new
                    self.y[i]=y_new
            # 获取迭代一次后的最优解
            ft,_=self.best()
            self.history['F'].append(ft)
            self.history['T'].append(self.T)
            self.T=self.T*self.alpha
            count +=1
        f_best,idx=self.best()
        print(f"f_best={f_best}& idx={idx} & x,y={self.x[idx],self.y[idx]}")

if __name__ == '__main__':
    sa=SA(func)
    sa.run()
