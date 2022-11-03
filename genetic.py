"""本代码用于模拟遗传算法"""
"""
遗传算法需要：
种群大小
遗传代数
交叉概率
编译概率
"""
import numpy as np


class Gen:

    def __init__(self, code, code_style, nind, maxgen, pc, pm, ggap):
        """这里是遗传算法需要的参数"""
        self.code = code
        # 传入需要计算的对象
        self.code_style = code_style
        # 设置编码方法
        self.nind = nind
        # 设置种群大小
        self.maxgen = maxgen
        # 设置遗传代数
        """选择合适的个体后进行交叉，不一定所有基因能找到对象，给一个概率"""
        self.pc = pc
        # 交叉概率
        """变异概率,遗传后代出现不一样的概率"""
        self.pm = pm
        # 变异概率，一般较小
        self.ggap = ggap
        # 代沟，选择下个种群的大小
        self.chorm = 0

    def init_group(self):
        """初始化种群,就是得到100个个体"""
        long_gene = self.code.shape[0]
        """给出基因的长度"""
        # print(self.long_gene)
        chorm = np.zeros([self.nind, long_gene])
        """chorm用于储存种群"""
        for i in np.arange(self.nind):
            chorm[i, :] = np.random.permutation(long_gene)
            """随机排列组合"""
        self.chorm = chorm
        """chorm为初始种群"""
        # print(chorm)
        # return chorm

    def optimize(self):
        """优化部分"""
        gen = 0
        dis = distance(self.code)
        # print(self.chorm)
        # len_road=path_lengeth(dis,self.chorm)
        # return len_road
        chorm = self.chorm
        """设置代数"""
        while gen < self.maxgen:
            """当遗传代数小于预定代数的时候循环"""
            """计算适宜度，适者生存，对于优化路线问题了，适宜度就是距离了"""
            len_road = path_lengeth(dis, chorm)
            """计算路线长度"""
            fitv = fitness(len_road)  # 适应率，越大越好
            # print('fitv:',fitv)
            """选择符合要求的部分，剔除不需要的"""
            selch = select(chorm, fitv, self.ggap)
            # print("1",selch)
            """对基因进行交叉"""
            selch = recombin(selch, self.pc)
            # print("2",selch)
            """输入轮盘赌后选中的个体与交叉概率"""
            selch = variation(selch, self.pm)
            """接下来进行一波变异,输入交叉后的数组和变异概率"""
            # print("3",selch)
            # print(self.chorm.shape)
            selch = reverse(selch, dis)
            """逆操作"""
            chorm = reins(chorm, selch, len_road)
            """重新插入,上一轮的种群，这一轮种群，路径长度（这里的适应度）"""
            gen += 1
        return chorm, dis


"""调用函数定义"""


def distance(matr):
    """点坐标矩阵边的距离"""
    m, n = matr.shape
    dis = np.zeros([m, m])
    for i in np.arange(m):
        for j in np.arange(m):
            dis[i, j] = np.sqrt((matr[i, 0] - matr[j, 0]) ** 2 + (matr[i, 1] - matr[j, 1]) ** 2)
    return dis


def path_lengeth(dis, chorm):
    """计算路径长度，也就是适应度"""
    # print(chorm.shape[0])
    len_road = np.zeros([chorm.shape[0], 1])
    # print(len_road)
    for i in np.arange(chorm.shape[0]):
        for j in np.arange(chorm.shape[1]):
            line = chorm[i, :]
            if j <= chorm.shape[1] - 2:
                len_road[i] += dis[np.uint8(line[j]), np.uint8(line[j + 1])]
            else:
                len_road[i] += dis[np.uint8(line[0]), np.uint8(line[j])]

    return len_road


def fitness(dis):
    """计算适应率"""
    fitv = 1 / dis
    return fitv


def select(chrom, fitv, ggap):
    """选择函数:输入图矩阵，适应度，代沟比例
    输出被选择的个体
    选择方法为轮盘读
    """
    nind = chrom.shape[0]
    nsel = max(np.ceil(nind * ggap), 2)
    """至少取两个样本"""
    newchrlx = sus(fitv, nsel)

    """轮盘赌代码实现"""
    selch = chrom[newchrlx, :]

    """选出相应的行送回去"""
    return selch
    # 轮盘赌算法


def sus(fitv, nsel):
    """轮盘赌算法"""
    """输入
    适应度
    取得个体数量
    输出：
    被选中个体的索引号"""
    nind, ans = fitv.shape  # 种群数量
    """计算累积概率"""
    cumfit = np.cumsum(fitv)
    cumfit = cumfit.reshape(nind, 1)
    # 100*1
    # 累积和
    nsel = np.uint8(nsel)
    mat = np.linspace(0, nsel - 1, nsel)
    mat = mat.reshape(nsel, 1)
    trails = cumfit[nind - 1] / nsel * (np.random.rand() + mat)
    # 产生随机数看落在哪个区间,90*1
    mf = np.tile(cumfit, nsel)
    # print(mf.shape)
    mt = np.tile(trails, nind).T  # 计算mt落在mf区间里面的数量与对应的标号
    # print(mt.shape)
    # 生成的随机数
    # 扩展矩阵
    ze = np.zeros([1, nsel])
    tt = np.vstack((ze, mf[0:nind - 1, :]))  ##这一行有问题
    aa, bb = np.where((tt <= mt) & (mt < mf))

    """注意这个傻逼括号一定要"""
    # print('a',aa)
    # tt=tt.reshape(nind,nsel)
    # print(cc)
    """这里输出的cc就是保留的行数"""
    """这里获取随机区间内的坐标索引"""
    al = np.random.permutation(aa)
    # print(al)
    """随机打乱输出的行，用于给下一个"，打乱便于交叉"""
    return al


def recombin(selch, pc):
    """对pc概率的个体进行交叉"""
    """先找出需要被交叉的个体"""
    nsel = selch.shape[0]
    for i in range(0, nsel - 1 - np.mod(nsel - 1, 2), 2):
        if pc >= np.random.rand():
            """使用概率"""
            selch[i, :], selch[i + 1, :] = intercross(selch[i, :], selch[i + 1, :])
    return selch

    """交叉方法"""
def intercross(a, b):
    len = a.shape[0]
    r1 = np.random.randint(0, len, 1)
    r2 = np.random.randint(0, len, 1)
    """生成随机数"""
    if r1 != r2:
        a0 = a.copy()
        b0 = b.copy()
        r = np.vstack((r1, r2))
        s = r.min()
        e = r.max()
        # s=np.uint8(s)
        # e=np.uint8(e)
        for i in range(s, e):
            a1 = a.copy()
            b1 = b.copy()
            a[i] = b0[i]
            b[i] = a0[i]
            x = np.where(a == a[i])
            y = np.where(b == b[i])
            x = np.setdiff1d(x, i)
            y = np.setdiff1d(y, i)
            # 剔除相同的地方
            if x.size != 0:
                a[x] = a1[i]
            if y.size != 0:
                b[y] = b1[i]
    return a, b


def variation(selch, pm):
    """输入待变化矩阵和"""
    mm, nn = selch.shape[0], selch.shape[1]
    for i in range(mm):
        if pm >= np.random.rand():
            al = np.random.permutation(nn)
            selch[i, al[0]], selch[i, al[1]] = selch[i, al[1]], selch[i, al[0]]
            """修改这里"""
    return selch


def reverse(selch, dis):
    mm, nn = selch.shape[0], selch.shape[1]
    objv = path_lengeth(dis, selch)
    selch1 = selch.copy()
    for i in range(mm):
        r1 = np.random.randint(0, nn, 1)
        r2 = np.random.randint(0, nn, 1)
        r = np.vstack((r1, r2))
        s = r.min()
        e = r.max()
        t = np.linspace(s, e, e - s + 1)
        t = np.uint8(t)
        t1 = np.flipud(t)
        selch1[i, t] = selch1[i, t1]
        """增加一个selch1交换se的步骤"""
    objv1 = path_lengeth(dis, selch1)
    row, col = np.where(objv1 < objv)
    """找出路径更小的排列进行替换"""
    selch[row, :] = selch1[row, :]
    return selch


def reins(chrom, selch, objv):
    """选取十个适应度最好的插回selch中，构成新的chrom"""

    index = np.argpartition(objv.ravel(), 10)[:10]
    # print("sss:",index)
    tt = chrom[index, :]
    chrom1 = np.vstack((tt, selch))
    return chrom1


if __name__ == '__main__':
    # x=np.random.rand(10,2)*10
    # print(x)
    """待处理矩阵"""
    """x=np.array([[9.10893551, 0.94758485],
 [3.19123719, 7.10030758],
 [8.99780653, 8.46583976],
 [8.59168471 ,8.16506294],
 [8.41959546 ,4.50936991],
 [3.16840804, 2.93466796],
 [6.2796096 , 0.58816662],
 [1.80523243 ,1.61825121],
 [7.3146517  ,5.02325565],
 [7.20971725 ,4.16120105]])"""

    x = np.array([[16.47, 96.10]
                     , [16.47, 94.44]
                     , [20.09, 92.54]
                     , [22.39, 93.37]
                     , [25.23, 97.24],
                  [22.00, 96.05],
                  [20.47, 97.02],
                  [17.20, 96.29],
                  [16.30, 97.38],
                  [14.05, 98.12],
                  [16.53, 97.38],
                  [21.52, 95.59],
                  [19.41, 97.13],
                  [20.09, 92.55]])
    dis = distance(x)
    # print(dis)
    x_size = x.shape

    heredity = Gen(x, 1, 100, 400, 0.9, 0.05, 0.9)
    """def __init__(self, code, code_style, nind, maxgen, pc, pm, ggap):"""
    heredity.init_group()
    chrom, dis = heredity.optimize()
    len_road = path_lengeth(dis, chrom)
    # print(len_road)
    index = np.argpartition(len_road.ravel(), 1)[:1]
    print("路径为：", chrom[index, :], "距离为：", len_road[index, :])
    # print(heredity.init_group())
