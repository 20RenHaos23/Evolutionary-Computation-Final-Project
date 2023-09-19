import os
import cv2
import random
import numpy as np
from operator import attrgetter
import matplotlib.pyplot as plt
from CAP import cap
import tensorflow as tf
import copy

crossover_list = [[1,0,0],[0,1,0],[0,0,1]]

def calRawPSNR(img_true, img_fog):
    img_fog_copy = img_fog.copy()
    for i in range(len(img_true)):
        img_fog_copy[i] = img_fog[i] / 255.0
    psnr = cal_psnr(img_true,img_fog_copy)
    print("沒有去霧前的平均PSNR為 {}".format(float(psnr)))
def mutation(img_true,img_fog,pop):
    
    for i in range(1):        
        mut_filter_r = random.choice(pop)
        num = random.randint(1,50) #均勻分布
        mut_filter_r.filter_r = num * 2 + 1
        J = cap(img_fog,mut_filter_r.filter_r,mut_filter_r.guided_r,mut_filter_r.epsilon)       
        psnr = cal_psnr(img_true, J)
        mut_filter_r.psnr = psnr
        print("突變filter_r的第{}次的平均PSNR為 {}".format(i+1,float(psnr)))
    
    for j in range(1):
        mut_epsilon = random.choice(pop)
        mut_epsilon.epsilon = random.uniform(0,1)
        J = cap(img_fog,mut_epsilon.filter_r,mut_epsilon.guided_r,mut_epsilon.epsilon)        
        psnr = cal_psnr(img_true, J)
        mut_epsilon.psnr = psnr
        print("突變EPSILON的第{}次的平均PSNR為 {}".format(j+1,float(psnr)))
        
    for k in range(1):       
        mut_guided_r = random.choice(pop)
        mut_guided_r.guided_r = random.randint(1,50)       
        J = cap(img_fog,mut_guided_r.filter_r,mut_guided_r.guided_r,mut_guided_r.epsilon)       
        psnr = cal_psnr(img_true, J)
        mut_guided_r.psnr = psnr
        print("突變guided_r的第{}次的平均PSNR為 {}".format(k+1,float(psnr)))
    return pop
    

def cal_psnr(true,dehazing):
    psnr_list = []
    
    for i in range(len(true)):
        img_true = true[i]
        img_dehazing = dehazing[i]        
        psnr = tf.image.psnr(img_true,img_dehazing,max_val=1.0)
        psnr_list.append(float(psnr))
    
    psnr_mean = sum(psnr_list)/len(psnr_list)
    return psnr_mean
def calculate(img_fog,population,populationSize):
    for i in range(populationSize):
        
        J = cap(img_fog,population[i])
        
        return J
        
def readImage():
   
    #讀取人工加霧之前的乾淨影像
    path_true = (r'image\clear')
    path_fog = (r'image\haze')
       
    true = os.listdir(path_true)
    img_true = []
    
    for filename in true:
        #print(filename)
        path = (r'{}\{}'.format(path_true,filename))
        #print(path)
        pic = cv2.imread(path)
        h,w = pic.shape[0:2]
        
        pic = pic / 255.0
        img_true.append(pic)
        
    #讀取人工加霧之後的影像
    
    fog = os.listdir(path_fog)
    img_fog = [] 
    
    for filename in fog:
        #print(filename)
        path = (r'{}\{}'.format(path_fog,filename))
        #print(path)
        img_fog.append(cv2.imread(path))
    
    
    return img_true,img_fog
#Print some useful stats to screen
def printStats(pop,gen):
    print('Generation:',gen)       
    filter_r_list = []
    guided_r_list = []
    epsilon_list = []
    psnr_list = []
    
    for p in pop:     
        filter_r_list.append(p.filter_r)
        guided_r_list.append(p.guided_r)
        epsilon_list.append(p.epsilon)
        psnr_list.append(p.psnr)
        
    avg_filter_r = sum(filter_r_list)/len(filter_r_list)
    avg_guided_r = sum(guided_r_list)/len(guided_r_list)  
    avg_epsilon = sum(epsilon_list)/len(epsilon_list)
    avg_psnr = sum(psnr_list)/len(psnr_list)
    std_psnr = np.std(psnr_list)
          
    print('Avg filter_r->',avg_filter_r)
    print('Avg guided_r->',avg_guided_r)
    print('Avg epsilon->',avg_epsilon)    
    print('Avg PSNR->',avg_psnr)    
    print('Std PSNR->',std_psnr)    
    print('')
    return avg_filter_r,avg_guided_r,avg_epsilon,avg_psnr,std_psnr
#A trivial Individual class
class Individual:
    def __init__(self,filter_r,guided_r,epsilon,psnr):
        self.filter_r=filter_r
        self.guided_r=guided_r
        self.epsilon=epsilon
        self.psnr=psnr
        
    def copy(self):
        return copy.copy(self)
    
    @staticmethod    
    def crossover(x1,x2):
        a = random.choice(crossover_list)
        x=x1.copy()
        y=x2.copy()
        if a[0] == 1:
            f_r = x.filter_r
            x.filter_r = y.filter_r
            y.filter_r = f_r
        if a[1] == 1:
            g_r = x.guided_r
            x.guided_r = y.guided_r
            y.guided_r = g_r
        if a[2] == 1:
            e = x.epsilon
            x.epsilon = y.epsilon
            y.epsilon = e
        
        return x,y
    

def main():
    print("開始讀取影像")
    #讀取影像    
    img_true,img_fog = readImage()
    
    print("讀取影像完畢")
    #參數設定
    #要跑幾個世代
    generationCount = 100
    populationSize = 20    
    population=[]   
    avg_filter_r_list = [] #存放平均filter_r值
    avg_guided_r_list = [] #存放平均guided_r值
    avg_epsilon_list = [] #存放平均epsilon值
    avg_psnr_list = [] #存放平均psnr值
    std_psnr_list = [] #存放標準差psnr值
    #先計算一開始的psnr  
    calRawPSNR(img_true, img_fog)
    print("開始產生第0代")
    for i in range(populationSize):
        
        #隨機產生filter_r值
        num = random.randint(1,50) #均勻分布
        filter_r = num * 2 + 1
        
        #隨機產生guided_r值
        guided_r = random.randint(1,50) #均勻分布
        #隨機產生epsilon值
        epsilon = random.random()
        #將生成的資訊拿去除霧
        J = cap(img_fog,filter_r,guided_r,epsilon)
        #計算PSNR
        psnr = cal_psnr(img_true, J)
        print("第0代第{}個的平均PSNR為 {}".format(i+1,float(psnr)))
        ind=Individual(filter_r,guided_r,epsilon,psnr)
        population.append(ind)
  
    print("第0代PSNR計算完成")
    avg_filter_r,avg_guided_r,avg_epsilon,avg_psnr,std_psnr = printStats(population,0)
    avg_filter_r_list.append(avg_filter_r)
    avg_guided_r_list.append(avg_guided_r)
    avg_epsilon_list.append(avg_epsilon)
    avg_psnr_list.append(avg_psnr)
    std_psnr_list.append(std_psnr)
    
    
    print("開始執行基因演算法")
    
    for gen in range(generationCount): 
        print("第{}代開始".format(gen+1))
        x_population = []       
        random.shuffle(population)
        print("開始交配Crossover")
        for C in range(int(populationSize/2)):
            #隨機取兩個人
            parents=random.sample(population,2)
            #被選取到的兩個人進行交配
            
            x_1 , x_2 = Individual.crossover(parents[0],parents[1])
            x_1_J = cap(img_fog,x_1.filter_r,x_1.guided_r,x_1.epsilon)
            x_2_J = cap(img_fog,x_2.filter_r,x_2.guided_r,x_2.epsilon)
            
            #計算psnr
            x_1_psnr = cal_psnr(img_true, x_1_J)
            x_2_psnr = cal_psnr(img_true, x_2_J)
            if x_1_psnr >= x_2_psnr:
                x_ind=Individual(x_1.filter_r,x_1.guided_r,x_1.epsilon,x_1_psnr)
                print("生第{}個孩子的平均PSNR為 {}".format(C+1,float(x_1_psnr)))
            else:
                x_ind=Individual(x_2.filter_r,x_2.guided_r,x_2.epsilon,x_2_psnr)
                print("生第{}個孩子的平均PSNR為 {}".format(C+1,float(x_2_psnr)))
            x_population.append(x_ind)
        #將新的跟舊的串接起來    
        population.extend(x_population)
        #排序 從大到小
        population.sort(key=attrgetter('psnr'),reverse=True)
        #選擇最大的前20個
        population=population[:populationSize]  
        #做突變
        print("開始做突變Mutation")
        y_population = mutation(img_true,img_fog,population[10:len(population)+1])
        
        #選擇最大的前十個
        population=population[:10] 
        #將新的跟舊的串接起來    
        population.extend(y_population)
        #排序 從大到小
        population.sort(key=attrgetter('psnr'),reverse=True)
        #選擇最大的前20個
        population=population[:populationSize]  
        print('')
        print("最好的PSNR {}".format(population[0].psnr))
        print('')
        avg_filter_r,avg_guided_r,avg_epsilon,avg_psnr,std_psnr = printStats(population,gen+1)
        avg_filter_r_list.append(avg_filter_r)
        avg_guided_r_list.append(avg_guided_r)
        avg_epsilon_list.append(avg_epsilon)
        avg_psnr_list.append(avg_psnr)
        std_psnr_list.append(std_psnr)
        print("第{}代結束".format(gen+1))
        print('')
    
    generationcount = list(range(0,len(avg_filter_r_list)))
    plt.figure(1,figsize=(13,5))
    lines = plt.plot(generationcount,avg_psnr_list,'b',label = 'avg_psnr_list')
    plt.setp(lines,marker = "o")
    lines = plt.plot(generationcount,std_psnr_list,'g',label = 'std_psnr_list')
    plt.setp(lines,marker = "o")
    plt.title("PSNR") # title
    plt.xticks(generationcount)
    plt.xlabel("generation count") # x label
    plt.grid(True) #網格
    plt.legend()
    
    plt.figure(2,figsize=(13,5))
    lines = plt.plot(generationcount,avg_filter_r_list,'r',label = 'avg_filter_r_list')
    plt.setp(lines,marker = "o")
    lines = plt.plot(generationcount,avg_guided_r_list,'y',label = 'avg_guided_r_list')
    plt.setp(lines,marker = "o")
    plt.title("R") # title
    plt.xticks(generationcount)
    plt.xlabel("generation count") # x label
    plt.grid(True) #網格
    plt.legend()
    
    plt.figure(3,figsize=(13,5))
    lines = plt.plot(generationcount,avg_epsilon_list,'b',label = 'avg_epsilon_list')
    plt.setp(lines,marker = "o")
    plt.title("EPSILON") # title
    plt.xticks(generationcount)
    plt.xlabel("generation count") # x label
    plt.grid(True) #網格
    plt.legend()
    
    final_J = cap(img_fog,population[0].filter_r,population[0].guided_r,population[0].epsilon)
    for j in range(populationSize):
        print("最後的世代第{}個的PSNR {}".format(j,population[j].psnr))
    
    for n in range(10):
        #cv2.imshow("DEHAZING_{}".format(n),final_J[n]) 
        cv2.imwrite(r'image\result\{}.png'.format(n),final_J[n]*255)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    
    
if __name__ == '__main__':   
    main()
    
