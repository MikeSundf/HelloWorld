# -*- coding: utf-8 -*-
"""
Created on Tue Apr 4 00:00:00 2017

@author: takuro
v1.0 create
"""

import os.path
import numpy as np
import pandas as pd
import scipy.spatial.distance as dis
from scipy import stats
from pylab import *
from pylab import *
from product import Product

import numpy as np

def kl_divergence(w1, w2):
#    return np.sum(np.where(w1 != 0, w1 * np.log2(w1 / w2), 0))
    return stats.entropy(w1, w2, 2)

def js_divergence(w1, w2):
    r = (w1 + w2) / 2
    return 0.5 * (stats.entropy(w1, r, 2) + stats.entropy(w2, r, 2))

class ChoiceDistribution(object):
    """
    choice distribution
    """
    TIME_SLOT_NUM = 8
    CG_LEVEL_NUM = 3
    IC_RANGE_NUM = 3

    def __init__(self):
        
        self.choice_count = 0
#        self.cnt_arr = [[[0 for i3 in range(ChoiceDistribution.IC_RANGE_NUM)] for i2 in range(ChoiceDistribution.CG_LEVEL_NUM)] for i1 in range(ChoiceDistribution.TIME_SLOT_NUM)]
#        self.prob_arr = [[[0 for i3 in range(IC_RANGE_NUM)] for i2 in range(CG_LEVEL_NUM)] for i1 in range(TIME_SLOT_NUM)]
        self.product_cnt_arr = np.zeros((ChoiceDistribution.TIME_SLOT_NUM, ChoiceDistribution.CG_LEVEL_NUM, ChoiceDistribution.IC_RANGE_NUM))
        self.product_prob_arr = np.zeros((ChoiceDistribution.TIME_SLOT_NUM, ChoiceDistribution.CG_LEVEL_NUM, ChoiceDistribution.IC_RANGE_NUM))
        self.attr_cnt_arr = {'DT': np.zeros(ChoiceDistribution.TIME_SLOT_NUM), 'CG': np.zeros(ChoiceDistribution.CG_LEVEL_NUM), 'IC': np.zeros(ChoiceDistribution.IC_RANGE_NUM) }
        self.attr_prob_arr = {'DT': np.zeros(ChoiceDistribution.TIME_SLOT_NUM), 'CG': np.zeros(ChoiceDistribution.CG_LEVEL_NUM), 'IC': np.zeros(ChoiceDistribution.IC_RANGE_NUM) }
        self.skew =  {'DT':0, 'CG': 0, 'IC': 0 }
        self.df_attr_prob_dist = {'DT':pd.DataFrame(index=[], columns=range(ChoiceDistribution.TIME_SLOT_NUM)), 'CG': pd.DataFrame(index=[], columns=range(ChoiceDistribution.CG_LEVEL_NUM)), 'IC': pd.DataFrame(index=[], columns=range(ChoiceDistribution.IC_RANGE_NUM))}

#        print(self.prob_arr)
                           
    def addSegment(self, product, prob):
        i1 = product.dt // 15
        if(i1 > ChoiceDistribution.TIME_SLOT_NUM -1):
           i1 = ChoiceDistribution.TIME_SLOT_NUM -1

        i2 = 0
        if(product.mode == "TAXI"):
            i2 = product.wt // 10
            if(i2 > ChoiceDistribution.CG_LEVEL_NUM -1):
               i2 = ChoiceDistribution.CG_LEVEL_NUM -1
        else:
            i2 = product.cg
            if(i2 > ChoiceDistribution.CG_LEVEL_NUM -1):
               i2 = ChoiceDistribution.CG_LEVEL_NUM -1

        i3 = 0
        i3 = (int)(product.cv // 5)
        if(i3 > ChoiceDistribution.IC_RANGE_NUM -1):
           i3 = ChoiceDistribution.IC_RANGE_NUM -1

        self.product_cnt_arr[i1][i2][i3] += float(prob)

    @classmethod
    def getSegment(cls, product):
        i1 = product.dt // 15
        if(i1 > ChoiceDistribution.TIME_SLOT_NUM -1):
           i1 = ChoiceDistribution.TIME_SLOT_NUM -1

        i2 = 0
        if(product.mode == "TAXI"):
            i2 = product.wt // 10
            if(i2 > ChoiceDistribution.CG_LEVEL_NUM -1):
               i2 = ChoiceDistribution.CG_LEVEL_NUM -1
        else:
            i2 = product.cg
            if(i2 > ChoiceDistribution.CG_LEVEL_NUM -1):
               i2 = ChoiceDistribution.CG_LEVEL_NUM -1

        i3 = 0
        i3 = product.cv // 5
        if(i3 > ChoiceDistribution.IC_RANGE_NUM -1):
           i3 = ChoiceDistribution.IC_RANGE_NUM -1

        return i1, i2, i3


    def addChoiceProb(self, assortment, prob_array):
        for j in range(len(assortment.list)):
            product = assortment.list[j]
            """ for test """
#                print ("product[%d] prob\t=%f" %(j, prob_array[j]))
            self.addSegment(product, prob_array[j])

        self.choice_count += 1

    def addChoice(self, product):
        self.addSegment(product, 1.0)
        self.choice_count += 1

    def getChoiceNum(self):
        return self.choice_count

        
    def getChoiceDistribution(self):
        return self.product_cnt_arr


    def clearChoiceDistribution(self):
        self.choice_count = 0
        self.product_cnt_arr = np.zeros((ChoiceDistribution.TIME_SLOT_NUM, ChoiceDistribution.CG_LEVEL_NUM, ChoiceDistribution.IC_RANGE_NUM))
        self.product_prob_arr = np.zeros((ChoiceDistribution.TIME_SLOT_NUM, ChoiceDistribution.CG_LEVEL_NUM, ChoiceDistribution.IC_RANGE_NUM))

        
    def updateChoiceProbDistribution(self):
        if(self.choice_count < 1):
            return

        for i1 in range(ChoiceDistribution.TIME_SLOT_NUM):
            for i2 in range(ChoiceDistribution.CG_LEVEL_NUM):
                for i3 in range(ChoiceDistribution.IC_RANGE_NUM):
                    self.product_prob_arr[i1][i2][i3] = self.product_cnt_arr[i1][i2][i3] / float(self.choice_count)

        levels = [ChoiceDistribution.TIME_SLOT_NUM, ChoiceDistribution.CG_LEVEL_NUM, ChoiceDistribution.IC_RANGE_NUM]
        attr_keys = ['DT', 'CG', 'IC']

        for i in range(3):
            level = levels[i]
            attr_key = attr_keys[i]

            for j in range(levels[i]):
                self.attr_cnt_arr[attr_key][j] = 0
                self.attr_prob_arr[attr_key][j] = 0

        for i1 in range(ChoiceDistribution.TIME_SLOT_NUM):
            for i2 in range(ChoiceDistribution.CG_LEVEL_NUM):
                for i3 in range(ChoiceDistribution.IC_RANGE_NUM):
                    self.attr_cnt_arr['DT'][i1] += self.product_cnt_arr[i1][i2][i3]
                    self.attr_cnt_arr['CG'][i2] += self.product_cnt_arr[i1][i2][i3]
                    self.attr_cnt_arr['IC'][i3] += self.product_cnt_arr[i1][i2][i3]

        for i in range(3):
            level = levels[i]
            attr_key = attr_keys[i]

            df = pd.DataFrame(index=[], columns=[attr_key])
            prob = []

            for j in range(levels[i]):
                self.attr_prob_arr[attr_key][j] = self.attr_cnt_arr[attr_key][j] / float(self.choice_count)
                prob.append(self.attr_prob_arr[attr_key][j])

                for k in range(int(self.attr_cnt_arr[attr_key][j])):
                    series = pd.Series([j], index=df.columns)
                    df = df.append(series, ignore_index = True)

            self.skew[attr_key] = df.skew()[attr_key]

            se = pd.Series(prob, index = self.df_attr_prob_dist[attr_key].columns)
            self.df_attr_prob_dist[attr_key] = self.df_attr_prob_dist[attr_key].append(se, ignore_index=True)


    def outputAttrChoiceProbDistribution(self, path):
    
        if(os.path.exists(path) == False):
            os.mkdir(path)
    
        attr_keys = ['DT', 'CG', 'IC']
        for i in range(3):
            attr_key = attr_keys[i]
            filepath = path + "/choice_dist_" + attr_keys[i]  + ".csv"
            self.df_attr_prob_dist[attr_key].to_csv(filepath)


    def getProductChoiceProbDistribution(self):
        return self.product_prob_arr
    

    def getSkew(self):
        return self.skew

    def getProductDistDistance(self, another):

        # euclid distance
#        d = np.linalg.norm(self.prob_arr - another.prob_arr)
        # cosine distance
#        d = dis.cosine(self.prob_arr, another.prob_arr)
        # js_divergence divergence
        prob_arr1 = self.product_prob_arr.reshape(-1, )
#        prob_arr1 = np.ravel(self.prob_arr)
        prob_arr2 = another.product_prob_arr.reshape(-1,)
#        prob_arr2 = np.ravel(another.prob_arr)
        d = js_divergence(prob_arr1, prob_arr2)
        
        return d


    def getAttrDistDistance(self, another):
        attr_keys = ['DT', 'CG', 'IC']
        d = {'DT':0, 'CG':0, 'IC':0}
        for i in range(3):
            attr_key = attr_keys[i]
            # js_divergence divergence
            prob_arr1 = self.attr_prob_arr[attr_key].reshape(-1, )
            prob_arr2 = another.attr_prob_arr[attr_key].reshape(-1, )
            d[attr_key] = js_divergence(prob_arr1, prob_arr2)

        return d


    def getAllAttrDistDistance(self, another):
        levels = [ChoiceDistribution.TIME_SLOT_NUM, ChoiceDistribution.CG_LEVEL_NUM, ChoiceDistribution.IC_RANGE_NUM]
        attr_keys = ['DT', 'CG', 'IC']
        all_attr_prob_arr1 = np.zeros((ChoiceDistribution.TIME_SLOT_NUM + ChoiceDistribution.CG_LEVEL_NUM + ChoiceDistribution.IC_RANGE_NUM))
        all_attr_prob_arr2 = np.zeros((ChoiceDistribution.TIME_SLOT_NUM + ChoiceDistribution.CG_LEVEL_NUM + ChoiceDistribution.IC_RANGE_NUM))
        d = 0
        k = 0
        for i in range(3):
            attr_key = attr_keys[i]
            for j in range(levels[i]):
                all_attr_prob_arr1[k] = self.attr_prob_arr[attr_key][j]
                all_attr_prob_arr2[k] = another.attr_prob_arr[attr_key][j]
                k += 1

        all_attr_prob_arr1 /= 3
        all_attr_prob_arr2 /= 3
        # js_divergence divergence
        prob_arr1 = all_attr_prob_arr1.reshape(-1, )
        prob_arr2 = all_attr_prob_arr2.reshape(-1, )
        d = js_divergence(prob_arr1, prob_arr2)

        return d


    def output(self):
#        print("cnt_arr=\n")
#        print(self.cnt_arr)

        print("choice num=%d\n" % (self.choice_count))
        print("\n")

        print("[DT][CG][IC]\tprob\tcount")
        for i1 in range(ChoiceDistribution.TIME_SLOT_NUM):
            for i2 in range(ChoiceDistribution.CG_LEVEL_NUM):
                for i3 in range(ChoiceDistribution.IC_RANGE_NUM):
                    print("[%d][%d][%d]\t%f\t%d\n" % (i1, i2, i3, self.product_prob_arr[i1][i2][i3], self.product_cnt_arr[i1][i2][i3]))
        print("\n")


    def output_to_file(self, f, stdout=False):
        s = "choice num=%d\n" %(self.choice_count)
        s += "\n"

        s += "[DT][CG][IC]\tprob\tcount\n"
        for i1 in range(ChoiceDistribution.TIME_SLOT_NUM):
            for i2 in range(ChoiceDistribution.CG_LEVEL_NUM):
                for i3 in range(ChoiceDistribution.IC_RANGE_NUM):
                    if(self.product_cnt_arr[i1][i2][i3] > 0):
                        s += "[%d][%d][%d]\t%f\t%d\n" %(i1, i2, i3, self.product_prob_arr[i1][i2][i3], self.product_cnt_arr[i1][i2][i3])
        s += "\n"

        s += "[DT]\tprob\tcount\n"
        for i1 in range(ChoiceDistribution.TIME_SLOT_NUM):
            s += "[%d]\t%f\t%d\n" %(i1, self.attr_prob_arr['DT'][i1], self.attr_cnt_arr['DT'][i1])
        s += "\n"

        s += "[CG]\tprob\tcount\n"
        for i2 in range(ChoiceDistribution.CG_LEVEL_NUM):
            s += "[%d]\t%f\t%d\n" %(i2, self.attr_prob_arr['CG'][i2], self.attr_cnt_arr['CG'][i2])
        s += "\n"

        s += "[IC]\tprob\tcount\n"
        for i3 in range(ChoiceDistribution.IC_RANGE_NUM):
            s += "[%d]\t%f\t%d\n" %(i3, self.attr_prob_arr['IC'][i3], self.attr_cnt_arr['IC'][i3])
        s += "\n"

        s += "skew DT="
        s += str(self.skew['DT'])
        s += "\n"

        s += "skew CG="
        s += str(self.skew['CG'])
        s += "\n"

        s += "skew IC="
        s += str(self.skew['IC'])
        s += "\n"

        if ((f is None) == False):
            f.write(s)
            f.flush()

        if(stdout):
            print(s)
           