# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 14:20:11 2017

@author: takuro

"""
import sys
import numpy as np
import itertools
import random
import math
import csv

class Assortment(object):
    """
    class of assortment of products
    """
    def __init__(self, id):
        self.id = id
        self.list = []

    def add(self, product):
        self.list.append(product)

    def get(self, idx):
        return self.list[idx]


    def output(self, f, stdout=False):
        s = 'assortment(%s)\n' %(self.id)

        if((f is None) == False):
            f.write(s)
            f.flush()

        for i in range(len(self.list)):
            product = self.list[i]
            product.output(f, stdout)


    def to_csv(self, f):
        csvWriter = csv.writer(f, lineterminator='\n')

        for i in range(len(self.list)):
            product = self.list[i]
            row = [self.id, product.seq, product.id, product.mode, product.fare, product.dt, product.ivtt, product.ovtt, product.wt, product.cg, product.cv]
            csvWriter.writerow(row)


class Product(object):
    """
    class of product(ride option)
    """
    attr_keys = ["mode", "fare", "ivtt", "ovtt", "dt", "wt", "cg", "cv"]

    def __init__(self, id, seq, mode, fare, dt, ivtt, ovtt, wt, cg, cv):
        self.id = id
        self.seq = seq
        self.mode = mode
        self.fare = fare
        self.dt = dt
        self.ivtt = ivtt
        self.ovtt = ovtt
        self.wt = wt
        self.cg = cg
        self.cv = cv

    def output(self, f, stdout=False):
        s = "product(%s): seq=%d, mode=%s, fare=%f, dt=%d, ivtt=%d, ovtt=%d, wt=%d, cg=%d, cv=%f" %(self.id, self.seq, self.mode, self.fare, self.dt, self.ivtt, self.ovtt, self.wt, self.cg, self.cv)

        if((f is None) == False):
            f.write(s)
            f.write("\n")
            f.flush()
        
        if(stdout):
            print(s)
            

    def to_csv(self, f):
        csvWriter = csv.writer(f, lineterminator='\n')

        header = ['id', 'seq']
        for i in range(len(attr_keys)):
            header.append(attr_keys[i])

        csvWriter.writerow(header)

        row = [self.id, self.seq, self.mode, self.fare, self.dt, self.ivtt, self.ovtt, self.wt, self.cg, self.cv]
        csvWriter.writerow(row)

