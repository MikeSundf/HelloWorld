# -*- coding: utf-8 -*-
"""
Created on Tue Apr 4 00:00:00 2017

@author: takuro
v1.0 create
"""

import numpy as np
import itertools
import csv

from numpy.random import *
#import random
import os.path

from choice_history import ChoiceDistribution

class BehaviorModelParam(object):
    """
    parameters of behavior choice model
    """
    coef_keys = ["beta_fare", "beta_dt", "beta_cg", "beta_ic", "beta_wt", "beta_ovtt", "beta_ivtt"]
    
    def __init__(self, beta_cg = 0.0, beta_fare = 0.0, beta_ic = 0.0, beta_dt = 0.0, beta_ivtt = 0.0, beta_ovtt = 0.0, beta_wt = 0.0):
        
        self.asc_mrt = -2
        self.asc_bus = -2
        self.asc_taxi = 0
        self.asc_shuttle_bus = -1
        self.asc_share_taxi = -0.5
        self.asc_dwell = -1.89

        self.coef = {}
        self.coef["beta_fare"] = beta_fare
        self.coef["beta_cg"] = beta_cg
        self.coef["beta_ic"] = beta_ic
        self.coef["beta_dt"] = beta_dt
        self.coef["beta_wt"] = beta_wt
        self.coef["beta_ivtt"] = beta_ivtt
        self.coef["beta_ovtt"] = beta_ovtt

        if self.asc_dwell > 0:
            print ("[WARNING] asc_dwell should be zero or negative: %f" %(self.asc_dwell))
            raise ValueError

        if self.coef["beta_ic"] < 0:
            print ("[WARNING] beta_ic should be zero or positive: %f" %(self.coef["beta_ic"]))
            raise ValueError

        if self.coef["beta_fare"] > 0:
            print ("[WARNING] beta_fare should be negative: %f" %(self.coef["beta_fare"]))
            raise ValueError

        if self.coef["beta_cg"] > 0:
            print ("[WARNING] beta_cg should be negative: %f" %(self.coef["beta_cg"]))
            raise ValueError

        if self.coef["beta_dt"] > 0:
            print ("[WARNING] beta_dt should be negative: %f" %(self.coef["beta_dt"]))
            raise ValueError

        if self.coef["beta_ivtt"] > 0:
            print ("[WARNING] beta_ivtt should be negative: %f" %(self.coef["beta_ivtt"]))
            raise ValueError

        if self.coef["beta_ovtt"] > 0:
            print ("[WARNING] beta_ovtt should be negative: %f" %(self.coef["beta_ovtt"]))
            raise ValueError

        if self.coef["beta_wt"] > 0:
            print ("[WARNING] beta_ovtt should be negative: %f" %(self.coef["beta_wt"]))
            raise ValueError

        if self.coef["beta_ivtt"] < self.coef["beta_ovtt"]:
            print ("[WARNING] beta_ovtt should be equal or less than beta_ivtt: beta_ivtt=%f, beta_ovtt=%f" %(self.coef["beta_ivtt"], self.coef["beta_ovtt"]))
            raise ValueError

        if self.coef["beta_dt"] < self.coef["beta_wt"]:
            print ("[WARNING] beta_wt should be equal or less than beta_wt: beta_ivtt=%f, beta_dt=%f" %(self.coef["beta_wt"], self.coef["beta_dt"]))
            raise ValueError


    def output(self, f, stdout=False):
        s = ""
        for key, value in self.coef.items():
            s += "%s\t= %f\n" % (key, value)
        s += "\n"

        if((f is None) == False):
            f.write(s)
            f.flush()

        if(stdout):
            print(s)


    def to_csv(self, f):
        csvWriter = csv.writer(f, lineterminator='\n')

        header = []
        for key, value in self.params.coef.items():
            header.append(key)

        csvWriter.writerow(header)

        row = []
        for key, value in self.params.coef.items():
            row.append(value)
        csvWriter.writerow(row)


    def get(self, key):
        if key not in BehaviorModelParam.coef_keys:
            raise ValueError
        return self.coef[key]

    def set(self, key, value):
        if key not in BehaviorModelParam.coef_keys:
            raise ValueError
        self.coef[key] = value

    def add(self, another, weight):

        for key, value in self.coef.items():
#            print ("add coef[%s]=%f, %f" % (key, value, another.coef[key])
            self.coef[key] += weight * another.coef[key]

    def div(self, denom):
        for key, value in self.coef.items():
#            print ("div coef[%s]=%f / %f" % (key, value, denom)
            self.coef[key] /= denom

    def getError(self, updated_params):
        error = {}
        for key, value in self.coef.items():
#            print ("error coef[%s]=%f - %f" % (key, updated_params.coef[key], value)
            error[key] = updated_params.coef[key] - self.coef[key]

        return error


    def getErrorIndexes(self, updated_params, value_range):
        
        """
        It is a function to calculate three errors:
            1. normalized error, which is calculated by (learnt value - true value)/(max of true value - min of true value) 
            2. absolute error, which is calculated by (learnt value - true value)
            3. (learnt value - true value)/(true value of the particular user)
        """
        error = self.getError(updated_params)

        error_norm = {}
        for key, value in self.coef.items():
            range_size = np.max(value_range[key]) - np.min(value_range[key])
            if(range_size != 0):
                error_norm[key] = error[key] / range_size
            else:
                if(error[key] == 0):
                    error_norm[key] = 0
                else:
                    error_norm[key] = float("inf")
        
        error_rate = {}
        for key, value in self.coef.items():
            error_rate[key] = abs(error[key]) / abs(self.coef[key])
        
        return error, error_norm, error_rate


    def updateParams(self, new_params, weight):

        if weight < 0:
            print ("[ERROR] weight should be between zero and 1: %f" %(weight))
            raise ValueError

        if weight > 1:
            print ("[ERROR] weight should be between zero and 1: %f" %(weight))
            raise ValueError

        posterior_params = BehaviorModelParam(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        for key, value in self.coef.items():
            posterior_params.coef[key] = (1.0 - weight) * self.coef[key] + weight * new_params.coef[key]

        return posterior_params


class UserAgent(object):
    """
    Behavior choice model
    """
    def __init__(self, id, beta_cg, beta_fare, beta_ic, beta_dt, beta_ivtt, beta_ovtt, beta_wt):
        self.id = id
        self.params = BehaviorModelParam(beta_cg, beta_fare, beta_ic, beta_dt, beta_ivtt, beta_ovtt, beta_wt)
        self.choice_history = ChoiceDistribution()


    def output(self, f, stdout=False):
        s = "user(%s):" % (self.id)

        if((f is None) == False):
            f.write(s)
            f.write("\n")

        if(stdout):
            print(s)

        self.params.output(f, stdout)


    def to_csv(self, f):
        csvWriter = csv.writer(f, lineterminator='\n')

        header = ['user_id']
        for key, value in self.params.coef.items():
            header.append(key)

        csvWriter.writerow(header)

        row = [self.id]
        for key, value in self.params.coef.items():
            row.append(value)
        csvWriter.writerow(row)


    def getUtility(self, product):
        """
        Function to calculate the utility of product
        """
        asc_mode = 0
        if product.mode == "TRAIN":
            asc_mode = self.params.asc_mrt
        if product.mode == "BUS":
            asc_mode = self.params.asc_bus
        if product.mode == "TAXI":
            asc_mode = self.params.asc_taxi
        if product.mode == "SHUTTLE":
            asc_mode = self.params.asc_shuttle_bus
        if product.mode == "STAXI":
            asc_mode = self.params.asc_share_taxi

        asc_dwell = 0            # Initializing asc_dwell => asc_dwell corresponds to decrease in utility when user is asked to dwell
        if(product.dt > 0):
            asc_dwell = self.params.asc_dwell    # Value of asc_dwell when DT > 0 

        return asc_mode + asc_dwell + self.params.coef["beta_fare"] * product.fare + self.params.coef["beta_ivtt"] * product.ivtt + self.params.coef["beta_dt"] * product.dt + self.params.coef["beta_ovtt"] * product.ovtt + self.params.coef["beta_wt"] * product.wt + self.params.coef["beta_cg"] * product.cg * product.cg + self.params.coef["beta_ic"] * product.cv


    def getChoiceProbability(self, assortment):
        """
        Function to calculate the utility and choice probability of each product
        """

        mu = 1
        product_num = len(assortment.list)

        V_array = np.zeros(product_num)                       ### Initializing the Utility values array with 0 values   
        expV_array = np.zeros(product_num)                    ### Initializing the Exponential Utility values array with 0 values
        prob_array = np.zeros(product_num)

        for i in range(product_num):
            product = assortment.list[i]
            V_array[i] = self.getUtility(product)

        sum = 0
        for i in range(product_num):
            expV_array[i] = np.exp(mu * V_array[i])
            sum = sum + expV_array[i]

        for i in range(product_num):
            prob_array[i] = expV_array[i] / sum

        max_prob_idx = 0
        max_prob = 0
        for i in range(product_num):
            if(prob_array[i] > max_prob):
                max_prob_idx = i
                max_prob = prob_array[i]
        
        return (prob_array, V_array, max_prob_idx) 


    def selectProduct(self, assortment):
        """
        select a product from assortment
        """
        selected_product_idx = -1

        prob_array, V_array, max_prob_idx = self.getChoiceProbability(assortment)

        accum = 0.0
        for j in range(len(assortment.list)):
            product = assortment.list[j]
#            product.output()
#            print ("product[%d] prob=%f" %(j, prob_array[j])

            accum += prob_array[j]

#        print("total prob=%f" % (accum))

#        r = random.random()
        r = rand()
#        print("random value=%f" %(r))

        accum = 0.0
        selected_product_prob = 0.0
        selected_flag = False
        for j in range(len(assortment.list)):
            product = assortment.list[j]
#            product.output()
#            print ("prob=%f" %(prob_array[j])

#            # assume that actual user select product with highest choice probability
#            if(j == max_prob_idx):
#                selected_flag = TRUE

            # simulate that actual user select product with choice probability
            accum += prob_array[j]
            if(accum >= r):
                selected_flag = True

            if(selected_flag == True):
                selected_product_idx = j
                selected_product_prob = prob_array[j]
                break

        return (selected_product_idx, selected_product_prob)

class DummyUserManager(object):
    """
    manager of dummy users
    """
    beta_range = {}
    beta_ic_set = (0.05, 0.15, 0.25, 0.35) # 4 level
    vot_ivtt_set = (0.1, 0.2, 0.3, 0.4, 0.5) # 5 level
    coef_ovtt_set = (1.5, 2.0, 2.5, 3.0, 3.5) # 5 level
    coef_wt_set = (1.6, 1.8, 2.0, 2.2, 2.4) # 5 level
#    coef_dt_set = (1.6, 1.8, 2.0, 2.2, 2.4) # 5 level
    coef_dt_set = (0.4, 0.6, 0.8, 1.0) # 4 level
    vocg_set = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0) # 6 level
    
    def __init__(self):
        
        self.user_map = {}

    def generate_dummy_users(self):
    
        level_num = 6
    
        #################### Creating all possible combinations of the parameters ################
        segments = list(itertools.product([0,1,2,3,4,5], repeat=level_num)) 
        # segments = list(itertools.product([0,1,2,3,4], repeat=10)) 
        ##########################################################################################
     
        beta = ["beta_dt", "beta_cg", "beta_ic", "beta_wt", "beta_ovtt", "beta_ivtt"]
    
        DummyUserManager.beta_range["beta_cg"] = [-2.79, -0.65]
        DummyUserManager.beta_range["beta_fare"] = [-0.4, -0.1]
        DummyUserManager.beta_range["beta_ic"] = [0.1, 0.4]
        DummyUserManager.beta_range["beta_dt"] = [-0.04, -0.004]
        DummyUserManager.beta_range["beta_wt"] = [-0.11, -0.01]
        DummyUserManager.beta_range["beta_ovtt"] = [-0.11, -0.01]
        DummyUserManager.beta_range["beta_ivtt"] = [-0.057, -0.0057]

        self.user_map = {}

        beta_array = {}
        for i in range(len(beta)):
            lower = DummyUserManager.beta_range[beta[i]][0]
            upper = DummyUserManager.beta_range[beta[i]][1]
            interval = float(upper - lower) / (level_num - 1)
            array = []
            for j in range(level_num):
                array.append(lower + j * interval)
    
            beta_array[beta[i]] = array
            print ("%s: %s" % (beta[i], beta_array[beta[i]]))
    
    
        for i in range(len(segments)):
    #        print segments[i]
    
            beta_dt = beta_array["beta_dt"][segments[i][0]]
            beta_cg = beta_array["beta_cg"][segments[i][1]]
            beta_ic = beta_array["beta_ic"][segments[i][2]]
            beta_wt = beta_array["beta_wt"][segments[i][3]]
            beta_ovtt = beta_array["beta_ovtt"][segments[i][4]]
            beta_ivtt = beta_array["beta_ivtt"][segments[i][5]]
    
            if beta_ivtt < beta_ovtt:
                continue
    
            if beta_dt < beta_wt:
                continue
    
            id = str(segments[i])
            beta_fare = - beta_ic
            user = UserAgent(id, beta_cg, beta_fare, beta_ic, beta_dt, beta_ivtt, beta_ovtt, beta_wt)
            self.user_map[id] = user
            
        print ("number of dummy users: %d" %(len(self.user_map)))
    
        return (self.user_map)


    def generate_dummy_users3(self):

        DummyUserManager.beta_range = {}
        self.user_map = {}
    

        beta_fare = - 0.5
        beta_ivtt = - 0.1
        beta_ic_0 = 0.1
        beta_cg_0 = -0.75
        beta_dt_0 = -0.057
        
        coef_ovtt = 1.7
        beta_ovtt = coef_ovtt * beta_ivtt
        coef_wt = 1.0
        beta_wt = coef_wt * beta_ivtt

        coef_ic_range = (1, 3, 5)
        coef_cg_range = (0.5, 1, 1.5, 2)
        coef_dt_range = (0.5, 1, 1.5, 2)

        i = 0
        for coef_ic, coef_cg, coef_dt in itertools.product(coef_ic_range, coef_cg_range, coef_dt_range):

            beta_ic = coef_ic * beta_ic_0
            beta_cg = coef_cg * beta_cg_0
            beta_dt = coef_dt * beta_dt_0
            
            if beta_ivtt < beta_ovtt:
                continue
    
            if beta_dt < beta_wt:
                continue
    
            id = "du[%d]" % i
    
            user = UserAgent(id, beta_cg, beta_fare, beta_ic, beta_dt, beta_ivtt, beta_ovtt, beta_wt)
    #        ref_model.output()
            self.user_map[id] = user
            i += 1
            
        print ("number of dummy users: %d" %(len(self.user_map)))
    
        for j in range(len(BehaviorModelParam.coef_keys)):
            key = BehaviorModelParam.coef_keys[j]
            DummyUserManager.beta_range[key] = [float("inf"), float("-inf")]
    
        for j in range(len(BehaviorModelParam.coef_keys)):
            key = BehaviorModelParam.coef_keys[j]
            for id, dummy_user in self.user_map.items():
                value = dummy_user.params.get(key)
                if DummyUserManager.beta_range[key][0] > value:
                    DummyUserManager.beta_range[key][0] = value
                if DummyUserManager.beta_range[key][1] < value:
                    DummyUserManager.beta_range[key][1] = value
    
        return (self.user_map)

    def generate_dummy_users2(self):
    
        DummyUserManager.beta_range = {}
        self.user_map = {}
    
        i = 0
        for beta_ic, vot_ivtt, coef_ovtt, coef_wt, coef_dt, vocg in itertools.product(DummyUserManager.beta_ic_set, DummyUserManager.vot_ivtt_set, DummyUserManager.coef_ovtt_set, DummyUserManager.coef_wt_set, DummyUserManager.coef_dt_set, DummyUserManager.vocg_set):
            beta_fare = - beta_ic
            beta_ivtt = vot_ivtt * beta_fare
            beta_cg = vocg * beta_fare
            beta_dt = coef_dt * beta_ivtt
            beta_ovtt = coef_ovtt * beta_ivtt
            beta_wt = coef_wt * beta_ivtt
            
            if beta_ivtt < beta_ovtt:
                continue
    
            if beta_dt < beta_wt:
                continue
    
            id = "du[%d]" % i
    
            user = UserAgent(id, beta_cg, beta_fare, beta_ic, beta_dt, beta_ivtt, beta_ovtt, beta_wt)
    #        ref_model.output()
            self.user_map[id] = user
            i += 1
            
        print ("number of dummy users: %d" %(len(self.user_map)))
    
        for j in range(len(BehaviorModelParam.coef_keys)):
            key = BehaviorModelParam.coef_keys[j]
            DummyUserManager.beta_range[key] = []
    
        for id, dummy_user in self.user_map.items():
            for j in range(len(BehaviorModelParam.coef_keys)):
                key = BehaviorModelParam.coef_keys[j]
                value = dummy_user.params.get(key)
                if len(DummyUserManager.beta_range[key]) == 0:
                    DummyUserManager.beta_range[key] = [value, value]
                else:
                    if DummyUserManager.beta_range[key][0] > value:
                        DummyUserManager.beta_range[key][0] = value
                    if DummyUserManager.beta_range[key][1] < value:
                        DummyUserManager.beta_range[key][1] = value
    
        return (self.user_map)

    def get_average_params(self):
    
        beta_ic_list = list(DummyUserManager.beta_ic_set)
        beta_ic = (beta_ic_list[0] + beta_ic_list[len(beta_ic_list) -1]) / 2

        vot_ivtt_list = list(DummyUserManager.vot_ivtt_set)
        vot_ivtt = (vot_ivtt_list[0] + vot_ivtt_list[len(vot_ivtt_list) -1]) / 2

        coef_ovtt_list = list(DummyUserManager.coef_ovtt_set)
        coef_ovtt = (coef_ovtt_list[0] + coef_ovtt_list[len(coef_ovtt_list) -1]) / 2

        coef_wt_list = list(DummyUserManager.coef_wt_set)
        coef_wt = (coef_wt_list[0] + coef_wt_list[len(coef_wt_list) -1]) / 2

        coef_dt_list = list(DummyUserManager.coef_dt_set)
        coef_dt = (coef_dt_list[0] + coef_dt_list[len(coef_dt_list) -1]) / 2

        vocg_list = list(DummyUserManager.vocg_set)
        vocg = (vocg_list[0] + vocg_list[len(vocg_list) -1]) / 2

    
        beta_fare = - beta_ic
        beta_ivtt = vot_ivtt * beta_fare
        beta_ovtt = coef_ovtt * beta_ivtt
        beta_wt = coef_wt * beta_ivtt
        beta_dt = coef_dt * beta_ivtt
        beta_cg = vocg * beta_fare
            
        if beta_ivtt < beta_ovtt:
            raise ValueError
    
        if beta_dt < beta_wt:
            raise ValueError
    
        params = BehaviorModelParam(beta_cg, beta_fare, beta_ic, beta_dt, beta_ivtt, beta_ovtt, beta_wt)
        
        return params

    def output_dummy_users(self, path):
    
        if(os.path.exists(path) == False):
            os.mkdir(path)
    
        f = open(path + "/dummy_users.csv", 'w')
    
        # header
        f.write('id,')
        for j in range(len(BehaviorModelParam.coef_keys)):
            key = BehaviorModelParam.coef_keys[j]
            f.write('%s' % key)
            if(j != len(BehaviorModelParam.coef_keys) -1):
                f.write(',')
    
        f.write('\n')
    
        for id, dummy_user in self.user_map.items():
            f.write('%s,' % id)
            for j in range(len(BehaviorModelParam.coef_keys)):
                key = BehaviorModelParam.coef_keys[j]
                value = dummy_user.params.get(key)
                f.write('%f' % value)
                if(j != len(BehaviorModelParam.coef_keys) -1):
                    f.write(',')
            f.write('\n')
    
        f.close()
