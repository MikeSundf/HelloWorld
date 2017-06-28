# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 14:20:11 2017

@author: takuro

"""
import psycopg2
import psycopg2.extras
import sys
import numpy as np
import pandas as pd
import random
import math
import os.path
import copy
import csv

from config import Config
from configparser import ConfigParser
from user_agent import UserAgent
from user_agent import BehaviorModelParam
from user_agent import DummyUserManager
#from user_agent import RangeCount
from histogram import Histogram
from choice_history import ChoiceDistribution

from product import Assortment
from product import Product

config = Config()

def config(filename = os.path.join(os.path.dirname(__file__),'database.ini'), section= 'postgresql'):
    #create a parser
    parser = ConfigParser()
    #read config file
    parser.read(filename)
    #get section, default to postgresql
    db = {}
    if parser.has_section(section):
        parameters = parser.items(section)
        for parameter in parameters:
            db[parameter[0]]=parameter[1]
    else:
        raise Exception('Section {0} not found in {1} file'.format(section, filename))
        
    return db


def output_stdout_file(f, log=""):
    """
    output to fine and stdout
    """
    sys.stdout.write(log+"\n")
    sys.stdout.flush()
    
    f.write(log+"\n")
    f.flush()
    

def create_random_assortment(id):
    """
    create assortment randomly for test
    """
    assortment = Assortment(id)

    cnt = 0
    mode_name = ["TRAIN","BUS","TAXI"]
    for i in range(0, 3): # TRAIN, BUS, TAXI
        mode = mode_name[i]
        # eliminate taxi product
        if i == 2:
            break

        ivtt = random.randint(10,60)
        ovtt = random.randint(0,30)
        fare = random.randint(1,30)
        #
        for j in range(0, 8): # 15min * 8 time slots

            """
            if (i != 2):
                cg = 0
            """
            id = mode + "_" + str(j)
            seq = cnt
            cg = random.randint(0,2)
            wt = random.randint(0,30)

            ic = 0
            if (cg == 0):
                ic = 0
            elif (cg == 1):
                ic = random.randint(0,5)
            else:
                ic = random.randint(5,10)

            dt = j * 15
            product = Product(id, seq, mode, fare, dt, ivtt, ovtt, wt, cg, ic)
#            product.output()
            assortment.add(product)
            cnt = cnt + 1
    
    return assortment


def create_assortment(id):
    """
    create assortment for test
    """
    assortment = Assortment(id)

    cnt = 0

    mode = "TRAIN"
    """
    if i == 0:
    elif i == 1:
    else:
    """
    ivtt = 30
    ovtt = 10
    wt = 0
    fare = 5
    cg_pattern = [2,2,2,2,1,1,0,0]
    for j in range(0, 8): # 15min * 8 time slots

        id = mode + "_" + str(j)
        seq = cnt
        cg = cg_pattern[j]

        ic = int(j) * 2
        if (ic > 12):
            ic = 12
        dt = j * 15
        product = Product(id, seq, mode, fare, dt, ivtt, ovtt, wt, cg, ic)
#            product.output()
        assortment.add(product)
        cnt = cnt + 1
    
    mode = "BUS"

    ivtt = 40
    ovtt = 5
    wt = 0
    fare = 4
    cg_pattern = [2,2,1,1,1,0,0,0]
    for j in range(0, 8): # 15min * 8 time slots

        id = mode + "_" + str(j)
        seq = cnt
        cg = cg_pattern[j]

        ic = int(j) * 2
        if (ic > 12):
            ic = 12
        dt = j * 15
        product = Product(id, seq, mode, fare, dt, ivtt, ovtt, wt, cg, ic)
#            product.output()
        assortment.add(product)
        cnt = cnt + 1

    """ eliminate taxi product
    mode = "TAXI"
    ivtt = 10
    ovtt = 0
    cg = 0
    wt_pattern = [30,20,20,10,10,5,0,0]
    fare = 30
    for j in range(0, 8): # 15min * 8 time slots

        id = mode + "_" + str(j)
        seq = cnt
        wt = wt_pattern[j]

        ic = int(j) * 2
        if (ic > 12):
            ic = 12
        dt = j * 15
        product = Product(id, seq, mode, fare, dt, ivtt, ovtt, wt, cg, ic)
#            product.output()
        assortment.add(product)
        cnt = cnt + 1
    """

    return assortment


class ParamDistribution(object):
    """
    distribution of parameters
    """
    beta_range = {}
    
    def __init__(self):
       
        self.histograms = {}

    def initialize(self, level_num, beta_range):
    
        self.histograms = {}
        for i in range(len(BehaviorModelParam.coef_keys)):
            lower = beta_range[BehaviorModelParam.coef_keys[i]][0]
            upper = beta_range[BehaviorModelParam.coef_keys[i]][1]
            print ("histogram[%s]" % (BehaviorModelParam.coef_keys[i]))
            self.histograms[BehaviorModelParam.coef_keys[i]] = Histogram(lower, upper, level_num)
    #        print "%s: %s" % (beta[i], beta_array[beta[i]])
        return self.histograms


    def add_param(self, params):
    
        for i in range(len(BehaviorModelParam.coef_keys)):
            histogram = self.histograms[BehaviorModelParam.coef_keys[i]]
            histogram.update(params.coef[BehaviorModelParam.coef_keys[i]])
    
        return self.histograms

    def getMostFrequentRange(self, key):
        histogram = self.histograms[key]
        frequent_range_list = histogram.get_frequent_range()
        if len(frequent_range_list) < 1:
            return None
        return frequent_range_list[0]

    def output(self, user):
        """
        output histogram
        """
        for i in range(len(BehaviorModelParam.coef_keys)):
    
            histogram = self.histograms[BehaviorModelParam.coef_keys[i]]
            value = user.params.get(BehaviorModelParam.coef_keys[i])
            print ("\nhistogram[%s]:" % (BehaviorModelParam.coef_keys[i]))
            for j in range(len(histogram.list)):
                range_count = histogram.list[j]
                mark = ""
                if value >= range_count.lower and value < range_count.upper:
                    mark = "(*)" + str(value)
                print ("range[%f, %f]: count=%d %s" % (range_count.lower, range_count.upper, range_count.count, mark))
    

def output_est_error(path, actual_user, params_history, beta_range):
    """
    output estimation error
    """

    if(os.path.exists(path) == False):
        os.mkdir(path)

    f1 = open(path + "/param_learnt.csv", 'w')
    f2 = open(path + "/param_error.csv", 'w')
    f3 = open(path + "/param_error_norm.csv", 'w')
    f4 = open(path + "/param_error_rate.csv", 'w')
    
    files = [f1, f2, f3, f4]

    # header
    for j in range(len(BehaviorModelParam.coef_keys)):
        key = BehaviorModelParam.coef_keys[j]
        for i in range(len(files)):
            files[i].write('%s' % key)
        if(j != len(BehaviorModelParam.coef_keys) -1):
            for i in range(len(files)):
                files[i].write(',')

    for i in range(len(files)):
        files[i].write('\n')

    for i in range(len(params_history)):
        params = params_history[i]
        error, error_norm, error_rate = actual_user.params.getErrorIndexes(params, beta_range)
        for j in range(len(BehaviorModelParam.coef_keys)):
            key = BehaviorModelParam.coef_keys[j]
            value = params.get(key)
            f1.write('%f' % (value))
            f2.write('%f' % (error[key]))
            f3.write('%f' % (error_norm[key]))
            f4.write('%f' % (error_rate[key]))

            if(j != len(BehaviorModelParam.coef_keys) -1):
                for i in range(len(files)):
                    files[i].write(',')

        for i in range(len(files)):
            files[i].write('\n')

    for i in range(len(files)):
        files[i].close()


def output_est_error2(path, actual_user, user_history, beta_range):
    """
    output estimation error
    """

    if (os.path.exists(path) == False):
        os.mkdir(path)

    f = open(path + "/users.csv", 'w')
    for i in range(len(user_history)):
        user = user_history[i]
        f.write('%s\n' % (user.id))

    f1 = open(path + "/param_learnt.csv", 'w')
    f2 = open(path + "/param_error.csv", 'w')
    f3 = open(path + "/param_error_norm.csv", 'w')
    f4 = open(path + "/param_error_rate.csv", 'w')

    files = [f1, f2, f3, f4]

    # header
    for j in range(len(BehaviorModelParam.coef_keys)):
        key = BehaviorModelParam.coef_keys[j]
        for i in range(len(files)):
            files[i].write('%s' % key)
        if (j != len(BehaviorModelParam.coef_keys) - 1):
            for i in range(len(files)):
                files[i].write(',')

    for i in range(len(files)):
        files[i].write('\n')

    for i in range(len(user_history)):
        user = user_history[i]
        error, error_norm, error_rate = actual_user.params.getErrorIndexes(user.params, beta_range)
        for j in range(len(BehaviorModelParam.coef_keys)):
            key = BehaviorModelParam.coef_keys[j]
            value = user.params.get(key)
            f1.write('%f' % (value))
            f2.write('%f' % (error[key]))
            f3.write('%f' % (error_norm[key]))
            f4.write('%f' % (error_rate[key]))

            if (j != len(BehaviorModelParam.coef_keys) - 1):
                for i in range(len(files)):
                    files[i].write(',')

        for i in range(len(files)):
            files[i].write('\n')

    for i in range(len(files)):
        files[i].close()


def algo2(actual_user, n_epoch, rand_assort):
    """
    algorithm to learn behavior parameters(based on joint choice probability)
    actual_user: actual user
    n_epoch: epoch
    rand_assort: test with random assortment(true) or one assortment(false) 
    """

    global config

    f = config.log_file

    dummy_user_map = config.dum.user_map

    same_choice_prob_map_history = []

    est_params_history = []

    ave_params = config.dum.get_average_params()
    est_params_history.append(ave_params)

    if(rand_assort == False):
        assortment = create_assortment(0)
        #    assortment = create_random_assortment(0)
        assortment.output(None, True)
        assortment.to_csv(config.assortment_file)

        prob_array, V_array, max_prob_idx = actual_user.getChoiceProbability(assortment)

        accum = 0.0
        for j in range(len(assortment.list)):
#                product = assortment.list[j]
#            print ("product[%d] prob=%f" %(j, prob_array[j]))
            accum += prob_array[j]

#        print("total prob=%f" % (accum))

    same_choice_prob_history = {}
    for id, dummy_user in dummy_user_map.items():
        same_choice_prob_history[id] = []

    for it in range(0, n_epoch):

        output_stdout_file (f, "\n---------- iteration %d ----------\n" %(it))

        if(rand_assort == True):
            assortment = create_random_assortment(it)
            assortment.output(None, True)
            assortment.to_csv(config.assortment_file)

            prob_array, V_array, max_prob_idx = actual_user.getChoiceProbability(assortment)

            accum = 0.0
            for j in range(len(assortment.list)):
#                product = assortment.list[j]
                print ("product[%d] prob=%f" %(j, prob_array[j]))

                accum += prob_array[j]

            print("total prob=%f" % (accum))
       
        selected_product_idx, selected_product_prob = actual_user.selectProduct(assortment)

        if selected_product_idx >= 0:
            output_stdout_file (f, "<<<<<<<< selected (prob=%f) >>>>>>>>" %(selected_product_prob))
            product = assortment.get(selected_product_idx)
            product.output(f, True)
            output_stdout_file (f, "")
            actual_user.choice_history.addChoice(product)
        else:
            output_stdout_file (f, "<<<<<<<< [rejected] >>>>>>>>")
            continue

        same_choice_prob_map = {}
        for id, dummy_user in dummy_user_map.items():
#            print("dummy_user(%s)" % dummy_user.id)
            prob_array, V_array, max_prob_idx = dummy_user.getChoiceProbability(assortment)
            same_choice_prob_history[id].append(prob_array[selected_product_idx])
            same_choice_prob_map[id] = prob_array[selected_product_idx]

#       find the dummy user with highest joint choice probability

        max_joint_prob_user_map = {}
        max_joint_prob_user_id = ""
        max_joint_prob_log = -float("inf")
        for user_id, prob_list in same_choice_prob_history.items():
            joint_prob_log = 0
            for j in range(len(prob_list)):
                prob = prob_list[j]
                if prob == 0:
                    joint_prob_log = -float("inf")
                    break

                joint_prob_log += math.log(prob)

            max_joint_prob_user_map[user_id] = joint_prob_log

            if joint_prob_log >= max_joint_prob_log:
                max_joint_prob_log = joint_prob_log
                max_joint_prob_user_id = user_id
    
        max_joint_prob_user = dummy_user_map[max_joint_prob_user_id]

        similar_users = sorted(max_joint_prob_user_map.items(), key=lambda x: x[1], reverse=True)
        output_stdout_file (f, "similar_users:")
        for i in range(len(similar_users)):
            output_stdout_file (f, "user(%s): %f" % (similar_users[i][0], similar_users[i][1]))
            if i > 9:
                break
        output_stdout_file (f)
        
#        if len(max_joint_prob_user_list) == 0:
#            continue

        est_params = copy.deepcopy(max_joint_prob_user.params)

        same_choice_prob_map_history.append(same_choice_prob_map)

        est_params_history.append(est_params)

    similar_users = sorted(max_joint_prob_user_map.items(), key=lambda x: x[1], reverse=True)
    output_stdout_file (f, "similar_users:")
    for i in range(len(similar_users)):
        output_stdout_file (f, "user(%s): %f" % (similar_users[i][0], similar_users[i][1]))
        similar_user = dummy_user_map[similar_users[i][0]]
        similar_user.params.output(f, True)

        if i > 9:
            break
    output_stdout_file (f)

    """
    output learnt parameters
    """
    learnt_params = est_params_history[len(est_params_history) -1]
                                       
    output_stdout_file(f, "\nlearnt parameters:")
    learnt_params.output(f, True)

    """
    output learnt parameters and errors
    """
    output_dir = config.output_dir + "/err1"
    output_est_error(output_dir, actual_user, est_params_history, config.dum.beta_range)


def algo4(actual_user, n_epoch, rand_assort):
    """
    algorithm to learn behavior parameters(based on the similarity of choice distribution)
    actual_user: actual user
    n_epoch: epoch
    rand_assort: test with random assortment(true) or one assortment(false) 
    """

    global config

    f_log = config.log_file

    attr_keys = ['DT', 'CG', 'IC']

    dummy_user_map = config.dum.user_map

    skew_history = pd.DataFrame(index=[], columns=['DT', 'CG', 'IC'])

    est_params_history = []
    min_attr_dist_distance_user = {'DT':[], 'CG':[], 'IC':[]}
    min_product_dist_distance_user = []
    min_all_attr_dist_distance_user = []

    ave_params = config.dum.get_average_params()
    est_params_history.append(ave_params)

    if(rand_assort == False):
        assortment = create_assortment(0)
        #    assortment = create_random_assortment(0)
#        assortment.output()
        assortment.to_csv(config.assortment_file)

        prob_array, V_array, max_prob_idx = actual_user.getChoiceProbability(assortment)

        accum = 0.0
        for j in range(len(assortment.list)):
#                product = assortment.list[j]
#            print ("product[%d] prob=%f" %(j, prob_array[j]))
            accum += prob_array[j]

#        print("total prob=%f" % (accum))

    product_choice_prob_distance_map = {}
    all_attr_choice_prob_distance_map = {}
    attr_choice_prob_distance_map = {'DT': {}, 'CG': {}, 'IC': {}}

    for it in range(0, n_epoch):

        output_stdout_file (f_log, "\n---------- actual user(%s), random_assort=%d, iteration %d / %d ----------\n" %(actual_user.id, rand_assort, it, n_epoch))

        if(rand_assort == True):
            assortment = create_random_assortment(it)
#            assortment.output()
            assortment.to_csv(config.assortment_file)

            prob_array, V_array, max_prob_idx = actual_user.getChoiceProbability(assortment)

            """ for test
            accum = 0.0
            for j in range(len(assortment.list)):
#                product = assortment.list[j]
#                print ("product[%d] prob=%f" %(j, prob_array[j]))
                accum += prob_array[j]

#            print("total prob=%f" % (accum))
            """

        # simulate that an actual user selects product among an assortment
        selected_product_idx, selected_product_prob = actual_user.selectProduct(assortment)

        if selected_product_idx >= 0:
            output_stdout_file (f_log, "<<<<<<<< selected (prob=%f) >>>>>>>>" %(selected_product_prob))
            product = assortment.get(selected_product_idx)
            product.output(f_log, True)
            output_stdout_file (f_log, "")
            actual_user.choice_history.addChoice(product)
            actual_user.choice_history.updateChoiceProbDistribution()
        else:
            output_stdout_file (f_log, "<<<<<<<< [rejected] >>>>>>>>")
            continue

        min_product_d = float("inf")
        min_product_user = None
        min_all_attr_d = float("inf")
        min_all_attr_user = None
        min_attr_d = {'DT':float("inf"), 'CG':float("inf"), 'IC':float("inf")}
        min_attr_user = {'DT':None, 'CG':None, 'IC':None}

        # for each dummy user
        for id, dummy_user in dummy_user_map.items():
#            print("dummy_user(%s)" % dummy_user.id)
            # calculate dummy user's choice probability of each product in an assortment
            prob_array, V_array, max_prob_idx = dummy_user.getChoiceProbability(assortment)

            dummy_user.choice_history.addChoiceProb(assortment, prob_array)
            dummy_user.choice_history.updateChoiceProbDistribution()


            """ for test
            for j in range(len(assortment.list)):
                product = assortment.list[j]
#                print ("product[%d] prob\t=%f" %(j, prob_array[j]))
            """""

            d = dummy_user.choice_history.getProductDistDistance(actual_user.choice_history)
            """ for test
            print ("choice history = %d" %(dummy_user.choice_history.count))
            print ("distance of dummy user(%s) = %f" %(id, d))
            """
            product_choice_prob_distance_map[id] = d

            if(d < min_product_d):
                min_product_user = dummy_user
                min_product_d = d

            d = dummy_user.choice_history.getAttrDistDistance(actual_user.choice_history)

            for i in range(3):
                attr_key = attr_keys[i]

                if(d[attr_key] < min_attr_d[attr_key]):
                    min_attr_user[attr_key] = dummy_user
                    min_attr_d[attr_key] = d[attr_key]

                attr_choice_prob_distance_map[attr_key][id] = d[attr_key]

            d = dummy_user.choice_history.getAllAttrDistDistance(actual_user.choice_history)
            if(d < min_all_attr_d):
                min_all_attr_user = dummy_user
                min_all_attr_d = d

        if(min_product_user is None):
            print ("no dummy user with minimum distance")
            continue
                
#       find the dummy user with most similar choice history

        actual_user.choice_history.output_to_file(f_log, True)

        skew = actual_user.choice_history.getSkew()
        series = pd.Series([skew['DT'], skew['CG'], skew['IC']], index=skew_history.columns)
        skew_history = skew_history.append(series, ignore_index = True)
#        print(skew_history)

        """ output similar users
        similar_users = sorted(product_choice_prob_distance_map.items(), key=lambda x: x[1], reverse=False)
        output_stdout_file (f, "similar_users:")
        for i in range(len(similar_users)):
            output_stdout_file (f, "user(%s): %f" % (similar_users[i][0], similar_users[i][1]))
            if i > 9:
                break
        output_stdout_file (f)
        """

        for i in range(3):
            attr_key = attr_keys[i]
            min_attr_dist_distance_user[attr_key].append(min_attr_user[attr_key])

        min_product_dist_distance_user.append(min_product_user)

        min_all_attr_dist_distance_user.append(min_all_attr_user)

        est_params = copy.deepcopy(min_product_user.params)

        est_params_history.append(est_params)

    output_stdout_file(f_log, "\nactual user(%s )\n" %(actual_user.id))

    """ output similar users
    similar_users = sorted(product_choice_prob_distance_map.items(), key=lambda x: x[1], reverse=False)
    output_stdout_file (f, "similar_users:")
    for i in range(len(similar_users)):
        output_stdout_file (f, "user(%s): distance=%f" % (similar_users[i][0], similar_users[i][1]))
        similar_user = dummy_user_map[similar_users[i][0]]
        similar_user.params.output_to_file(f, True)

        if i > 9:
            break
    output_stdout_file (f)
    """

    """
    output learnt parameters
    """
    learnt_params = est_params_history[len(est_params_history) -1]
                                       
    output_stdout_file(f_log, "\nlearnt parameters:")
    learnt_params.output(f_log, True)

    """
    output learnt parameters and errors
    """
    output_dir = config.output_dir + "/err1"
    output_est_error(output_dir, actual_user, est_params_history, config.dum.beta_range)

    output_dir = config.output_dir + "/err_all_attr"
    output_est_error2(output_dir, actual_user, min_all_attr_dist_distance_user, config.dum.beta_range)

    output_dir = config.output_dir + "/err_dt"
    output_est_error2(output_dir, actual_user, min_attr_dist_distance_user['DT'], config.dum.beta_range)

    output_dir = config.output_dir + "/err_cg"
    output_est_error2(output_dir, actual_user, min_attr_dist_distance_user['CG'], config.dum.beta_range)

    output_dir = config.output_dir + "/err_ic"
    output_est_error2(output_dir, actual_user, min_attr_dist_distance_user['IC'], config.dum.beta_range)

    """
    output skew
    """
    output_dir = config.output_dir + "/skew"

    if(os.path.exists(output_dir) == False):
        os.mkdir(output_dir)

    filepath = output_dir + "/skew.csv"
    skew_history.to_csv(filepath)

    df_du_skew = pd.DataFrame(index=[], columns=['user_id', 'DT', 'CG', 'IC'])
    for id, dummy_user in dummy_user_map.items():
        skew = dummy_user.choice_history.getSkew()
        series = pd.Series([id, skew['DT'], skew['CG'], skew['IC']], index=df_du_skew.columns)
        df_du_skew = df_du_skew.append(series, ignore_index=True)

    filepath = output_dir + "/du_skew.csv"
    df_du_skew.to_csv(filepath)

    """
    output attribute choice distribution
    """
    output_dir = config.output_dir + "/choice"
    actual_user.choice_history.outputAttrChoiceProbDistribution(output_dir)


def validate4(actual_user, n_epoch, rand_assort):
    """
    validation
    actual_user: actual user
    n_epoch: epoch
    rand_assort: test with random assortment(true) or one assortment(false)
    """

    global config

    f_log = config.log_file

    for i in range(2):

        output_dir = config.output_dir + "/err1"
        if(i == 1):
            output_dir = config.output_dir + "/err_all_attr"

        filepath = output_dir + "/param_learnt.csv"

        df = pd.read_csv(filepath)
        user_num = len(df.index)
        l = (len(df) - 1)
        est_user = UserAgent("eu", df.ix[l, "beta_cg"], df.ix[l, "beta_fare"], df.ix[l, "beta_ic"],
                      df.ix[l, "beta_dt"], df.ix[l, "beta_ivtt"], df.ix[l, "beta_ovtt"], df.ix[l, "beta_wt"])

        est_user.output(None, True)

        df = pd.DataFrame(index=[], columns=['exact_match', 'exact_rate', 'segment_match', 'segment_rate'])

        if (rand_assort == False):
            assortment = create_assortment(0)

        exact_count = 0
        segment_count = 0
        for it in range(0, n_epoch):

            output_stdout_file(f_log, "\n---------- [validation] actual user(%s), random_assort=%d, iteration %d / %d ----------\n" % (actual_user.id, rand_assort, it, n_epoch))

            if (rand_assort == True):
                assortment = create_random_assortment(it)

            # simulate that an actual user selects product among an assortment
            selected_product_idx1, selected_product_prob1 = actual_user.selectProduct(assortment)
            product1 = assortment.get(selected_product_idx1)

            prob_array, V_array, max_prob_idx = actual_user.getChoiceProbability(assortment)
            actual_user.choice_history.addChoiceProb(assortment, prob_array)
            actual_user.choice_history.updateChoiceProbDistribution()

            # simulate that an estimated user selects product among an assortment
            selected_product_idx2, selected_product_prob2 = est_user.selectProduct(assortment)
            product2 = assortment.get(selected_product_idx2)

            prob_array, V_array, max_prob_idx = est_user.getChoiceProbability(assortment)
            est_user.choice_history.addChoiceProb(assortment, prob_array)
            est_user.choice_history.updateChoiceProbDistribution()

            exact_match = 0
            if(selected_product_idx1 == selected_product_idx2):
                exact_match = 1
                exact_count += 1
            exact_rate = exact_count / (it + 1)

            dt1, cg1, ic1 = ChoiceDistribution.getSegment(product1)
            dt2, cg2, ic2 = ChoiceDistribution.getSegment(product2)
            segment_match = 0
            if((dt1 == dt2) and (cg1 == cg2) and (ic1 == ic2)):
                segment_match = 1
                segment_count += 1
            segment_rate = segment_count / (it + 1)

            series = pd.Series([exact_match, exact_rate, segment_match, segment_rate], index=df.columns)
            df = df.append(series, ignore_index=True)

        filepath = output_dir + "/param_valid.csv"
        df.to_csv(filepath)


def algo(algo_type, rand_user, user_i, n_epoch):
    """
    algorithm to learn behavior parameters
    user_i: i th test
    n_epoch: epoch
    """

    config.initialize(algo_type, user_i)

    dummy_user_map = config.dum.user_map

    actual_user = None
    if(rand_user):
        r = random.randint(0, len(list(dummy_user_map.keys())) -1)
        actual_dummy_user = dummy_user_map[list(dummy_user_map.keys())[r]]
    #    actual_user = UserAgent(1, -0.525, -0.35, 0.35, -0.28, -0.175, -0.4375, -0.28)
        id = "au[copy of %s]" % actual_dummy_user.id
    #    actual_user = UserAgent(id, dummy_user.params.beta_cg, dummy_user.params.beta_fare, dummy_user.params.beta_ic, dummy_user.params.beta_dt, dummy_user.params.beta_ivtt, dummy_user.params.beta_ovtt, dummy_user.params.beta_wt)
        actual_user = copy.deepcopy(actual_dummy_user)
        actual_user.id = id
    else:
        actual_user = config.au_list[user_i]

    actual_user.output(None, True)

    file_name = config.user_dir + "/actual_user.csv"
    f = open(file_name, 'w')
    actual_user.to_csv(f)
    f.write("\n")
    f.close()

    for i in range(2):
        if(i == 0):
            rand_assort = False
            continue ################################# test
        else:
            rand_assort = True
            
        config.output_dir = config.user_dir
        if(rand_assort == True):
            config.output_dir += "/random"
        else:
            config.output_dir += "/fixed"
        
        if(os.path.exists(config.output_dir) == False):
            os.mkdir(config.output_dir)
    
        log_dir = config.output_dir + "/log"
        if(os.path.exists(log_dir) == False):
            os.mkdir(log_dir)
    
        file_name = log_dir + "/log.txt"
        config.log_file = open(file_name, 'w')
        
        file_name = config.output_dir + "/assortment.csv"
        config.assortment_file = open(file_name, 'w')
        csvWriter = csv.writer(config.assortment_file, lineterminator='\n')

        header = ['assortment', 'seq', 'product']
        for key in Product.attr_keys:
            header.append(key)
        csvWriter.writerow(header)

        if(algo_type == 2):
            algo2(actual_user, n_epoch, rand_assort)
        elif(algo_type == 4):
            algo4(actual_user, n_epoch, rand_assort)
            validate4(actual_user, 100, rand_assort)
        else:
            print("[ERROR]: algo_type %d is not supported" % algo_type)

        config.log_file.close()
        config.assortment_file.close()


    config.uninitialize()


def main():
    
    global config
    
    config = Config()

    args = sys.argv
    algo_type = 1
    user_num = 1
    n_epoch = 10
#    print args
#    print len(args)
    
    # algorithm type
    if len(args) > 1:
        algo_type = int(args[1])
        print("algorithm: %d" % algo_type)
    else:
        print("[Usage]: %s (algo_type[2,4])(no of users) (no of epoch)" % args[0])
        return        

    if(algo_type > 4):
        print("[ERROR]: unknown algo type %d" % algo_type)
        return

    rand_user = True
    # test num with diffierent actual users
    if len(args) > 2:
        user_num = int(args[2])
        if(user_num < 0):
            print("specify no of users")
            return
        print("test %d user(s)" % user_num)

        if(user_num == 0):
            df = pd.read_csv('actual_users.csv', index_col='id' )
            rand_user = False
#            print(df)       # show all column
#            print(df.shape)
#            print(df.ix[1 , :])
            user_num = len(df.index)
            print(user_num)
            for i in range(user_num):
                config.au_list.append(UserAgent(df.index[i], df.ix[i,"beta_cg"], df.ix[i,"beta_fare"], df.ix[i,"beta_ic"], df.ix[i,"beta_dt"], df.ix[i,"beta_ivtt"], df.ix[i,"beta_ovtt"], df.ix[i,"beta_wt"]))

            df_new = pd.DataFrame(index=df.index, columns=["", 2])

    # no of epoch in each test
    if len(args) > 3:
        n_epoch = int(args[3])
        if(n_epoch < 1):
            print("specify no of epoch")
            return
        print("no of epoch: %d" % n_epoch)

    for l in range(5):
        if(algo_type != 0):
            if(algo_type != (l + 1)):
                continue

        for i in range(0, user_num):
            algo((l + 1), rand_user, i, n_epoch)

    print ("finished!")


if __name__ == "__main__":
    
    main()
