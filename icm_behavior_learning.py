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
import behavior_learning

from user_agent import UserAgent
from user_agent import BehaviorModelParam
from user_agent import DummyUserManager
from choice_history import ChoiceDistribution

from product import Assortment
from product import Product
from behavior_learning import config

def output(f, log="", stdout=True):
    """
    output to fine and stdout
    """
    if ((f is None) == False):
        f.write(log + "\n")
        f.flush()

    if (stdout):
        sys.stdout.write(log + "\n")
        sys.stdout.flush()


def icm_behavior_learning(user_id):
    """
    algorithm to learn behavior parameters(based on the similarity of choice distribution)
    user_id: actual user
    """

    # create reference models(dummy users)
    dum = DummyUserManager()
    dum.generate_dummy_users3()
    dummy_user_map = dum.user_map

    output_dir = "./icm"
    if (os.path.exists(output_dir) == False):
        os.mkdir(output_dir)

    file_name = output_dir + "/log.txt"
    #f_log = open(file_name, 'w')
    f_log = None
    est_params_history = []
    min_product_dist_distance_user = []

    ave_params = dum.get_average_params()
    est_params_history.append(ave_params)

    ############################################################################

    """
    ToDo: load assortment and choice history of user_id from database
    length of assortment_list and selected_product_id_list must be same
    """
    conn = None
    parameters = config()
    try:
        conn = psycopg2.connect(**parameters)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        print 'Connected!\n'
    except psycopg2.DatabaseError:
        print 'I am unable to connect the database:'
        sys.exit(1)

    sql = """SELECT T1.*,T2.cid,coalesce(T2.cv,0) FROM 
        (SELECT  a.user_id,a.id as assort_id,a.selected_product,b.id as pid,b.seq,b.egress_mode,b.cost,b.dwell_duration,
        b.travel_duration,b.access_duration,b.waiting_duration,b.congestion from assortment a, 
        product b WHERE a.user_id= %s and a.id=b.assortment_id
        and a.selected_product is not null
        ) T1 LEFT JOIN 
        (SELECT c.product_id as cid, avg(d.coupon_value) as cv from product_coupon c, coupon d where c.coupon_id=d.id
        group by cid) T2 on T1.pid = T2.cid 
        order by assort_id """ % user_id
    try:
        #print 'SQL: %s' % (sql)
        cursor.execute(sql)
    except psycopg2.DatabaseError:
        print 'unable to retrieve the records'
        sys.exit(1)

    assortmentdata = cursor.fetchall()
    cursor.close()

    assortment_list = []
    selected_product_id_list = []
    idx = 0
    for i in range(0, len(assortmentdata)):
        product = Product(assortmentdata[i][3], assortmentdata[i][4], assortmentdata[i][5],
                          float(assortmentdata[i][6]), assortmentdata[i][7], assortmentdata[i][8],
                          assortmentdata[i][9], assortmentdata[i][10], assortmentdata[i][11],
                          float(assortmentdata[i][13]))
        if i == 0:
            idx=0
            assortment = Assortment(assortmentdata[i][1]);
            assortment.add(product)
            assortment_list.append(assortment)
        else:
            if assortmentdata[i][1] == assortment.id :
                assortment.add(product)
            else:
                idx=0
                assortment = Assortment(assortmentdata[i][1]);
                assortment.add(product)
                assortment_list.append(assortment)

        if product.id == assortmentdata[i][2]:
            selected_product_id_list.append(idx)
        idx = idx + 1

    """print("assortment check start")
    for i in range(0,len(assortment_list)):
        assortment = assortment_list[i]
        print("------")
        assortment.output(f_log, True)
        print(selected_product_id_list[i])
        product = assortment.get(selected_product_id_list[i])
        product.output(f_log, True)
        print("------")
    #print("assortment check end")"""

    """
    This part will be removed by replacing with above part
    """
    """
    for i in range(0, 5):
        assortment = behavior_learning.create_random_assortment(i)
        assortment_list.append(assortment)
        n = len(assortment.list)
        id = random.randint(0, n -1)
        selected_product_id_list.append(id)
    """
    ############################################################################


    n_epoch = len(assortment_list)
    choice_history = ChoiceDistribution()

    product_choice_prob_distance_map = {}

    if n_epoch > 0 :

        for it in range(0, n_epoch):

            output(f_log,
                               "\n---------- actual user(%s), iteration %d / %d ----------\n" % (user_id, it, n_epoch))

            assortment = assortment_list[it]

            # simulate that an actual user selects product among an assortment
            selected_product_idx = selected_product_id_list[it]

            if selected_product_idx >= 0:
                output(f_log, "<<<<<<<< selected >>>>>>>>")
                product = assortment.get(selected_product_idx)
                product.output(f_log, True)
                output(f_log, "")
                choice_history.addChoice(product)
                choice_history.updateChoiceProbDistribution()
            else:
                output(f_log, "<<<<<<<< [rejected] >>>>>>>>")
                continue

            min_product_d = float("inf")
            min_product_user = None

            # for each dummy user
            for id, dummy_user in dummy_user_map.items():
                #            print("dummy_user(%s)" % dummy_user.id)
                # calculate dummy user's choice probability of each product in an assortment
                prob_array, V_array, max_prob_idx = dummy_user.getChoiceProbability(assortment)

                dummy_user.choice_history.addChoiceProb(assortment, prob_array)
                dummy_user.choice_history.updateChoiceProbDistribution()

                d = dummy_user.choice_history.getProductDistDistance(choice_history)

                product_choice_prob_distance_map[id] = d

                if (d < min_product_d):
                    min_product_user = dummy_user
                    min_product_d = d

            if (min_product_user is None):
                print("no dummy user with minimum distance")
                continue

            # find the dummy user with most similar choice history

            choice_history.output_to_file(f_log, True)

            min_product_dist_distance_user.append(min_product_user)

            est_params = copy.deepcopy(min_product_user.params)

            est_params_history.append(est_params)

        """
        output learnt parameters
        """
        learnt_params = est_params_history[len(est_params_history) - 1]

        output(f_log, "\nlearnt parameters:")
        learnt_params.output(f_log, True)


        ############################################################################

        """
        ToDo: store learnt parameters into database
        """

        update_user = """UPDATE user_profile
                                    SET beta_cg = %s,
                                    beta_i = %s,
                                    beta_dt = %s,
                                    beta_ivtt = %s,
                                    beta_ovtt = %s,
                                    beta_wt = %s
                                    WHERE user_id = %s """
        try:

            cursor = conn.cursor()
            cursor.execute(update_user, (
            learnt_params.get("beta_cg"), learnt_params.get("beta_ic"), learnt_params.get("beta_dt"),
            learnt_params.get("beta_ivtt"), learnt_params.get("beta_ovtt"),
            learnt_params.get("beta_wt"), user_id))
            updated_rows = cursor.rowcount
            #print(updated_rows)
            conn.commit()
            cursor.close()

        except psycopg2.DatabaseError:
            print "unable to update the coefficients of user!"

        ############################################################################

    #f_log.close()

    print("finished!")


if __name__ == "__main__":

    args = sys.argv

    if len(args) != 2:

        print("arg error")

    icm_behavior_learning(args[1])
