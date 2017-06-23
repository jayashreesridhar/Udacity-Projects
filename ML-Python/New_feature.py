# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:07:47 2017

@author: JAYASHREE
"""

from __future__ import division
import numpy as np



def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """


    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0.
    if poi_messages =='NaN' or all_messages=='NaN':
        fraction=0
    else:
        fraction=poi_messages/all_messages
    

   
    return fraction


#####################

def submitDict():
    return submit_dict
    
def combine_feature(salary,bonus):
    fraction = 0.
    if salary =='NaN' or bonus=='NaN':
        fraction=0
    else:
        fraction=(salary+bonus)/2        
    
    return fraction
    
