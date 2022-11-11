import numpy as np
import pandas as pd
from scipy.linalg import solve, eig
from scipy.optimize import minimize
import time
import itertools
from compute_optimal_threshold_single_link_general_network import *


def create_states(A, capacities):
    num_resources = A.shape[1] 
    num_requests = A.shape[0]
    one_request_possibilities = [str(i) for i in range(1+int(np.max(capacities)/np.min(A[A>0])))]
    
    possible_states = [':'.join(items) for items in itertools.product(one_request_possibilities, repeat=num_requests)]
    
    states = []
    for i in range(len(possible_states)):
        if min((np.matmul(np.array(possible_states[i].split(':')).astype(int),A) <=capacities).astype(int))==1:
            states = states + [possible_states[i]]
    
    return states

def compute_next_states(capacities, bandwidths, decisions, arrival_rates, state):
    temp = []
    rates = []
    
    current_numbers = np.array(state.split(':')).astype(int)

    for i in range(len(bandwidths)):
        if current_numbers[i] > 0:
            temp1 = np.copy(current_numbers)
            temp1[i] = temp1[i]-1
            temp = temp + [':'.join(temp1.astype(str))]
            rates = rates + [current_numbers[i]]
   
    for i in range(len(bandwidths)):
        if decisions[i] > 0 and arrival_rates[i] > 0:
            temp1 = np.copy(current_numbers)
            temp1[i] = temp1[i]+1
            temp = temp + [':'.join(temp1.astype(str))]
            rates = rates + [arrival_rates[i]]
        
    return temp, rates


def compute_state_index(states, state):
    return np.where(np.array(states) == state)[0][0]

def compute_revenue_rate(revenues, state):
    current_numbers = np.array(state.split(':')).astype(int)
    return np.dot(current_numbers, revenues)

def create_default_policy(A, state, capacities):
    policy = []
    num_requests = A.shape[0]
    current_numbers = np.array(state.split(':')).astype(int)
    for i in range(num_requests):
        temp = np.copy(current_numbers)
        temp[i] = temp[i] + 1
        if min((np.matmul(temp,A) <=capacities).astype(int))==1:
            policy = policy + [1]
        else:
            policy = policy + [0]
    return policy

def compute_link_by_link_heuristic(A, capacities, revenue_rates, arrival_rates, bandwidths):
    num_requests = A.shape[0]
    num_resources = A.shape[1]
    
    B = np.zeros(((num_requests),(num_resources)))
    T = np.zeros(((num_requests),(num_resources)))
    R = np.zeros((num_resources))
    
    B_temp = np.copy(B)
    T_temp = np.copy(T)
    
    revenue_old = 0
    revenue_diff = 1
    count = 1
    while revenue_diff >0.01 and count<=5:
        print('iteration count for link by link computation', count)
        count = count + 1
        for j in range(num_resources):
            request_indices = (np.where(A[:,j] > 0)[0]).astype(int)
            capacity_link = capacities[j]
            bandwidths_link = np.take(bandwidths, request_indices)
            
            blocking_probabilities_multiplier = np.zeros(len(request_indices))
            for i in range(len(request_indices)):
                others_array = np.setdiff1d(1-B[request_indices[i]][B[request_indices[i]]>0], 1-B[request_indices[i],j])
                if len(others_array)==0:
                    blocking_probabilities_multiplier[i] = 1
                else:
                    blocking_probabilities_multiplier[i] = np.prod(others_array)
                    
            arrival_rates_link = np.maximum(blocking_probabilities_multiplier, 0)*np.take(arrival_rates, request_indices)
            revenue_rates_link = np.take(revenue_rates, request_indices)/(np.array([len(np.where(A[i]>0)[0]) for i in request_indices]))
            
            print('arrival rates:', arrival_rates_link)
            print('revenue rates:', revenue_rates_link)
            thresholds_link, blockings_link, revenue_link = compute_optimal_threshold_blocking_single_link(capacity_link, bandwidths_link, arrival_rates_link, revenue_rates_link)
            print('thresholds:', thresholds_link)
            print('blockings:', blockings_link)
            for i in range(len(request_indices)):
                if arrival_rates_link[i]<=1e-6:
                    B_temp[request_indices[i],j] = 1
                    T_temp[request_indices[i],j] = capacity_link+1
                else:
                    B_temp[request_indices[i],j] = blockings_link[i]
                    T_temp[request_indices[i],j] = np.array(thresholds_link.split(':')).astype(int)[i]
            R[j] = revenue_link
            print('temporary T:', T_temp)
            print('temporary B:', B_temp)
        revenue_diff = np.abs(np.sum(R) - revenue_old)
        revenue_old = np.sum(R)
        T = np.copy(T_temp)
        B = np.copy(B_temp)
        print('blocking probability matrix:', B)
    return T, B
