import numpy as np
import itertools


def create_states_single_link(capacity, bandwidths):
    states =[]
    num_requests = len(bandwidths)
    
    one_request_possibilities = [str(i) for i in range(1+int(capacity/np.min(bandwidths)))]
    
    possible_states = [':'.join(items) for items in itertools.product(one_request_possibilities, repeat=num_requests)]
    
    states = []
    for i in range(len(possible_states)):
        if np.dot(np.array(possible_states[i].split(':')).astype(int),bandwidths) <= capacity:
            states = states + [possible_states[i]]

    return states

def compute_optimal_threshold_blocking_single_link(capacity, bandwidths, arrival_rates, revenues):
    states = create_states_single_link(capacity, bandwidths)
    all_thresholds = [':'.join(items) for items in  itertools.product([str(i) for i in list(range(1,capacity+2))], repeat = len(bandwidths))][:-1]
    
    temp1 = []
    blocking_probabilities = np.zeros((len(all_thresholds), len(bandwidths)))
    for i in range(len(all_thresholds)):
        thresholds = np.array(all_thresholds[i].split(':')).astype(int)
        if np.min(thresholds[arrival_rates > 0])==1:        
            revenue, blocking_probabilities[i] = compute_revenue_blocking_single_link_simulation(states, capacity, bandwidths, thresholds, arrival_rates, revenues)
            temp1.append(revenue)
        else:
            revenue = 0
            temp1.append(revenue)
    max_temp1 = np.max(temp1)    
    
    return all_thresholds[np.argmax(temp1)], blocking_probabilities[np.argmax(temp1)], max_temp1


def compute_free_capacity_single_link(capacity, bandwidths, state):
    current_numbers = np.array(state.split(':')).astype(int)
    
    return capacity - np.dot(bandwidths, current_numbers)

def compute_state_index_single_link(states, state):
    return np.where(np.array(states) == state)[0][0]

def compute_next_states_single_link(capacity, bandwidths, thresholds, arrival_rates, state):
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
        if compute_free_capacity_single_link(capacity, bandwidths, state) >=thresholds[i] and arrival_rates[i] > 0:
            temp1 = np.copy(current_numbers)
            temp1[i] = temp1[i]+1
            if np.dot(temp1, bandwidths)<=capacity:
                temp = temp + [':'.join(temp1.astype(str))]
                rates = rates + [arrival_rates[i]]
                
    return temp, rates

def compute_revenue_rate_single_link(revenues, state):
    current_numbers = np.array(state.split(':')).astype(int)
    return np.dot(current_numbers, revenues)

def find_blocking_requests(capacity, bandwidths, thresholds, arrival_rates, state):
    temp = np.zeros(len(bandwidths))
    current_numbers = np.array(state.split(':')).astype(int)
    for i in range(len(bandwidths)):
        if compute_free_capacity_single_link(capacity, bandwidths, state) >=thresholds[i]:
            temp1 = np.copy(current_numbers)
            temp1[i] = temp1[i]+1
            if np.dot(temp1, bandwidths)<=capacity:
                temp[i] = 0
            else:
                temp[i] = 1
        else:
            temp[i] = 1
    return temp

    
def compute_revenue_blocking_single_link_simulation(states, capacity, bandwidths, thresholds, arrival_rates, revenues):
    steps = 200000
    current_state = states[0]
    total_revenue = 0
    total_time = 0
    state_counts = np.zeros(len(states))
    total_rates = np.zeros(len(states))
    for i in range(steps):
        current_state_index = compute_state_index_single_link(states, current_state)
        state_counts[current_state_index]+=1
        next_states, next_rates = compute_next_states_single_link(capacity, bandwidths, thresholds, arrival_rates, current_state)
        total_rates[current_state_index] = np.sum(next_rates)
        random_times = [np.random.exponential(1/next_rates[j]) for j in range(len(next_rates))]
        total_revenue += compute_revenue_rate_single_link(revenues, current_state)*min(random_times)
        total_time += min(random_times)
        current_state = next_states[np.argmin(random_times)]
    
    blocking_probabilities = np.zeros(len(bandwidths))
    embedded_chain_prob = state_counts/np.sum(state_counts)

    ctmc_prob = np.zeros(len(states))
    common_factor = embedded_chain_prob/total_rates
    common_factor = np.sum(common_factor[~np.isnan(common_factor)])
    for i in range(len(states)):
        if embedded_chain_prob[i] > 0:
            ctmc_prob[i] = (1/total_rates[i])/((1/embedded_chain_prob[i])*common_factor)
    
    if np.abs(np.sum(ctmc_prob)-1)>0.001:
        ctmc_prob = ctmc_prob/np.sum(ctmc_prob)
        print('results may be incorrect, more simulation needed')
        
    for j in range(len(states)):
        state = states[j]
        temp = find_blocking_requests(capacity, bandwidths, thresholds, arrival_rates, state)
        for i in range(len(bandwidths)):
            if temp[i]==1:
                blocking_probabilities[i]+=ctmc_prob[j]
    
    return total_revenue/total_time, blocking_probabilities
