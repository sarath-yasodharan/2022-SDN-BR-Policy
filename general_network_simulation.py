import pandas as pd
import numpy as np
from general_network_modules import *

# Provide the network parameters here

capacities = list((10*np.ones(18)).astype(int))
arrival_rates = [1,2,1,3,1,2] 
revenue_rates = [8,2,3,2,10,1] 
bandwidths = [1,1,1,1,1,1]
A = np.array([[ 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],\
                     [ 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],\
                     [ 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],\
                     [ 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],\
                     [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],\
                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]])
    


def create_link_by_link_policy(T, A, state, capacities):
    policy = []
    num_requests = A.shape[0]
    num_resources = A.shape[1]
    current_numbers = np.array(state.split(':')).astype(int)
    for i in range(num_requests):
        temp = []
        link_indices = np.where(A[i]>0)[0]
        for j in link_indices:
            if capacities[j] - np.matmul(current_numbers, A)[j] >=T[i][j]:
                temp = temp + [1]
            else:
                temp = temp + [0]
        if np.prod(temp) == 1:
            policy = policy + [1]
        else:
            policy = policy + [0]
    return policy

def simulate_general_network(A, T, policy_type):
    steps = 2000000
    num_requests = A.shape[0]
    current_state = [':'.join(items) for items in itertools.product([str(0)], repeat=num_requests)][0]
    total_revenue = 0
    total_time = 0
    for i in range(steps):
        if policy_type == 'accept_all':
            policy = create_default_policy(A, current_state, capacities)
        elif policy_type == 'link_by_link':
            policy = create_link_by_link_policy(T, A, current_state, capacities)
        next_states, next_rates = compute_next_states(capacities, bandwidths, policy, arrival_rates, current_state)
        random_times = [np.random.exponential(1/next_rates[j]) for j in range(len(next_rates))]
        total_revenue += compute_revenue_rate(revenue_rates, current_state)*min(random_times)
        total_time += min(random_times)
        current_state = next_states[np.argmin(random_times)]
    
    average_revenue = total_revenue/total_time
    
    return average_revenue 
    
results = pd.DataFrame(columns =['capacities', 'A_matrix', 'arrival_rates', 'revenues', 'T_matrix', 'B_matrix', 'link_by_link_revenue', 'accept_all_revenue'])


T = np.copy(A)
accept_all_revenue = simulate_general_network(A, T, 'accept_all')
print('accept all revenue:', accept_all_revenue)
T, B = compute_link_by_link_heuristic(A, capacities, revenue_rates, arrival_rates, bandwidths)
link_by_link_revenue = simulate_general_network(A, T, 'link_by_link')

print('link by link revenue:', link_by_link_revenue)

new_row = {'capacities': capacities,  'A_matrix': A, 'arrival_rates': arrival_rates,  'revenues': revenue_rates, 'T_matrix': T, 'B_matrix': B, 'link_by_link_revenue':link_by_link_revenue, 'accept_all_revenue': accept_all_revenue}
results = results.append(new_row, ignore_index=True)
results.to_csv('results.csv', index=False)

