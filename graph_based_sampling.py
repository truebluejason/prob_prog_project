import time
import torch
import torch.distributions as distributions
from torch.distributions import Uniform, Normal
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

from daphne import daphne
import numpy as np

from primitives import PRIMITIVES
from tests import is_tol, run_prob_test,load_truth
from utils import *


def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return PRIMITIVES[op](*map(deterministic_eval, args))
    elif type(exp) in [int, float, bool]:
        # We use torch for all numerical objects in our evaluator
        return torch.Tensor([float(exp)]).squeeze()
    elif type(exp) is torch.Tensor:
        return exp
    elif exp == None:
        return None
    else:
        raise Exception("Expression type unknown.", exp)

def topological_sort(nodes, edges):
    result = []
    visited = {}
    def helper(node):
        if node not in visited:
            visited[node] = True
            if node in edges:
                for child in edges[node]:
                    helper(child)
            result.append(node)
    for node in nodes:
        helper(node)
    return result[::-1]

def plugin_parent_values(expr, trace):
    if type(expr) == str and expr in trace:
        return trace[expr]
    elif type(expr) == list:
        return [plugin_parent_values(child_expr, trace) for child_expr in expr]
    else:
        return expr

def compute_log_joint(sorted_nodes, links, trace_dict):
    joint_log_prob = 0.
    for node in sorted_nodes:
        link_expr = links[node][1]
        dist      = deterministic_eval(plugin_parent_values(link_expr, trace_dict))
        joint_log_prob += dist.log_prob(trace_dict[node])
    return joint_log_prob

def sample_from_joint(graph):
    """
    This function does ancestral sampling starting from the prior.
    1. Run topological sort on V using V and A, resulting in an array of v's
    2. Iterate through sample sites of the sorted array, and save sampled results on trace dictionary using P and Y
    - If keyword is sample*, first recursively replace sample site names with trace values in the expression from P. Then, run deterministic_eval.
    - If keyword is observe*, put the observation value in the trace dictionary
    3. Filter the trace dictionary for things sample sites you should return
    """
    procs, model, expr = graph[0], graph[1], graph[2]
    nodes, edges, links, obs = model['V'], model['A'], model['P'], model['Y']
    sorted_nodes = topological_sort(nodes, edges)

    sigma = {}
    trace = {}
    for node in sorted_nodes:
        keyword = links[node][0]
        if keyword == "sample*":
            link_expr = links[node][1]
            link_expr = plugin_parent_values(link_expr, trace)
            dist_obj  = deterministic_eval(link_expr)
            trace[node] = dist_obj.sample()
        elif keyword == "observe*":
            trace[node] = obs[node]

    expr = plugin_parent_values(expr, trace)
    return deterministic_eval(expr), sigma, trace


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)


def MH_within_Gibbs(graph, num_samples=10):
    procs, model, expr = graph[0], graph[1], graph[2]
    nodes, edges, links, obs = model['V'], model['A'], model['P'], model['Y']
    sorted_nodes = topological_sort(nodes, edges)
    sorted_unobserved_nodes = list(filter(lambda node: links[node][0] == "sample*", sorted_nodes))

    def accept(node, trace_dict, trace_dict_new, proposal_dict):
        # Compute acceptance probability using: p(x)q(x'|x) / p(x)q(x|x')
        link_expr         = proposal_dict[node][1]
        proposal_dist     = deterministic_eval(plugin_parent_values(link_expr, trace_dict))
        proposal_dist_new = deterministic_eval(plugin_parent_values(link_expr, trace_dict_new))
        log_alpha = proposal_dist_new.log_prob(trace_dict[node]) - proposal_dist.log_prob(trace_dict_new[node])
        # for simplicity just take all nodes after 'node' in topological order
        free_vars = sorted_nodes[sorted_nodes.index(node):]
        for var in free_vars:
            link_expr    = links[var][1]
            var_dist     = deterministic_eval(plugin_parent_values(link_expr, trace_dict))
            var_dist_new = deterministic_eval(plugin_parent_values(link_expr, trace_dict_new))
            log_alpha    = log_alpha + var_dist_new.log_prob(trace_dict_new[var])
            log_alpha    = log_alpha - var_dist.log_prob(trace_dict[var])
        return torch.exp(log_alpha)

    def gibbs_step(trace_dict, proposal_dict):
        # Given an existing trace dictionary, for each x_i in trace_dict in topological order:
        # 1. Sample x_i' from a proposal distribution given previous x_i
        # 2. Initialize trace_dict_new as trace_dict with x_i set to x_i'
        # 3. Compute acceptance probability and replace trace_dict with trace_dict_new if accepted
        for node in sorted_unobserved_nodes:
            val                  = trace_dict[node]
            link_expr            = proposal_dict[node][1]
            val_new              = deterministic_eval(plugin_parent_values(link_expr, trace_dict)).sample()
            trace_dict_new       = {**trace_dict}
            trace_dict_new[node] = val_new
            accept_threshold     = accept(node, trace_dict, trace_dict_new, proposal_dict)
            uniform_sample       = Uniform(0.,1.).sample()
            if uniform_sample < accept_threshold:
                trace_dict = trace_dict_new
        return trace_dict

    # Initialize using trace variables sampled from the prior
    # Use prior for proposal
    proposal_dict = links
    trace_dicts = [sample_from_joint(graph)[2]]
    for i in range(1,num_samples):
        trace_dict = gibbs_step({**trace_dicts[i-1]}, proposal_dict)
        trace_dicts.append(trace_dict)

    # Convert each trace_dicts return expression
    traces          = list(map(lambda trace_dict: deterministic_eval(plugin_parent_values(expr, trace_dict)), trace_dicts))
    joint_log_probs = list(map(lambda trace_dict: compute_log_joint(sorted_nodes, links, trace_dict), trace_dicts))
    return traces, joint_log_probs


def HMC(graph, num_samples=10, num_jumps=20, jump_dist=0.1):
    procs, model, expr = graph[0], graph[1], graph[2]
    nodes, edges, links, obs = model['V'], model['A'], model['P'], model['Y']
    sorted_nodes = topological_sort(nodes, edges)
    sorted_unobserved_nodes = list(filter(lambda node: links[node][0] == "sample*", sorted_nodes))

    def hamiltonian(trace_dict, momentum, precision=1.):
        unobserved_trace_list = [trace_dict[node] for node in sorted_unobserved_nodes]
        U = potential(unobserved_trace_list)
        K = - 0.5 * sum([precision * node_momentum * node_momentum for node_momentum in momentum])
        result = U + K
        return result
    
    def potential(unobserved_trace_list):
        E_X, E_Y = 0., 0.
        trace_dict = {**dict(zip(sorted_unobserved_nodes, unobserved_trace_list)), **obs}
        for node in sorted_nodes:
            link_expr = links[node][1]
            node_dist = deterministic_eval(plugin_parent_values(link_expr, trace_dict))
            if node in obs:
                E_Y = E_Y + node_dist.log_prob(trace_dict[node])
            else:
                E_X = E_X + node_dist.log_prob(trace_dict[node])
        result = -1 * (E_X + E_Y)
        return result

    def potential_grad(unobserved_trace_list):
        for value in unobserved_trace_list:
            value.requires_grad = True
        energy = potential(unobserved_trace_list)
        energy.backward()
        result = [value.grad for value in unobserved_trace_list]
        for value in unobserved_trace_list:
            value.requires_grad = False
        return result

    def leapfrog(trace_dict, momentum_new):
        trace_list   = [trace_dict[node] for node in sorted_unobserved_nodes] # Only leapfrog for unobserved nodes
        grad_list    = potential_grad(trace_list)
        momentum_new = [node_momentum - 0.5*jump_dist*node_grad for node_momentum, node_grad in zip(momentum_new, grad_list)]
        for j in range(num_jumps):
            trace_list   = [el + jump_dist*momentum_new[i] for i, el in enumerate(trace_list)]
            grad_list    = potential_grad(trace_list)
            momentum_new = [node_momentum - jump_dist*node_grad for node_momentum, node_grad in zip(momentum_new, grad_list)]
        trace_list     = [el + jump_dist*momentum[i] for i, el in enumerate(trace_list)]
        momentum_new   = [node_momentum - 0.5*jump_dist*node_grad for node_momentum, node_grad in zip(momentum_new, grad_list)]
        trace_dict_new = {**dict(zip(sorted_unobserved_nodes, trace_list)), **obs}
        return trace_dict_new, momentum_new

    # Initialize using trace variables sampled from the prior
    # Use prior for proposal
    proposal_dict = links
    trace_dicts = [sample_from_joint(graph)[2]]
    for i in range(1,num_samples):
        momentum = [] # Just support diagonal covariance for now
        for node in sorted_unobserved_nodes:
            node_shape = trace_dicts[0][node].shape
            node_momentum  = Normal(torch.zeros(node_shape), torch.ones(node_shape)).sample()
            momentum.append(node_momentum)
        trace_dict       = deepcopy(trace_dicts[i-1])
        trace_dict_new, momentum_new = leapfrog(deepcopy(trace_dict), deepcopy(momentum))
        uniform_sample   = Uniform(0.,1.).sample()
        accept_threshold = torch.exp(hamiltonian(trace_dict, momentum) - hamiltonian(trace_dict_new, momentum_new))
        if uniform_sample < accept_threshold:
            trace_dicts.append(trace_dict_new)
        else:
            trace_dicts.append(trace_dict)

    # Convert each trace_dicts return expression
    traces          = list(map(lambda trace_dict: deterministic_eval(plugin_parent_values(expr, trace_dict)), trace_dicts))
    joint_log_probs = list(map(lambda trace_dict: compute_log_joint(sorted_nodes, links, trace_dict), trace_dicts))
    return traces, joint_log_probs


#Testing:

def run_deterministic_tests():
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')



def run_probabilistic_tests():
    
    #TODO: 
    num_samples = 1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        print(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


def print_tensor(tensor):
    tensor = np.round(tensor.numpy(), decimals=3)
    print(tensor)
        

if __name__ == '__main__':
    

    #run_deterministic_tests()
    #run_probabilistic_tests()
    for i in [1,2,3,4,5]:
        graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
        start_time = time.time()
        traces, joint_log_probs = MH_within_Gibbs(graph, num_samples=10000)
        end_time   = time.time()
        print(f"==============================")
        print(f"PROGRAM {i} RUNTIME: {end_time-start_time}")
        print(f"==============================")
        means = compute_mean(traces)
        for j, mean in enumerate(means):
            print(f"Posterior mean at site {j}: {mean}")
        if i == 2: print(f"Posterior covariance: {compute_covariance(traces)}")
        variances = compute_variance(traces)
        for j, variance in enumerate(variances): 
            print(f"Posterior variance at site {j}: {variance}")
        plot_posterior(f'p{i}gibbs', traces)
        plot_sample_trace(f'p{i}gibbs_sample', traces)
        plot_log_joint(f'p{i}gibbs_joint', joint_log_probs)

    for i in [1,2,5]:
        graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
        start_time = time.time()
        traces, joint_log_probs = HMC(graph, num_samples=1000)
        end_time   = time.time()
        print(f"==============================")
        print(f"PROGRAM {i} RUNTIME: {end_time-start_time}")
        print(f"==============================")

        means = compute_mean(traces)
        for j, mean in enumerate(means):
            print(f"Posterior mean at site {j}: {mean}")
        if i == 2: print(f"Posterior covariance: {compute_covariance(traces)}")
        variances = compute_variance(traces)
        for j, variance in enumerate(variances): 
            print(f"Posterior variance at site {j}: {variance}")
        plot_posterior(f'p{i}hmc', traces)
        plot_sample_trace(f'p{i}hmc_sample', traces)
        plot_log_joint(f'p{i}hmc_joint', joint_log_probs)
        # traces, joint_log_probs = MH_within_Gibbs(graph, num_samples=100)

        # samples, n = [], 1000
        # for j in range(n):
        #     sample = sample_from_joint(graph)[0]
        #     samples.append(sample)

        # print(f'\nExpectation of return values for program {i}:')
        # if type(samples[0]) is list:
        #     expectation = [None]*len(samples[0])
        #     for j in range(n):
        #         for k in range(len(expectation)):
        #             if expectation[k] is None:
        #                 expectation[k] = [samples[j][k]]
        #             else:
        #                 expectation[k].append(samples[j][k])
        #     for k in range(len(expectation)):
        #         print_tensor(sum(expectation[k])/n)
        # else:
        #     expectation = sum(samples)/n
        #     print_tensor(expectation)
