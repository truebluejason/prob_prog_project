import torch
from primitives import env as penv
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from pyrsistent import pmap, plist, PMap
import time
from utils import *
import sys
sys.setrecursionlimit(5000)


def standard_env():
    "An environment with some Scheme standard procedures."
    env = Env(penv.keys(), penv.values()) # pmap(penv)
    return env

class Env:

    def __init__(self, params, args, outer=None):
        "An environment: a dict of {'var': val} pairs with an outer Env"
        self.dict = pmap({k:v for k,v in zip(params,args)})
        self.dict = self.dict.update({'alpha':''})
        self.outer = outer

    def contains(self, var):
        "Check if key 'var' exists at any level"
        if var in self.dict:
            return True
        elif self.outer is None:
            return False
        else:
            return self.outer.contains(var)

    def find(self, var):
        "Find the innermost Env where var appears"
        return self.dict[var] if (var in self.dict) else self.outer.find(var)

class Procedure(object):

    def __init__(self, params, body, env):
        "A user defined Scheme procedure"
        self.params, self.body, self.env = params, body, env

    def __call__(self, *args, sigma=None):
        return eval(self.body, sigma=sigma, env=Env(params=self.params, args=args, outer=self.env))


def eval(expr, sigma, env):
    if is_const(expr, env):
        if type(expr) in [int, float, bool]:
            expr = torch.Tensor([expr]).squeeze()
        return expr, sigma
    elif is_var(expr, env):
        return env.find(expr), sigma
    elif is_if(expr, env):
        cond_expr, true_expr, false_expr = expr[1], expr[2], expr[3]
        cond_value, sigma = eval(cond_expr, sigma, env)
        if cond_value:
            return eval(true_expr, sigma, env)
        else:
            return eval(false_expr, sigma, env)
    elif is_fn(expr, env):
        params, body = expr[1], expr[2]
        return Procedure(params, body, env), sigma
    elif is_sample(expr, env):
        addr_expr, dist_expr = expr[1], expr[2]
        addr, sigma          = eval(addr_expr, sigma, env)
        dist_obj, sigma      = eval(dist_expr, sigma, env)
        # return sample from the distribution object
        return dist_obj.sample(), sigma
    elif is_observe(expr, env):
        addr_expr, dist_expr, obs_expr = expr[1], expr[2], expr[3]
        addr, sigma                    = eval(addr_expr, sigma, env)
        dist_obj, sigma                = eval(dist_expr, sigma, env)
        obs_value, sigma               = eval(obs_expr, sigma, env)
        return obs_value, sigma
    else:
        proc_name  = expr[0]
        parameters = []
        # retrieve function object by name from the environment
        proc_obj, sigma = eval(proc_name, sigma, env)
        # evaluate parameter expressions
        for i in range(1,len(expr)):
            param, sigma = eval(expr[i], sigma, env)
            parameters.append(param)
        # if procedure is user-defined, return result and updated sigma. else, return result and same sigma.
        if type(proc_obj) == Procedure:
            proc_result, sigma = proc_obj(*parameters, sigma=sigma)
            return proc_result, sigma
        else:
            proc_result = proc_obj(*parameters)
            return proc_result, sigma

def evaluate(expr):
    env   = standard_env()
    sigma = pmap()
    proc_obj, sigma = eval(expr, sigma, env)
    return proc_obj(sigma=sigma)

def is_const(expr, env):
    return type(expr) not in [tuple, list, dict] and not env.contains(expr)

def is_var(expr, env):
    return type(expr) not in [tuple, list, dict] and env.contains(expr)

def is_if(expr, env):
    return expr[0] == "if"

def is_fn(expr, env):
    return expr[0] == "fn"

def is_sample(expr, scope):
    return expr[0] == "sample"

def is_observe(expr, scope):
    return expr[0] == "observe"

def get_stream(exp):
    while True:
        yield evaluate(exp)


def sample_prior(ast, num_samples=10):
    traces, weights = [], []
    stream = get_stream(ast)
    for i in range(num_samples):
        sample, sigma = next(stream)
        traces.append(sample)
        weights.append(0.)
    return traces, weights


def run_deterministic_tests():

    for i in range(1,14):
        exp = daphne(['desugar-hoppl', '-i', '../hoppl/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, _ = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        print(f'FOPPL test {i} passed')

    for i in range(1,13):
        exp = daphne(['desugar-hoppl', '-i', '../hoppl/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret, _ = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        print(f'HOPPL test {i} passed')     
    print('All deterministic tests passed')


def run_probabilistic_tests():

    num_samples=1e4
    max_p_value = 1e-2

    for i in range(1,7):
        exp = daphne(['desugar-hoppl', '-i', '../hoppl/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))

        stream = get_stream(exp)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert(p_val > max_p_value)

    print('All probabilistic tests passed')    



if __name__ == '__main__':

    #run_deterministic_tests()
    #run_probabilistic_tests()

    for i in range(1,4):
        ast = daphne(['desugar-hoppl', '-i', '../hoppl/programs/{}.daphne'.format(i)])
        start_time = time.time()
        traces, weights = sample_prior(ast, num_samples=5000)
        end_time   = time.time()
        print(f"==============================")
        print(f"PROGRAM {i} RUNTIME: {end_time-start_time}")
        print(f"==============================")
        means = compute_mean(traces, weights)
        for j, mean in enumerate(means):
            print(f"Posterior mean at site {j}: {mean}")
        variances = compute_variance(traces, weights)
        for j, variance in enumerate(variances): 
            print(f"Posterior variance at site {j}: {variance}")
        if i == 3:
            plot_posterior_jointly(f'p{i}prior', traces, 6, 3, weights)
        else:
            plot_posterior(f'p{i}prior', traces, weights)
