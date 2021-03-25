from evaluator import evaluate
import torch
import numpy as np
import json
import sys
import time
from utils import *


def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res

def resample_particles(particles, log_weights):
    resample_probs  = torch.Tensor(log_weights).exp()
    resample_probs  = resample_probs+1e-5 / sum(resample_probs+1e-5) # for numerical stability
    logZ            = torch.mean(resample_probs).log()
    num_samples     = len(particles)
    sample_idxs     = torch.multinomial(resample_probs, num_samples=num_samples, replacement=True)
    new_particles   = []
    for idx in sample_idxs:
        new_particles.append(particles[idx]) # NOTE: Are we handling immutability correctly?
    return logZ, new_particles



def SMC(n_particles, exp):

    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):

        res = evaluate(exp, env=None)('addr_start', output)
        logW = 0.


        particles.append(res)
        weights.append(logW)

    #can't be done after the first step, under the address transform, so this should be fine:
    done = False
    smc_cnter = 0
    while not done:
        # print('In SMC step {}, Zs: '.format(smc_cnter), logZs)
        curr_addr = None
        for i in range(n_particles): #Even though this can be parallelized, we run it serially
            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res
                if i == 0:
                    done = True  #and enforces everything to be the same as the first particle
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                #TODO: check particle addresses, and get weights and continuations
                particles[i] = res
                if i == 0:
                    curr_addr = res[2]['addr']
                else:
                    addr = res[2]['addr']
                    if not addr == curr_addr:
                        raise RuntimeError(f'Failed SMC: Expected to see address: {curr_addr} but saw: {addr}')
                logW = res[2]['log_prob']
                weights[i] += logW

        if not done:
            #resample and keep track of logZs
            logZn, particles = resample_particles(particles, weights)
            weights = [0.] * len(weights)
            logZs.append(logZn)
        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles


if __name__ == '__main__':

    for i in range(1,5):
        with open('programs/{}.json'.format(i),'r') as f:
            exp = json.load(f)
        n_particles = [1, 10, 100, 1000, 10000, 100000]
        #n_particles = [10000]
        for n_particle in n_particles:
            start_time = time.time()
            logZ, particles = SMC(n_particle, exp)
            end_time   = time.time()
            print(f"==============================")
            print(f"PROGRAM {i} #PARTICLES {n_particle} RUNTIME: {round(end_time-start_time,0)}")
            print(f"==============================")
            print('logZ: ', logZ)
            samples = [p[0] for p in particles]
            means = compute_mean(samples)
            for j, mean in enumerate(means):
                print(f"Posterior mean at site {j}: {mean}")
            if i == 3:
                plot_posterior_jointly(f'p{i}_n{n_particle}', samples, 6, 3)
            else:
                plot_posterior(f'p{i}_{n_particle}', samples)

            # values = torch.stack(samples).type(torch.float32)
            #TODO: some presentation of the results
            # print(values)
