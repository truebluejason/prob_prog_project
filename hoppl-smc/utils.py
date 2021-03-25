import math
import torch
import matplotlib.pyplot as plt


def compute_mean(traces, weights=None):
    # weights are log weights
    if weights is None:
        weights = [0.]*len(traces)
    weights     = [math.exp(weight) for weight in weights]
    weight_sum  = sum(weights)
    means       = []
    num_latent  = len(traces[0].flatten()) if type(traces[0]) == torch.Tensor else 1
    for i in range(num_latent):
        marginal_samples = []
        for trace, weight in zip(traces, weights):
            marginal_sample = trace.unsqueeze(-1).flatten()[i]
            weighted_sample = marginal_sample * weight/weight_sum
            marginal_samples.append(weighted_sample)
        mean = sum(marginal_samples)
        means.append(mean)
    return means

def compute_variance(traces, weights=None):
    if weights is None:
        weights = [0.]*len(traces)
    means       = compute_mean(traces, weights)
    weights     = [math.exp(weight) for weight in weights]
    weight_sum  = sum(weights)
    variances   = []
    num_latent  = len(traces[0].flatten()) if type(traces[0]) == torch.Tensor else 1
    for i in range(num_latent):
        marginal_diffs = []
        for trace, weight in zip(traces, weights):
            marginal_sample = trace.unsqueeze(-1).flatten()[i]
            weighted_sample = (marginal_sample**2) * weight/weight_sum
            marginal_diffs.append(weighted_sample)
        variance = sum(marginal_diffs) - (means[i]**2)
        variances.append(variance)
    return variances

def compute_covariance(traces, weights=None):
    # Compute covariance between two random variables
    if weights is None:
        weights = [0.]*len(traces)
    means       = compute_mean(traces, weights)
    weights     = [math.exp(weight) for weight in weights]
    weight_sum  = sum(weights)
    xys         = []
    for trace, weight in zip(traces, weights):
        sample1 = trace.unsqueeze(-1).flatten()[0]
        sample2 = trace.unsqueeze(-1).flatten()[1]
        xys.append(weight/weight_sum * sample1 * sample2)
    covariance = sum(xys) - (means[0]*means[1])
    return covariance

def plot_sample_trace(plotname, traces):
    # Possible for 1D or 2D
    dimension = len(traces[0].flatten()) if type(traces[0]) == torch.Tensor else 1
    if dimension == 1:
        plt.plot(traces)
        plt.savefig(f'figures/hw6/{plotname}_0.png')
        plt.clf()
    else:
        d1 = [trace[0] for trace in traces]
        d2 = [trace[1] for trace in traces]
        plt.plot(d1)
        plt.savefig(f'figures/hw6/{plotname}_0.png')
        plt.clf()
        plt.plot(d2)
        plt.savefig(f'figures/hw6/{plotname}_1.png')
        plt.clf()

def plot_losses(plotname, losses):
    plt.plot(losses)
    plt.savefig(f'figures/hw6/{plotname}.png')
    plt.clf()

def plot_posterior(plotname, traces, weights=None):
    # Plot (optionally weighted) histograms of latent variables per dimension
    if weights is None:
        weights = [0.]*len(traces)
    weights     = [math.exp(weight) for weight in weights]
    weights     = torch.Tensor(weights)/sum(weights)
    num_latent  = len(traces[0].flatten()) if type(traces[0]) == torch.Tensor else 1
    for i in range(num_latent):
        marginal_samples = []
        for trace in traces:
            marginal_sample = trace.unsqueeze(-1).flatten()[i]
            marginal_samples.append(marginal_sample)
        plt.hist(marginal_samples, weights=weights)
        plt.savefig(f'figures/hw6/{plotname}_{i}.png')
        plt.clf()

def plot_posterior_jointly(plotname, traces, x_dim, y_dim, weights=None):
    # Plot (optionally weighted) histograms of latent variables per dimension in one image
    if weights is None:
        weights = [0.]*len(traces)
    weights     = [math.exp(weight) for weight in weights]
    weights     = torch.Tensor(weights)/sum(weights)
    num_latent  = len(traces[0].flatten()) if type(traces[0]) == torch.Tensor else 1
    fig, axs = plt.subplots(y_dim,x_dim)
    for i in range(num_latent):
        marginal_samples = []
        for trace in traces:
            marginal_sample = trace.unsqueeze(-1).flatten()[i]
            marginal_samples.append(marginal_sample)
        axs[i//x_dim,i%x_dim].hist(marginal_samples, weights=weights)
    plt.tight_layout()
    plt.savefig(f'figures/hw6/{plotname}_joint.png')
