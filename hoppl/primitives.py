import torch
import torch.distributions as dist
from copy import deepcopy
torch.random.manual_seed(0)


# Distribution Classes
class Normal(dist.Normal):

    def __init__(self, loc, scale):
        if scale > 20.:
            self.optim_scale = scale.clone().detach().requires_grad_()
        else:
            self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()
        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))

    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]

    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]
        return Normal(*ps)

    def log_prob(self, x):
        self.scale = torch.nn.functional.softplus(self.optim_scale)
        return super().log_prob(x)

class Bernoulli(dist.Bernoulli):

    def __init__(self, probs=None, logits=None):
        if logits is None and probs is None:
            raise ValueError('set probs or logits')
        elif logits is None:
            if type(probs) is float:
                probs = torch.tensor(probs)
            logits = torch.log(probs/(1-probs)) ##will fail if probs = 0
        super().__init__(logits = logits)

    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.logits]

    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        logits = [p.clone().detach().requires_grad_() for p in self.Parameters()][0]
        return Bernoulli(logits = logits)

class Categorical(dist.Categorical):

    def __init__(self, probs=None, logits=None, validate_args=None):
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            probs = probs / probs.sum(-1, keepdim=True)
            logits = dist.utils.probs_to_logits(probs)
        else:
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        super().__init__(logits = logits)
        self.logits = logits.clone().detach().requires_grad_()
        self._param = self.logits

    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.logits]

    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        logits = [p.clone().detach().requires_grad_() for p in self.Parameters()][0]
        return Categorical(logits = logits)    

class Dirichlet(dist.Dirichlet):

    def __init__(self, concentration):
        #NOTE: logits automatically get added
        super().__init__(concentration)

    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.concentration]

    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        concentration = [p.clone().detach().requires_grad_() for p in self.Parameters()][0]
        return Dirichlet(concentration)

class Gamma(dist.Gamma):
    
    def __init__(self, concentration, rate):
        if rate > 20.:
            self.optim_rate = rate.clone().detach().requires_grad_()
        else:
            self.optim_rate = torch.log(torch.exp(rate) - 1).clone().detach().requires_grad_()
        super().__init__(concentration, torch.nn.functional.softplus(self.optim_rate))

    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.concentration, self.optim_rate]

    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        concentration,rate = [p.clone().detach().requires_grad_() for p in self.Parameters()]
        return Gamma(concentration, rate)

    def log_prob(self, x):
        self.rate = torch.nn.functional.softplus(self.optim_rate)
        return super().log_prob(x)

class Dirac:

    def __init__(self, point):
        self.point = point

    def sample(self):
        return self.point

    def log_prob(self, obs):
        value = 0. if obs == self.point else -float('inf')
        return torch.Tensor([value]).squeeze()

class Uniform(dist.Uniform):

    def __init__(self, low, high):
        if high <= low:
            high = 10.
        super().__init__(low, high)

    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.low, self.high]

    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        low, high = [p.clone().detach().requires_grad_() for p in self.Parameters()]
        return Uniform(low, high)

    def log_prob(self, x):
        return super().log_prob(x)


# Primitive Functions
def add(addr,a,b):
    return torch.add(a,b)

def subtract(addr,a,b):
    return torch.subtract(a,b)

def multiply(addr,a,b):
    return torch.multiply(a,b)

def divide(addr,a,b):
    return torch.divide(a,b)

def gt(addr,a,b):
    return torch.Tensor([a>b])

def lt(addr,a,b):
    return torch.Tensor([a<b])

def eq(addr,a,b):
    return torch.Tensor([a==b])

def compare_and(addr,a,b):
    return torch.Tensor([a and b])

def compare_or(addr,a,b):
    return torch.Tensor([a or b])

def sqrt(addr,a):
    return torch.sqrt(a)

def log(addr,a):
    return torch.log(a)

def tanh(addr,a):
    return torch.tanh(a)

def first(addr,data):
    return data[0]

def second(addr,data):
    return data[1]

def rest(addr,data):
    return data[1:]

def last(addr,data):
    return data[-1]

def nth(addr,data,index):
    return data[index]

def abs(addr,data):
    return torch.abs(data)

def conj(addr,data,el):
    if len(el.shape) == 0: el = el.reshape(1)
    return torch.cat([data, el], dim=0)

def cons(addr,data,el):
    if len(el.shape) == 0: el = el.reshape(1)
    return torch.cat([el, data], dim=0)

def vector(*args):
    args = args[1:] # throw out address
    if len(args) == 0:
        return torch.Tensor([])
    # sniff test: if what is inside isn't int,float,or tensor return normal list
    if type(args[0]) not in [int, float, torch.Tensor]:
        return [arg for arg in args]
    # if tensor dimensions are same, return stacked tensor
    if type(args[0]) is torch.Tensor:
        sizes = list(filter(lambda arg: arg.shape == args[0].shape, args))
        if len(sizes) == len(args):
            return torch.stack(args)
        else:
            return [arg for arg in args]
    raise Exception(f'Type of args {args} could not be recognized.')

def hashmap(*args):
    args = args[1:] # throw out address
    result, i = {}, 0
    while i<len(args):
        key, value  = args[i], args[i+1]
        if type(key) is torch.Tensor:
            key = key.item()
        result[key] = value
        i += 2
    return result

def get(addr,struct,index):
    if type(index) is torch.Tensor:
        index = index.item()
    if type(struct) in [torch.Tensor, list, tuple]:
        index = int(index)
    return struct[index]

def put(addr,struct,index,value):
    if type(index) is torch.Tensor:
        index = int(index.item())
    if type(struct) in [torch.Tensor, list, tuple]:
        index = int(index)
    result = deepcopy(struct)
    result[index] = value
    return result

def bernoulli(addr, p, obs=None):
    return Bernoulli(p)

def beta(addr, alpha, beta, obs=None):
    return torch.distributions.Beta(alpha, beta)

def normal(addr, mu, sigma):
    return Normal(mu, sigma)

def uniform(addr, a, b):
    return Uniform(a, b)

def exponential(addr, lamb):
    return torch.distributions.Exponential(lamb)

def discrete(addr, vector):
    return Categorical(vector)

def gamma(addr, concentration, rate):
    return Gamma(concentration, rate)

def dirichlet(addr, concentration):
    return Dirichlet(concentration)

def dirac(addr, point):
    return Dirac(point)

def transpose(addr, tensor):
    return tensor.T

def repmat(addr, tensor, size1, size2):
    if type(size1) is torch.Tensor: size1 = int(size1.item())
    if type(size2) is torch.Tensor: size2 = int(size2.item())
    return tensor.repeat(size1, size2)

def matmul(addr, t1, t2):
    return t1.matmul(t2)

def empty(addr, tensor):
    return torch.Tensor([len(tensor) == 0])

def push_addr(alpha, value):
    return alpha + value


env = {
    'push-address': push_addr,
    "+": add,
    "-": subtract,
    "*": multiply,
    "/": divide,
    ">": gt,
    "<": lt,
    "=": eq,
    "sqrt": sqrt,
    "log": log,
    "first": first,
    "peek": last,
    "second": second,
    "rest": rest,
    "last": last,
    "nth": nth,
    "append": conj,
    "and": compare_and,
    "or": compare_or,
    "abs": abs,
    "conj": conj,
    "cons": cons,
    "vector": vector,
    "hash-map": hashmap,
    "list": vector,
    "get": get,
    "put": put,
    "flip": bernoulli,
    "beta": beta,
    "normal": normal,
    "uniform": uniform,
    "uniform-continuous": uniform,
    "exponential": exponential,
    "discrete": discrete,
    "gamma": gamma,
    "dirichlet": dirichlet,
    "dirac": dirac,
    "mat-transpose": transpose,
    "mat-add": add,
    "mat-tanh": tanh,
    "mat-repmat": repmat,
    "mat-mul": matmul,
    "empty?": empty
}
