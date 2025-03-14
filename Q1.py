#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

Nsat=100
a=2.4
b=0.25
c=1.6

xmin, xmax = 10**-4, 5

# Test function for differentiation and integration
def test_func(x):
    return np.sin(x)+3

class n_class:
    def __init__(self,A,Nsat,a,b,c):
        self.A = A
        self.Nsat = Nsat
        self.a = a
        self.b = b
        self.c = c
        pass

    def n(self, x):
        return self.A*self.Nsat*(x/self.b)**(self.a-3)*np.exp(-(x/self.b)**self.c)

    def p(self, x):
        return 4*np.pi*x**2*self.A*(x/self.b)**(self.a-3)*np.exp(-(x/self.b)**self.c)
    
# 1a

def integrand(x,a=a,b=b,c=c):
    return x**2*((x/b)**(a-3))*np.exp(-(x/b)**c)

def extended_midpoint_Romberg(f, a, b, m=8):
    """Computes the integral of f between a and b (with a<b) using the open method.
    Returns the best estimate value of the integral and an estimate for the error."""
    one_third = 1/3
    # Set initial step size
    h = b-a
    # Initialize array of estimates r of size m to zero
    r = np.zeros(m)
    # Calculate initial estimate using midpoint rule for N=1
    r[0] = h*f(0.5*(a+b))
    # Computing estimates for N=3, N=9, etc
    Np = 2
    for i in range(1, m):
        h *= one_third
        x = a + h*0.5
        # Add midpoint rule for new points
        j = 1 # Tracker of step size between new points (either h or 2h)
        for _ in range(Np):
            r[i] += h*f(x)
            x += (j+1)*h # j+1 alternates between 1 and 2
            j = (j+1)%2 # j alternates between 0 and 1
        # Add values for already calculated points
        r[i] += one_third*r[i-1]
        Np *= 3
    # Do weighted combinations
    Np = 1
    for i in range(1, m):
        Np *= 9
        factor = 1/(Np-1)
        for j in range(m-i):
            r[j] = (Np*r[j+1]-r[j])*factor
    return r[0], np.absolute(r[0]-r[1])

# Testing integration
print(f"The test integral and error are {extended_midpoint_Romberg(test_func, 0, 20)}")

# TODO check whether A is correct
A = 1/(4*np.pi*extended_midpoint_Romberg(integrand, 0, xmax)[0])
print(f"A is {A}")

# 1b

n1 = n_class(A, Nsat, a, b, c)

class RNG_class:
    def __init__(self, seed):
        if (seed < 0):
            print("The seed for the RNG is invalid.")
        self.state1 = np.uint64(seed)
        self.m = np.uint64(4930622455819)
        self.a = np.uint64(3741260)
        self.fraction = 1/(self.m-1) # period is m-1
        pass

    def random(self):
        """Returns a uniformly random float between (0,1)."""
        # 64-bit XOR-shift method
        self.state1 = self.state1^(self.state1>>np.uint64(21))
        self.state1 = self.state1^(self.state1<<np.uint64(35))
        self.state1 = self.state1^(self.state1>>np.uint64(4))
        # Multiple linear congruential generator
        random = (self.a*self.state1) % self.m
        # Convert to interval (0,1)
        random = random * self.fraction
        return random
    
# Seed random number generator
RNG1 = RNG_class(seed = 123456789)

# Testing random number generator
randoms = np.zeros(100000)
for i in range(len(randoms)):
    randoms[i] = RNG1.random()
plt.hist(randoms)
plt.savefig('hist.pdf')

def rejection_sampling(p, xmin, xmax, num):
    samples = np.zeros(num)
    i = 0
    while (i < num):
        # Draw a random x between xmin and xmax
        x = xmin + (xmax-xmin)*RNG1.random()
        # Draw random probability between (0,1) and check whether p(x) greater than this
        if (RNG1.random() < p(x)):
            samples[i] = x
            i += 1
    return samples

# Histogram in log-log space
samples = rejection_sampling(n1.p, xmin, xmax, 10000)
# 21 edges of 20 bins in log-space
edges = 10**np.linspace(np.log10(xmin), np.log10(xmax), 21)
hist = np.histogram(samples, bins=edges)[0]
# Correcting for bin width and normalization offset 10000/Nsat=100
hist_scaled = hist / (edges[1:]-edges[:-1]) * 0.01
# Getting analytical solution of N(x) for values between xmin and xmax in log space
relative_radius = 10**np.linspace(np.log10(xmin), np.log10(xmax), 100)
analytical_function = n1.p(relative_radius) * Nsat # N(x)

# Plotting histogram and analytical solution
fig1b, ax = plt.subplots()
ax.stairs(hist_scaled, edges=edges, fill=True, label='Satellite galaxies')
plt.plot(relative_radius, analytical_function, 'r-', label='Analytical solution')
ax.set(xlim=(xmin, xmax), ylim=(10**(-3), 10), yscale='log', xscale='log',
       xlabel='Relative radius', ylabel='Number of galaxies')
ax.legend()
plt.savefig('my_solution_1b.png', dpi=600)

# 1c

def quicksort_recursive(array, low, high):
    """Recursively sort array from low to high using quicksort algorithm. Alters the passed array."""
    # Set pivot to middle element
    pivot = np.int64(np.ceil((low+high)/2))
    x_pivot = array[pivot]
    # Looping to sort elements with respect to the pivot
    i = low
    j = high
    while True:
        while (array[i] < x_pivot):
            i += 1
        while (array[j] > x_pivot):
            j -= 1
        if (j<=i):
            break
        else:
            mem = array[i]
            array[i] = array[j]
            array[j] = mem
            # If pivot is swapped, change location of pivot to new location
            # Let complementary indexer continue to prevent infinite looping in case array[i]=array[j]=array[pivot]
            if (i == pivot):
                pivot = j
                i += 1 
            elif (j == pivot):
                pivot = i
                j -= 1
    # Apply algorithm recursively to subarrays left and right of the pivot
    if (low < pivot-1):
        array = quicksort_recursive(array, low, pivot-1)
    if (high > pivot+1):
        array = quicksort_recursive(array, pivot+1, high)
    return array

def quicksort(array):
    """Sort array using quicksort algorithm. Returns the sorted array."""
    # Sort first, last and middle element as pre-step
    ar = np.array([array[0], array[-1], array[(array.shape[0])>>1]])
    low = np.min(ar)
    middle = np.median(ar)
    high = np.max(ar)
    array[0] = low
    array[-1] = high
    array[(array.shape[0])>>1] = middle
    # Apply quicksort algorithm to the array
    quicksort_recursive(array, 0, array.shape[0]-1)
    return array

def random_samples_from_array(array, num):
    samples = np.zeros(num)
    for i in range(num):
        # Draw random integer (index) in [0, N-1] with N the current size of array
        idx = np.int64(np.floor(array.shape[0]*RNG1.random()))
        samples[i] = array[idx]
        array = np.delete(array, idx)
    return samples

# Select 100 random samples from the previous 10000 samples from (b)
samples_100 = random_samples_from_array(samples, 100)
samples_100 = quicksort(samples_100)

# Cumulative plot of the chosen galaxies
fig1c, ax = plt.subplots()
ax.plot(samples_100, np.arange(100))
ax.set(xscale='log', xlabel='Relative radius', 
       ylabel='Cumulative number of galaxies',
       xlim=(xmin, xmax), ylim=(0, 100))
plt.savefig('my_solution_1c.png', dpi=600)

# 1d

def analytical_derivative_n(x,A,Nsat,a,b,c):
    """Returns the value of the analytical derivative of n at the point x."""
    return -A*Nsat/b**2*(a-3)*c*(x/b)**(a+c-5)*np.exp(-(x/b)**c)
# TODO is this correct? ik denk het niet...

def Ridders_differentiation(f, x, m=5, h=0.1, d=2, target_error=1e-13):
    # Calculate first approximations to f'(x) using central differences
    approx = np.zeros(m)
    d_inverse = 1/d
    error_old = np.inf
    for i in range(m):
        approx[i] = (f(x+h)-f(x-h)) * 0.5 / h
        h *= d_inverse
    print(approx)
    # Combine pairs of approximations
    for j in range(1, m):
        power = np.power(d, 2*j)
        factor = 1/(power-1)
        for i in range(m-j):
            approx[i] = (power*approx[i+1]-approx[i])*factor
        # Terminate when improvement over previous best approximation is smaller than target error
        error_new = np.absolute(approx[0]-approx[i+1])
        if (error_new < target_error):
            print('hi')
            return approx[0], error_new
        # Terminate early if the error grows, return best approximation from before
        if (error_old < error_new):
            print('ho')
            return approx[i+1], error_old
        error_old = error_new
        print(j)
    return approx[0], error_new

print(f"The analytical derivative of n at x=1 is {analytical_derivative_n(1,A,Nsat,a,b,c)}")
print(f"The numerical derivative of n at x=1 is {Ridders_differentiation(n1.n,1,m=10,h=0.1)[0]}")
# TODO check 12 significant digits

# TODO look at increasing m (niet echt nodig...)


print(Ridders_differentiation(test_func, 1))
