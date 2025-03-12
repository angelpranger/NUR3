#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

Nsat=100
a=2.4
b=0.25
c=1.6

xmin, xmax = 10**-4, 5

def n(x,A,Nsat,a,b,c):
    return A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)

def integrand(x,a=2.4,b=0.25,c=1.6):
    return x**2*((x/b)**(a-3))*np.exp(-(x/b)**c)

# 1a

def extended_midpoint_Romberg(f, a, b, m=5):
    """Computes the integral of f between a and b (with a<b) using the open method.
    Returns the best estimate value of the integral and an estimate for the error."""
    one_third = 1/3
    # Set initial step size
    h = b-a
    # Initialize array of estimates r of size m to zero
    r = np.zeros(m)
    # Calculate initial estimate using midpoint rule for N=1
    r[0] = h*f(b-a)
    # Computing estimates for N=3, N=9, etc
    Np = 2
    for i in range(1, m):
        h *= one_third
        x = a + h*0.5
        # Add midpoint rule for new points
        for _ in range(Np):
            r[i] += h*f(x)
            x += 2*h
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

# TODO test
A = 1/(4*np.pi*extended_midpoint_Romberg(integrand, 0, xmax)[0])
print(f"A is {A}")

# 1b

# slice sampling



def RNG(state1:np.int64, N:np.int64):
    """Returns an array containing N unequal uniformly random unsigned integers."""
    if (state1 <= 0):
        "The seed for the RNG is invalid."
        return
    randoms = np.zeros(N)
    for i in range(N):
        # 64-bit XOR-shift method
        state1 = state1|(state1>>np.int64(21))
        state1 = state1|(state1<<np.int64(35))
        state1 = state1|(state1>>np.int64(4))
        # Multiple linear congruential generator
        state2 = (123456789*state1) % 42943527
        # Add random number
        randoms[i] = state2
    return randoms

# 1c

# selecting 100 random unequal integers for galaxy selection, with seed=97
galaxy_selection_idx = RNG(97, 100)
# TODO dit is incorrect
# random integers scaled to [0, 10000)
galaxy_selection_idx = np.floor(galaxy_selection_idx / np.max(galaxy_selection_idx) * 10000)

# TODO apply galaxy_selection_idx to the 10000 galaxies

def quicksort_recursive(array, low, high):
    """Recursively sort array from low to high using quicksort algorithm. Alters the passed array."""
    # Set pivot to middle element
    pivot = np.ceil((low+high)/2)
    x_pivot = array[pivot]
    # Looping to sort elements with respect to the pivot
    j = high
    for i in range(low, high+1):
        if (array[i] >= x_pivot):
            while (array[j] > x_pivot):
                j -= 1
            if (j<=i):
                break
            else:
                mem = array[i] #TODO .copy?
                array[i] = array[j]
                array[j] = mem
                if (i == pivot):
                    pivot = j
                elif (j == pivot):
                    pivot = i
    # Apply algorithm recursively to subarrays left and right of the pivot
    quicksort_recursive(array, low, pivot-1)
    quicksort_recursive(array, pivot+1, high)
    return

def quicksort(array:np.array):
    """Sort array using quicksort algorithm. Returns the sorted array."""
    # Select pivot as median of the first, last and middle element
    ar = np.array([array[0], array[-1], array[(array.shape[0]+1)>>1]])
    low = np.min(ar)
    middle = np.median(ar)
    high = np.max(ar)
    array[0] = low
    array[-1] = high
    array[(array.shape[0]+1)>>1] = middle
    # Apply quicksort algorithm to the array
    quicksort_recursive(array, 0, array.shape[0])
    return array
    
# 1d

def analytical_derivative_n(x,A,Nsat,a,b,c):
    """Returns the value of the analytical derivative of n at the point x."""
    return A*Nsat*((x/b)**(a-4))*(a-3)*np.exp(-(x/b)**c)*c*(-(x/b)**(c-1))/b**2

def Ridders_differentiation(f, x, m=5, h=0.1, d=2, target_error=1e-14):
    # Calculate first approximations to f'(x) using central differences
    approx = np.zeros(m)
    d_inverse = 1/d
    error_old = 0
    for i in range(m):
        approx[i] = (f(x+h)-f(x-h)) * 0.5 / h
        h *= d_inverse
    for j in range(1, m):
        factor = 1/(np.power(d, 2*j)-1)
        for i in range(m-j):
            approx[i] = (np.power(d, 2*j)*approx[i+1]-approx[i])*factor
        error_new = np.absolute(approx[0]-approx[i+1])
        if (error_new < target_error):
            return approx[0], error_new
        if (error_old < error_new):
            return approx[i+1], error_old
        error_old = error_new
    return approx[0], error_new

print(f"The analytical derivative of n at x=1 is {analytical_derivative_n(1,A,Nsat,a,b,c)}")
print(f"The numerical derivative of n at x=1 is {Ridders_differentiation(n,1)}")
# TODO check 12 significant digits

# TODO look at increasing m



# Plot of histogram in log-log space with line (question 1b)
N_generate = 10000

# 21 edges of 20 bins in log-space
edges = 10**np.linspace(np.log10(xmin), np.log10(xmax), 21)
hist = np.histogram(xmin + np.sort(np.random.rand(N_generate))*(xmax-xmin), bins=edges)[0] #replace!
hist_scaled = 1e-3*hist #replace; this is NOT what you should be plotting, this is just a random example to get a plot with reasonable y values (think about how you *should* scale hist)

relative_radius = edges.copy() #replace!
analytical_function = edges.copy() #replace

fig1b, ax = plt.subplots()
ax.stairs(hist_scaled, edges=edges, fill=True, label='Satellite galaxies') #just an example line, correct this!
plt.plot(relative_radius, analytical_function, 'r-', label='Analytical solution') #correct this according to the exercise!
ax.set(xlim=(xmin, xmax), ylim=(10**(-3), 10), yscale='log', xscale='log',
       xlabel='Relative radius', ylabel='Number of galaxies')
ax.legend()
plt.savefig('my_solution_1b.png', dpi=600)

# Cumulative plot of the chosen galaxies (1c)
chosen = xmin + np.sort(np.random.rand(Nsat))*(xmax-xmin) #replace!
fig1c, ax = plt.subplots()
ax.plot(chosen, np.arange(100))
ax.set(xscale='log', xlabel='Relative radius', 
       ylabel='Cumulative number of galaxies',
       xlim=(xmin, xmax), ylim=(0, 100))
plt.savefig('my_solution_1c.png', dpi=600)