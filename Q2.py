#!/usr/bin/env python
import numpy as np
import time

k=1.38e-16 # erg/K
aB = 2e-13 # cm^3 / s
A = 5e-10 # erg
xi = 1e-15# /s

Z = 0.015
psi = 0.929
Tc = 1e4 # K

# here no need for nH nor ne as they cancel out
def equilibrium1(T, Z=Z, Tc=Tc, psi=psi):
    return psi*Tc*k - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T*k

class Equilibrium2:
    def __init__(self, nH):
        self.nH = nH
        pass

    def equilibrium2(self, T):
        return (psi*Tc - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T - .54 * ( T/1e4 )**.37 * T)*k*self.nH*aB + A*xi + 8.9e-26 * (T/1e4)

# 2a

def bisection_root_step(f, a, b):
    c = (a+b)*0.5
    if (f(a)*f(c) < 0):
        b = c
    else:
        a = c
    return a, b

def false_position_method(f, a, b, max_iterations=50, target_abs=0.1, target_rel=1e-10, safeguards=False):
    """Finds the root of a function using the false position method. 
    Stops after max_iterations or when the target accuracy is met.
    Returns interval a, b enclosing the root and the number of iterations used."""
    for steps in range(max_iterations):
        # Save value for f(a) as we need it again later
        f_a = f(a)
        # Linearly estimate root from 2 last guesses (a and b)
        c = b-(b-a)/(f(b)-f_a)*f(b)
        # Find the counterpoint
        if (f_a*f(c) < 0):
            # Apply bisection if the interval has not reduced by at least half
            if ( safeguards & (c > 0.5*(a+b)) ):
                a, b = bisection_root_step(f, a, b)
            else: 
                b = c
                print('h')
        else:
            # Apply bisection if the interval has not reduced by at least half
            if ( safeguards & (c < 0.5*(a+b)) ):
                a, b = bisection_root_step(f, a, b)
            else:
                a = c
                print('h')
        if (((b-a) < np.absolute(target_rel*a)) | ((b-a) < target_abs)):
            break
    return a, b, steps+1

def root_from_interval(f, a, b):
    if (np.absolute(f(a)) < np.absolute(f(b))):
        return a
    else:
        return b

start = time.time()
a, b, iterations = false_position_method(equilibrium1, 1, 1e7, target_abs=0.1, safeguards=True)
print(f"The equilibrium temperature (root) is {root_from_interval(equilibrium1, a, b)} K with an error estimate of {b-a:.3} K.")
end = time.time()
print(f"The execution time is {(end-start)*10**3:.3} ms.")
print(f"The number of iterations used is {iterations}.")

# 2b

print(f"The equilibrium temperature (root) and estimated error are given.")
print("n_e [cm$^{-3}$]     T_equilibrium [K]     estimated absolute error [K]     time [ms]     iterations")
for n_e in [1e-4, 1, 1e4]:
    start = time.time()
    func_class = Equilibrium2(n_e)
    a, b, iterations = false_position_method(func_class.equilibrium2, 1, 1e15, target_rel=1e-10, safeguards=True)
    end = time.time()
    print(f"{n_e}     {root_from_interval(func_class.equilibrium2, a, b)}     {b-a:.3}     {(end-start)*10**3:.3}     {iterations}")


# TODO try Newton-Raphson method? maybe in combination with false position method