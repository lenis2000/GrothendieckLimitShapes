import cmath
import numpy as np

xpar, ypar, beta = 1/80.0, 1/80.0, -0.00000000000001

epsilon = 0.0001

# If true, will be slower and will calculate the whole height function for Schur Process
# If false, will be faster and will only calculate the L function for Grothendieck

need_sums = True

max_xi = 1.09
min_tau = 0.999
max_tau = 1.0
min_xi_sums = 0.92
max_xi_sums = max_xi

progress_step = 5

def G(z, xi, tau, x, y, beta):
    return (z**2 - x*y*z**2 + y*beta - 2*z*beta + x*z**2*beta + y*z*xi
            - z**2*xi - x*y*z**2*xi + x*z**3*xi - y*beta*xi
            + z*beta*xi + x*y*z*beta*xi - x*z**2*beta*xi
            - y*beta*tau + z*beta*tau + x*y*z*beta*tau
            - x*z**2*beta*tau)

def d(xi, tau, x, y, beta):
    return y*beta - y*beta*xi - y*beta*tau

def c(xi, tau, x, y, beta):
    return -2*beta + y*xi + beta*xi + x*y*beta*xi + beta*tau + x*y*beta*tau

def b(xi, tau, x, y, beta):
    return 1 - x*y + x*beta - xi - x*y*xi - x*beta*xi - x*beta*tau

def a(xi, tau, x, y, beta):
    return x*xi

# Main Function takes in the coefficient of the Cubic Polynomial
# as parameters and it returns the roots in form of numpy array.
# Polynomial Structure -> ax^3 + bx^2 + cx + d = 0

def solve(a, b, c, d):

    if (a == 0 and b == 0):                     # Case for handling Liner Equation
        return np.array([(-d * 1.0) / c])                 # Returning linear root as numpy array.

    elif (a == 0):                              # Case for handling Quadratic Equations

        D = c * c - 4.0 * b * d                       # Helper Temporary Variable
        if D >= 0:
            D = cmath.sqrt(D)
            x1 = (-c + D) / (2.0 * b)
            x2 = (-c - D) / (2.0 * b)
        else:
            D = cmath.sqrt(-D)
            x1 = (-c + D * 1j) / (2.0 * b)
            x2 = (-c - D * 1j) / (2.0 * b)
            
        return np.array([x1, x2])               # Returning Quadratic Roots as numpy array.

    f = findF(a, b, c)                          # Helper Temporary Variable
    g = findG(a, b, c, d)                       # Helper Temporary Variable
    h = findH(g, f)                             # Helper Temporary Variable

    if f == 0 and g == 0 and h == 0:            # All 3 Roots are Real and Equal
        if (d / a) >= 0:
            x = (d / (1.0 * a)) ** (1 / 3.0) * -1
        else:
            x = (-d / (1.0 * a)) ** (1 / 3.0)
        return np.array([x, x, x])              # Returning Equal Roots as numpy array.

    elif h <= 0:                                # All 3 roots are Real

        i = cmath.sqrt(((g ** 2.0) / 4.0) - h)   # Helper Temporary Variable
        j = i ** (1 / 3.0)                      # Helper Temporary Variable
        k = cmath.acos(-(g / (2 * i)))           # Helper Temporary Variable
        L = j * -1                              # Helper Temporary Variable
        M = cmath.cos(k / 3.0)                   # Helper Temporary Variable
        N = cmath.sqrt(3) * cmath.sin(k / 3.0)    # Helper Temporary Variable
        P = (b / (3.0 * a)) * -1                # Helper Temporary Variable

        x1 = 2 * j * cmath.cos(k / 3.0) - (b / (3.0 * a))
        x2 = L * (M + N) + P
        x3 = L * (M - N) + P

        return np.array([x1, x2, x3])           # Returning Real Roots as numpy array.

    elif h > 0:                                 # One Real Root and two Complex Roots
        R = -(g / 2.0) + cmath.sqrt(h)           # Helper Temporary Variable
        if R >= 0:
            S = R ** (1 / 3.0)                  # Helper Temporary Variable
        else:
            S = (-R) ** (1 / 3.0) * -1          # Helper Temporary Variable
        T = -(g / 2.0) - cmath.sqrt(h)
        if T >= 0:
            U = (T ** (1 / 3.0))                # Helper Temporary Variable
        else:
            U = ((-T) ** (1 / 3.0)) * -1        # Helper Temporary Variable

        x1 = (S + U) - (b / (3.0 * a))
        x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * cmath.sqrt(3) * 0.5j
        x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * cmath.sqrt(3) * 0.5j

        return np.array([x1, x2, x3])           # Returning One Real Root and two Complex Roots as numpy array.


# Helper function to return float value of f.
def findF(a, b, c):
    return ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0


# Helper function to return float value of g.
def findG(a, b, c, d):
    return (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a **2.0)) + (27.0 * d / a)) /27.0


# Helper function to return float value of h.
def findH(g, f):
    return ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)


def find_positive_imaginary_root(roots,xi,tau):
    for root in roots:
        if root.imag > 0:
            return root
    
    # Special code handling the frozen boundary for beta close to 0 and small x,y:
    if(xi<1):
        return -1
    else:
        return 1

    # Special code handling the frozen boundary for beta=-100, x=y=1/40:
    # if(xi < 0.285464 and tau > 0.600723):
    #     return -1
    # if(xi < 0.285464 and tau < 0.600723):
    #     return 1
    # if(xi + tau < 1.96):
    #     return -1
    # return 1

    # Special code handling the frozen boundary for beta=-6, x=1/3, y=1/5:
    # if(xi < 0.301 and tau > 0.533868):
    #     return -1
    # if(xi < 0.301 and tau < 0.533868):
    #     return 1
    # if(xi + tau < 2):
    #     return -1
    # return 1

    # Special code handling the frozen boundary for xpar, ypar, beta = 1/3.0, 1/5.0, 1/10.0
    # if(xi < 1):
    #     return -1
    # if(xi > 1.5):
    #     return 1

# Range of xi and tau values
xi_range = np.linspace(0, max_xi, int(max_xi/epsilon))
# print(xi_range)
tau_range = np.linspace(min_tau, max_tau, int((max_tau-min_tau)/epsilon))
# print(tau_range)

# Create a dictionary to store roots for each (xi, tau) pair
positive_imaginary_roots = {}
positive_imaginary_roots_args = {}

current_iteration = 0
total_iterations = len(xi_range) * len(tau_range)

for xi in xi_range:
    for tau in tau_range:
        roots = solve(a(xi, tau, xpar, ypar, beta), b(xi, tau, xpar, ypar, beta), c(xi, tau, xpar, ypar, beta), d(xi, tau, xpar, ypar, beta))
        positive_imaginary_root = find_positive_imaginary_root(roots,xi,tau)
        positive_imaginary_roots[(xi, tau)] = positive_imaginary_root
        positive_imaginary_roots_args[(xi, tau)] = np.angle(positive_imaginary_root) / np.pi
       
    #    Progress report
        current_iteration += 1
        progress = 100 * current_iteration / total_iterations
        if current_iteration % int(progress_step / 100 * total_iterations) == 0:
            print(f"Root progress: {progress:.2f}%")



# Create a list of tuples containing xi, tau, and root values
roots_list = [(key[0], key[1], value) for key, value in positive_imaginary_roots.items()]

# Convert the list to a string representation in Mathematica format
roots_str = "{" + ", ".join([f"{{{x:.16f}, {tau:.16f}, {root.real:.16f} + {root.imag:.16f} I}}" for x, tau, root in roots_list]) + "}"

# Save the string representation to a file
with open("roots.txt", "w") as file:
    file.write(roots_str)

def linear_interpolation(x, x1, x2, y1, y2):
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


xi_range_sums = np.linspace(min_xi_sums, max_xi_sums, int((max_xi_sums - min_xi_sums)/epsilon))
# print(xi_range)
tau_range_sums = np.linspace(min_tau, max_tau, int((max_tau - min_tau)/epsilon))

if(need_sums == True):

    sums = {}

    current_iteration = 0
    total_iterations = len(xi_range_sums) * len(tau_range_sums)

    for tau in tau_range_sums:
        for xi in xi_range_sums:
            total_sum = 0
            for xi1 in xi_range:
                if xi <= xi1 < max_xi:
                    total_sum += positive_imaginary_roots_args[(xi1, tau)]
            sums[(xi, tau)] = total_sum * epsilon

        #    Progress report
            current_iteration += 1
            progress = 100 * current_iteration / total_iterations
            if current_iteration % int(progress_step / 100 * total_iterations) == 0:
                print(f"Sum progress: {progress:.2f}%")


    sums_list = [(key[0], key[1], value) for key, value in sums.items()]
    sums_str = "{" + ", ".join([f"{{{x:.16f}, {tau:.16f}, {val:.16f}}}" for x, tau, val in sums_list]) + "}"
    with open("sums.txt", "w") as file:
        file.write(sums_str)

    L = {}

    for tau in tau_range_sums:
        found = False
        closest_lower = None
        closest_upper = None

        for xi in xi_range_sums:
            if sums[(xi, tau)] == tau:
                L[tau] = xi
                found = True
                break
            elif sums[(xi, tau)] < tau:
                if closest_lower is None or sums[(xi, tau)] > sums[(closest_lower, tau)]:
                    closest_lower = xi
            else:  # sums[(xi, tau)] > tau
                if closest_upper is None or sums[(xi, tau)] < sums[(closest_upper, tau)]:
                    closest_upper = xi

        if not found:
            if closest_lower is not None and closest_upper is not None:
                L[tau] = linear_interpolation(tau, sums[(closest_lower, tau)], sums[(closest_upper, tau)], closest_lower, closest_upper)
            else:
                L[tau] = None  # Or any default value if interpolation is not possible

    L[1.0] = 0.0

else:

    L = {}
    current_iteration = 0
    total_iterations = len(xi_range_sums) * len(tau_range_sums)

    for tau in tau_range_sums:
        found = False
        closest_lower = None
        closest_upper = None
        total_sum = 0

        for xi in xi_range_sums:
            total_sum = 0
            for xi1 in xi_range:
                if xi <= xi1 < max_xi:
                    total_sum += positive_imaginary_roots_args[(xi1, tau)]

            total_sum *= epsilon

            if total_sum == tau:
                L[tau] = xi
                found = True
                break
            elif total_sum < tau:
                if closest_lower is None or total_sum > closest_lower[1]:
                    closest_lower = (xi, total_sum)
            else:  # total_sum > tau
                if closest_upper is None or total_sum < closest_upper[1]:
                    closest_upper = (xi, total_sum)

            current_iteration += 1
            progress = 100 * current_iteration / total_iterations
            if current_iteration % int(progress_step / 100 * total_iterations) == 0:
                print(f"Interpolation progress: {progress:.2f}%")

        if not found and closest_lower is not None and closest_upper is not None:
            L[tau] = linear_interpolation(tau, closest_lower[1], closest_upper[1], closest_lower[0], closest_upper[0])
        else:
            L[tau] = None  # Or any default value if interpolation is not possible

    L[1.0] = 0.0

L_list = [(key, value) for key, value in L.items()]
L_str = "{" + ", ".join([f"{{{tau:.16f}, {val:.16f}}}" for tau, val in L_list]) + "}"
with open("L.txt", "w") as file:
    file.write(L_str)
