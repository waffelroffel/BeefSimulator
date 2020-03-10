from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from numpy import array, zeros, diag, diagflat, dot
#import importlib
#importlib.import_module(3Dmodel, package=None)



#Convergence of Jacobi iterations:A sufficient (but not necessary) condition for the method to converge is that the matrix A
# is strictly or irreducibly diagonally dominant. Strict row diagonal dominance means that for each row, the absolute value
# of the diagonal term is greater than the sum of absolute values of other terms.
#The Jacobi method sometimes converges even if these conditions are not satisfied.


def jacobi(A: np.array,b :np.array,N:int,x:array=None) -> np.array:
    """Solves the equation Ax=b via the Jacobi iterative method."""
    #N is the number of time steps
    # Create an initial guess if needed
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A
    # and subtract them from A
    D = diag(A)
    R = A - diagflat(D)

    # Iterate for N times
    for i in range(N):
        x = (b - dot(R,x)) / D
    return x

def convergence_jacobi(A:np.array, b:np.array, Ns:np.array,x:np.array=None):
    """Make a plot of the maximum error against the number of iterations and computes experimental order of convergence"""
    errs=[]
    plt.figure()
    for N in Ns:
        sol = jacobi(A,b,N,x)
        error=abs(np.dot(A, sol)-b)
        max_error=max(error)
        errs.append(max_error)
        #print("For N=", N, "the solution is ", sol)
        print("For N=", N, "the maximum error is", max_error)
        plt.scatter(N, max_error, label="N="+str(N))
    plt.xlabel("N (number of iterations)")
    plt.ylabel("Maximum error")
    plt.legend()
    plt.show()

    EOCs = []

    if len(errs) != len(Ns):
        print("Something is wrong! Length of N list does not match length of error list!")
    else:
        for i in range(1, len(errs)):
            numerator = np.log(errs[i]) - np.log(errs[i - 1])
            denominator = np.log(Ns[i - 1]) - np.log(Ns[i])
            EOC = numerator / denominator
            EOCs.append(EOC)

    for i in range(1, len(EOCs) + 1):
        print("EOC for N = ", Ns[i], " : ", EOCs[i - 1], "\n")

#Example1. Get convergence. A1 is strictly diagonally dominant, which implies convergence.
print("Example 1")
A1 = array([[4.0,1.0],[1.0,-5.0]])
b1 = array([13.0,-2.0])
guess1 = array([0.5,1.0])
N_list1=array([1, 2, 4, 8, 16, 32])
convergence_jacobi(A1, b1, N_list1, x=guess1)
print("\n")

#Example2. Do not get convergence
print("Example 2")
A2=array([[1.0, 2.0], [1.0, 2.0]])
b2=array([3.0, 3.0])
guess2=array([0.1, 2.0])
N_list2=array([1, 2, 4, 8, 16, 32])
convergence_jacobi(A2, b2,  N_list2, x=guess2)

#Example 3.
A3 = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0.0, 3., -1., 8.]])
b3 = np.array([6., 25., -11., 15.])
N_list3=array([1, 2, 4, 8, 16, 32, 64])
convergence_jacobi(A3, b3,  N_list3)

#Convergence test for the Jacobi algorithm in 3Dmodel
def convergence_Jacobi(T0:np.array, C0:np.array,T_ex:np.array, C_ex:np.array, Ns:np.array):
    errsT = []
    errsC=[]
    plt.figure()
    for N in Ns:
        T, C = Jacobi(T0, C0, N)     #NB
        error_T = abs(T-T_ex)
        error_C=abs(C-C_ex)
        max_errorT = max(error_T)
        max_errorC= max(error_C)
        errsT.append(max_errorT)
        errsC.append(max_errorC)
        # print("For N=", N, "the solution is ", sol)
        print("For N=", N, "the maximum error in T is", max_errorT)
        print("For N=", N, "the maximum error in C is", max_errorC)
        plt.scatter(N, max_errorT, label="For T with N=" + str(N))
        plt.scatter(N, max_errorC, label="For C with N="+str(N))
    plt.xlabel("N (number of iterations)")
    plt.ylabel("Maximum error")
    plt.legend()
    plt.show()

#Defining a test case
T0 = np.zeros((5, 5, 5))
T0[:, 0] = 10
#print(T0)
C0 = np.ones_like(T0) * 3
C0[0, :] = 0
#print(C0)

#T0 and C0 both have dimensions 5*5*5
Ns = array([100, 200, 400, 800, 1600])





