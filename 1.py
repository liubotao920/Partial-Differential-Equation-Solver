from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


def matrix_setup(n):
    """
    """
    N = n**2
    h = 1./(n+1)
    A = np.zeros([N, N])
    A += np.diag(np.ones(N)*-4, 0)
    diagonal2 = np.ones(N-1)
    for i in range(n-1, N-1, n):
        diagonal2[i] = 0
    A += np.diag(diagonal2, 1) + np.diag(diagonal2, -1)
    A += np.diag(np.ones(N-n), n) + np.diag(np.ones(N-n), -n)
    A = A/(h**2)
    #
    b=np.zeros(N)
    b[(N-1)/2]=2
    
    return A,b

def solve(method,A,b):
    N = b.size
    n=np.sqrt(N)
    h = 1./(n+1)
    
    if method == "inbuilt":
        #
        u = np.linalg.solve(A,b)
        
    u = np.reshape(u, [n,n])
    #
    u_tmp = np.zeros([n+2,n+2])
    u_tmp[1:n+1,1:n+1] = u
    u = u_tmp.copy()
        
    midpt = (n+1)/2
    lapl_midpt = (u[midpt+1,midpt] + u[midpt-1,midpt] - 4*u[midpt,midpt] + u[midpt,midpt + 1] + u[midpt,midpt -1])/h**2
    err = 2. - lapl_midpt
    print("Laplacian at mid point = {}, Error = {}".format(lapl_midpt, err)) 
        
    return u_tmp



if __name__ == '__main__':
    n = 9
    A,b = matrix_setup(n)
    u = solve("inbuilt",A,b)

    #3D Plotting part
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(0,1,n+2)
    Y = np.linspace(0,1,n+2)
    X, Y = np.meshgrid(X, Y)

    surf = ax.plot_wireframe(X, Y, u)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    plt.show()


