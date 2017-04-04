from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


def matrix_setup(n):
    """
    """
    N = n**2 # Number of points
    h = 1./(n+1) # gridspacing
    A = np.zeros([N, N])
    A += np.diag(np.ones(N)*-4, 0)
    diagonal2 = np.ones(N-1)
    for i in range(n-1, N-1, n):
        diagonal2[i] = 0
    A += np.diag(diagonal2, 1) + np.diag(diagonal2, -1) \
        + np.diag(np.ones(N-n), n) + np.diag(np.ones(N-n), -n)
    A = A/(h**2)
    #
    b=np.zeros(N)
    b[(N-1)/2]=2
    
    return A,b

def stencil2_setup(n):
    """
    """
    N = n**2 # Number of points
    h = 1./(n+1) # gridspacing
    A = np.zeros([N, N])

    lead_diag = np.diag(np.ones(N)*-60, 0)
    
    outer_diags1 = np.ones(N-1)*16
    outer_diags1[n-1::n] = 0
    outer_diags1 = np.diag(outer_diags1, 1) + np.diag(outer_diags1, -1)

    outer_diags2 = np.ones(N-2)*-1
    for i in range(n,N-2,n):        
        outer_diags2[i-1]=0
        outer_diags2[i-2]=0
    outer_diags2 = np.diag(outer_diags2, 2) + np.diag(outer_diags2, -2)

    #Diagonals dependent on n
    n_diags1 = np.diag(np.ones(N-n)*16, n) + np.diag(np.ones(N-n)*16, -n)
    n_diags2 = np.diag(np.ones(N-2*n)*-1, 2*n) + np.diag(np.ones(N-2*n)*-1, -2*n)

    #Populate the matrix A
    A += lead_diag + outer_diags1 + outer_diags2 + n_diags1 + n_diags2
    A = A/(12*h**2 )

    #Populate b matrix
    b=np.zeros(N)
    b[(N-1)/2]=2.0   

    return A,b






    


def solve(stencil,method,A,b,k=10,p=1,omega=1.0,tol = 1.0e-9):
    """
    """
    N = b.size
    n=np.sqrt(N)
    h = 1./(n+1)
    
    if method == "inbuilt":
        #
        u = np.linalg.solve(A,b)

    elif method == "SOR":

        """
        Define a function that will carry out the SOR iterations
        """        
        def sor(omega=1.):
            u_old = u.copy()
            for i in range(N):
                sigma1 = np.dot(A[i, 0:i], u[0:i])
                sigma2 = np.dot(A[i, i+1:-1], u[i+1:-1])
                u[i] = (omega / A[i,i]) * (b[i] - sigma1 - sigma2) + (1 - omega) * u_old[i]  
            du = np.sqrt(np.dot(u-u_old,u-u_old))
            return du,u
    
        #For initial guess set u=0 everywhere except for a spike at the centre where u=-1
        u = np.zeros(N)
        u[(N-1)/2] = -1
        du_new=0

        """
        Carry out k interations with omega = 1 (k = 10)
        Use this to find the change in u between iterations in order to find
        an optimum value for omega as shown on p88 Numerical Methods in Engineering With
        Python 3 (Jaan Kiusalaas)
        """

        for i in range(k+p):
                du_old = du_new
                du_new,u = sor()
                
        omega_opt = 2.0/(1.0 + np.sqrt(1.0 - (float(du_new)/du_old)**(1.0/p)))
        print("An optimum omega value of ", omega_opt, " has been calculated")

        """
        Now perform subsequent interations using the optimised omega 
        """
          
        for i in range(501):
            du_old = du_new
            du_new,u = sor(omega=omega_opt)             
            if du_new < tol: 
                break
            
        if i == 501:
            raise RuntimeError("SOR method has not converged") 
    
    u = np.reshape(u, [n,n])
    #
    u_tmp = np.zeros([n+2,n+2])
    u_tmp[1:n+1,1:n+1] = u
    u = u_tmp.copy()
        
    midpt = (n+1)/2
    
    if stencil == "5point":
        lapl_midpt = (u[midpt+1,midpt] + u[midpt-1,midpt] - 4*u[midpt,midpt] + u[midpt,midpt + 1] + u[midpt,midpt -1])/h**2
    if stencil == "9point":
        lapl_midpt = (-u[midpt,midpt-2] + 16 *(u[midpt,midpt-1] + u[midpt,midpt+1] + u[midpt-1,midpt] + u[midpt+1,midpt])\
                      - 60. * u[midpt,midpt] - u[midpt,midpt+2] - u[midpt-2,midpt] - u[midpt+2,midpt])/(12.*h**2)
    err = abs(2. - lapl_midpt)
    print("Laplacian at mid point = {}, Error (absolute) = {}".format(lapl_midpt, err))
        
    return u



if __name__ == '__main__':
    n = 7
    
    #stencil = "5point"    
    #A,b = matrix_setup(n)
    stencil = "9point"
    A,b = stencil2_setup(n)
    
    u = solve(stencil,"inbuilt",A,b)



##    #3D Plotting part
##    fig = plt.figure()
##    ax = fig.add_subplot(111, projection='3d')
##    X = np.linspace(0,1,n+2)
##    Y = np.linspace(0,1,n+2)
##    X, Y = np.meshgrid(X, Y)
##
##    surf = ax.plot_wireframe(X, Y, u)
##    ax.set_xlabel("x")
##    ax.set_ylabel("y")
##    ax.set_zlabel("u")
##    plt.show()


