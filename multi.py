import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
 

#------------------------------------------------------------------------------------------
def matrix_setup(n):
    """
    This function sets up the matrices A and b using the first order central differences
    stencil to discretise Poisson's equation in 2D.
    
    """
    N = n**2 # Number of points
    h = 1./(n+1) # gridspacing
    A = np.zeros([N, N]) # initialise A
    
    #Diagonals
    lead_diag = np.diag(np.ones(N)*-4, 0)
    outer_diags = np.ones(N-1)
    for i in range(n-1, N-1, n):
        outer_diags[i] = 0
    outer_diags = np.diag(outer_diags, 1) + np.diag(outer_diags, -1)

    #Diagonals dependent on n
    n_diags = np.diag(np.ones(N-n), n) + np.diag(np.ones(N-n), -n)

    #Populate A matrix
    A += lead_diag + outer_diags + n_diags
    A = A/(h**2)
    
    #Populate the RHS b matrix
    b=np.zeros(N)
    b[(N-1)/2]=2
    
    return A,b # The matrix problem A.u = b can now be solved to give a solution to the PDE.
#------------------------------------------------------------------------------------------
def stencil2_setup(n):
    """
    This function creates the A and b matrices fpr the higher order central differences
    stencil.
    
    """
    N = n**2 # Number of points
    h = 1./(n+1) # gridspacing
    A = np.zeros([N, N])# initialise A

    #Diagonals
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

    #Populate the RHS b matrix
    b=np.zeros(N)
    b[(N-1)/2]=2.0 # Forcing term  

    return A,b # The matrix problem A.u = b can now be solved to give a solution to the PDE.
#------------------------------------------------------------------------------------------
def solve(method,stencil,n=11,k=10,p=1,omega=1.0,tol = 1.0e-9,n_its=501,printing="ON",plotting="OFF"):
    """
    This function provides the solution to the linear system of equations given by A.u = b
    and thus returns u.
    _______________________________________________________________________________

    Parameters for the Successive Over-Relaxation (SOR) and "Red-Black" solver code:
    _______________________________________________________________________________

    omega = the relaxation factor, see p88 in Numerical Methods in Engineering With Python 3
            (Jaan Kiusalaas)
    
    k = the minimum number of iterations required before calculating an optimum value for
        omega

    p = some positive integer that represents additional iterations before calculating
        an optimised omega

    tol = the tolerance within which the code will accept that convergence has occured.

    n_its = no. of iterations the code will carry out
    
    """
    N = n**2
    h = 1./(n+1)

    if stencil == "5point":
        A,b = matrix_setup(n)
    elif stencil == "9point":
        A,b = stencil2_setup(n)
        
    
    #initialise u matrix for the SOR solver and for the "red-black" formulation.
    u = np.zeros(N)
    u[(N-1)/2] = -1 #For initial guess set u=0 everywhere except for a spike at the centre where u=-1
    du_new=0
#------------------------------------------------------------------------------------------  
        
    def sor(omega=1.):
        """
        This function that will carry out the Successive Over-Relaxation iterations, as seen
        on p6 of the course notes, that will be used for SOR.
        
        """
        u_old = u.copy()
        for i in range(N):
            sigma1 = np.dot(A[i, 0:i], u[0:i])
            sigma2 = np.dot(A[i, i+1:-1], u[i+1:-1])
            u[i] = (omega / A[i,i]) * (b[i] - sigma1 - sigma2) + (1 - omega) * u_old[i]  
        du = np.sqrt(np.dot(u-u_old,u-u_old))
        return du,u

    
#------------------------------------------------------------------------------------------    
    if method == "inbuilt":
        """
        This piece of code will use numpy's built-in linear algebra library to solve the
        matrix problem A.u=b
        
        """
        u = np.linalg.solve(A,b)
#------------------------------------------------------------------------------------------
    elif method == "SOR":      
    
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
        if printing == "ON":
            print("Optimum omega = {}".format(omega_opt))

        """
        Now perform subsequent interations using the optimised omega
        
        """
          
        for i in range(n_its):
            du_old = du_new
            du_new,u = sor(omega=omega_opt)             
            if du_new < tol: 
                break
            
        if i == n_its:
            raise RuntimeError("SOR method has not converged")
#------------------------------------------------------------------------------------------
   
    elif method == "redblack":
        """
        Here the Red-Black solver will iterate over all of the 'red' and then all of the 'black'
        nodes, employing a vanilla Gauss-Seidel solver. When u is represented as a column vector
        the 'red' nodes are those with even indices and the 'black' nodes are those with odd indices.
    
        """
        for i in range(1,n_its):
            u_old = u.copy()
            for i in range(0,N,2): #This gives the even i values, i.e. the indices for the red nodes                        
                sigma1 = np.dot(A[i, 0:i], u[0:i])
                sigma2 = np.dot(A[i, i+1:-1], u[i+1:-1])
                u[i] = (1. / A[i,i]) * (b[i] - sigma1 - sigma2) 

            for i in range(1,N,2): #This gives the odd i values, i.e. the indices for the black nodes
                
                sigma1 = np.dot(A[i, 0:i], u[0:i])
                sigma2 = np.dot(A[i, i+1:-1], u[i+1:-1])
                u[i] = (1. / A[i,i]) * (b[i] - sigma1 - sigma2) 

            du = np.sqrt(np.dot(u-u_old,u-u_old))                     
                
            if du < tol: 
                break
            
        if i == n_its:
            raise RuntimeError("Red-Black method has not converged")
#------------------------------------------------------------------------------------------
    """
    Here the row vector u is reshaped into an nxn matrix. Then the Laplacian at the midpoint
    is found using the central difference approximation (dependent upon which stencil
    has been used).
    
    """
       
    u = np.reshape(u, [n,n])
    #
    u_tmp = np.zeros([n+2,n+2])
    u_tmp[1:n+1,1:n+1] = u
    u = u_tmp.copy()
        
    midpt = (n+1)/2
    
    if stencil == "5point":
        lapl_midpt = (u[midpt+1,midpt] + u[midpt-1,midpt] - 4*u[midpt,midpt]\
                      + u[midpt,midpt + 1] + u[midpt,midpt -1])/h**2
    if stencil == "9point":
        lapl_midpt = (-u[midpt,midpt-2] + 16 *(u[midpt,midpt-1] + u[midpt,midpt+1]\
                      + u[midpt-1,midpt] + u[midpt+1,midpt])- 60. * u[midpt,midpt]\
                      - u[midpt,midpt+2] - u[midpt-2,midpt] - u[midpt+2,midpt])/(12.*h**2)
    err = abs(2. - lapl_midpt)
#------------------------------------------------------------------------------------------
    """
    This section of the solver determines what output will be displayed to the user. If the
    'printing' switch is set to 'ON' then the code will print the Laplacian at the midpoint
    and the error between that value and the true value of 2.0.

    If the plotting switch is set to 'ON' then the program will generate a surface plot of
    u against the spatial coordinates x and y.
    
    """
    if printing == "ON":
        print("Laplacian at mid point = {}, Error (absolute) = {}".format(lapl_midpt, err))

    if plotting == 'ON':
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
        
    return u
#------------------------------------------------------------------------------------------
def graphplot(method,stencil,n=11,k=10,p=1,omega=1.0,tol = 1.0e-9,n_its=501,printing="OFF"):
    """
    This function plots a 3D surface plot of u against the spatial coordinates x and y.
    
    """
    u=solve(method,stencil,n,k,p,omega,tol,n_its,printing,plotting="OFF")
    
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
#------------------------------------------------------------------------------------------
def time_solver(method,stencil,n=11,k=10,p=1,omega=1.0,tol = 1.0e-9,n_its=501,printing="OFF",plotting="OFF"):
    """
    This function calculates the time taken to solve the PDE with any of the above methods.
    
    """    
    t_start = time.time()
    solve(method,stencil,n,k,p,omega,tol,n_its,printing)
    t_end = time.time()
    time_lapsed = t_end - t_start
    print("Time taken to solve PDE: {}s".format(time_lapsed))

    return time_lapsed
#------------------------------------------------------------------------------------------    
       

if __name__ == '__main__':
    """
    In the main function the solver, timer and plotter can all be called. Below are the
    checks that the Laplacian is correct for each question.
    
    """
    
    print("Q1: inbuilt solver")
    u1 = solve("inbuilt","5point",n=31,plotting="ON")  
    
    print("\nQ2: SOR solver")
    u2 = solve("SOR","5point",n=31)
    
    print("\nQ3: higher order stencil")
    u3 = solve("SOR","9point",n=31)
    
    print("\nQ4: red-black solver")
    u4 = solve("redblack","5point",n=5)
    
    

#------------------------------------------------------------------------------------------

##    graphplot("inbuilt","5point",n=51)
##    time_solver("redblack","5point",n=71)    

##    for i in [3,11,21,31,41,51,61]:
##        print("For n = {}\n".format(i))
##        t_sor = time_solver("SOR","5point",n=i)
##        t_redblack = time_solver("redblack","5point",n=i)
##        print("the sor solver took {}s".format(t_sor))
##        print("the direct solver took {}s\n".format(t_redblack))


    


