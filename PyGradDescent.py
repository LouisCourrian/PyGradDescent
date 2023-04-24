import autograd.numpy as np
import sys
import time

def gradient_descent(cost_fun, init, alpha, epochs, project=lambda w: w):
    """
    Arguments
    ---------
     cost_fun - Function to minimize             | Callable that takes an array and returns a scalar
         init - Initialization                   | Array of the shape expected by 'cost_fun'
        alpha - Step size                        | Scalar
       epochs - Number of iterations             | Integer
      project - Projection onto the feasible set | Callable that takes an array and returns an array
        
    Returns
    -------
    list of params - Solution found by gradient descent
    history - List of function evaluations
    """
    
    start_time = time.time()
    # Automatic gradient
    from autograd import grad
    gradient = grad(cost_fun)

    # Initialization
    w = np.array(init, dtype=float)
    v = 0
    k = 0
    
    # Iterative refinement
    history = []
    w_list = []
    running_time = None
    for i in range(epochs):
        
        # Projected gradient step
        u = v
        g = gradient(w)
        v = project(w - alpha * g)
        
        # Adaptive restart
        if np.ravel(g) @ np.ravel(v - u) <= 0:
            k = k + 1
        else:
            k = 1

        # Acceleration
        w = v + (k-1)/(k+2) * (v - u)
        w_list.append(w)
        # Bookkeping
        hist = cost_fun(w)
        history += [hist]
        running_time = "{:.2f}".format(time.time() - start_time)
        sys.stdout.write(f'Progress: {i+1}/{epochs}, History: {hist}, running time: {running_time}\r')
        if hist != hist:
            sys.stdout.write(f'ERROR: history = nan\n')
            return w_list, history
        
    sys.stdout.write(f'Progress: {epochs}/{epochs}, Last history: {hist}, Execution time: {running_time}\n')

    return w_list,  history