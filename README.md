# PyGradDescent

PyGradDescent is a Python package that provides an implementation of the gradient descent algorithm for optimizing unconstrained and constrained functions. The package includes support for adaptive restart, acceleration, and projection onto the feasible set, which can help improve the convergence and robustness of the optimization process. PyGradDescent can be used for a variety of applications, such as parameter estimation, machine learning, and data analysis.

## Installation

To use this project, you will need the following dependencies installed on your system:

- `numpy`: A library for numerical computations in Python
- `autograd`: A library for automatic differentiation in Python

To install these dependencies, you can use the following commands in your terminal:
```
pip install numpy
pip install autograd
```

Once you have installed these dependencies, you can download the `PyGradDescent.py` file and place it in the same directory as your Python program.

Alternatively, you can clone this repository to your local machine using the following command:

```
git clone https://github.com/LouisCourrian/PyGradDescent.git
```

Then, navigate to the project directory and place the `PyGradDescent.py` file in the same directory as your Python program.

Finally, import the `PyGradDescent` module in your Python program using the following line:

```python
from PyGradDescent import gradient_descent
```
And that's it! You should now be able to use the gradient_descent function in your code. 


## Usage

Here's an example usage of `gradient_descent` to minimize the Rosenbrock function:

```python
import autograd.numpy as np
from PyGradDescent import gradient_descent

# Define the Rosenbrock function
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Set up the optimization parameters
init = [0, 0] # Starting point for optimization
alpha = 0.001 # Learning rate
epochs = 1000 # Number of iterations

# Define the projection function (optional)
def project(x):
    return np.clip(x, -5, 5)

# Run gradient descent to minimize the Rosenbrock function
params, history = gradient_descent(rosenbrock, init, alpha, epochs, project)

# Print the minimum value found and the corresponding point
print(f"Minimum value: {history[-1]:.4f}")
print(f"Minimum point: {params[-1]}")
```

In this example, we define the Rosenbrock function and set up the optimization parameters (initialization, learning rate, and number of iterations). We also define an optional projection function to keep the optimization within a certain range. We then call gradient_descent with the Rosenbrock function, the optimization parameters, and the projection function (if desired). Finally, we print the minimum value found and the corresponding point.

## Function Documentation



### gradient_descent

```
gradient_descent(cost_fun, init, alpha, epochs, project=lambda w: w)
```

Arguments:
- `cost_fun`: Function to minimize. Callable that takes an array and returns a scalar.
- `init`: Initialization. Array of the shape expected by `cost_fun`.
- `alpha`: Step size. Scalar.
- `epochs`: Number of iterations. Integer.
- `project`: Projection onto the feasible set. Callable that takes an array and returns an array.

Returns:
- `w_list`: List of parameters found by gradient descent.
- `history`: List of function evaluations.

## License

This project is licensed under the MIT License




