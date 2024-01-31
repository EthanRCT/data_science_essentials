# iPyParallel - Intro to Parallel Programming
from ipyparallel import Client
import numpy as np
import time
import matplotlib.pyplot as plt


# Problem 1
def prob1():
    """
    Write a function that initializes a Client object, creates a Direct
    View with all available engines, and imports scipy.sparse as sparse on
    all engines. Return the DirectView.
    """
    client = Client() # Start the client
    dv = client[:] # Create a DirectView with all available engines
    dv.block = True # Set blocking to True
    dv.execute("import scipy.sparse as sparse") # Import scipy.sparse on all engines
    client.close()
    return dv # Return the DirectView

# Problem 2
def variables(dx):
    """
    Write a function variables(dx) that accepts a dictionary of variables. Create
    a Client object and a DirectView and distribute the variables. Pull the variables back and
    make sure they haven't changed. Remember to include blocking.
    """
    client = Client() # Start the client
    dv = client[:] # Create a DirectView with all available engines
    dv.block = True # Set blocking to True

    dv.push(dx) # Distribute the variables
    
    for key in dx.keys():
        
        engine_vals = dv.pull(key) # Pull the variable from all engines
    
        for val in engine_vals:
            if val != dx[key]:
                raise ValueError(f"Variable {key} has changed")

    client.close()
    return dv # Return the DirectView

# Problem 3
def prob3(n=1000000):
    """
    Write a function that accepts an integer n.
    Instruct each engine to make n draws from the standard normal
    distribution, then hand back the mean, minimum, and maximum draws
    to the client. Return the results in three lists.
    
    Parameters:
        n (int): number of draws to make
        
    Returns:
        means (list of float): the mean draws of each engine
        mins (list of float): the minimum draws of each engine
        maxs (list of float): the maximum draws of each engine.
    """
    means, mins, maxs = [], [], [] # Initialize lists to store results

    client = Client() # Start the client
    dv = client[:] # Create a DirectView with all available engines
    dv.block = True # Set blocking to True

    dv.execute("import numpy as np") # Import numpy on all engines
    dv.execute(f"draws = np.random.normal(size={n})") # Make n draws
    dv.execute("mean = np.mean(draws)") # Calculate the mean
    dv.execute("min = np.min(draws)") # Calculate the minimum
    dv.execute("max = np.max(draws)") # Calculate the maximum

    means = dv.gather("mean") # Gather the means
    mins = dv.gather("min") # Gather the minimums
    maxs = dv.gather("max") # Gather the maximums
    client.close()
    return means, mins, maxs # Return the results

# Problem 4
def prob4():
    """
    Time the process from the previous problem in parallel and serially for
    n = 1000000, 5000000, 10000000, and 15000000. To time in parallel, use
    your function from problem 3 . To time the process serially, run the drawing
    function in a for loop N times, where N is the number of engines on your machine.
    Plot the execution times against n.
    """
    # Get number of cores
    client = Client()
    num_cores = client.ids
    client.close()

    # Initialize lists to store times
    parallel = []
    serial = []
    ns = [1000000, 5000000, 10000000, 15000000]

    for n in ns:
        # Parallel
        start = time.time()
        prob3(n)
        end = time.time()
        parallel.append(end - start)

        # Serial
        start = time.time()
        for _ in range(len(num_cores)):
            draws = np.random.normal(size=n)
        end = time.time()
        serial.append(end - start)
    client.close(client)
    # Plot
    plt.plot(ns, parallel, label="Parallel")
    plt.plot(ns, serial, label="Serial")
    plt.xlabel("n")
    plt.ylabel("Time (s)")
    plt.title("Parallel vs Serial Computing Times")
    plt.legend()
    plt.show()

# Problem 5
def parallel_trapezoidal_rule(f, a, b, n=200):
    """
    Write a function that accepts a function handle, f, bounds of integration,
    a and b, and a number of points to use, n. Split the interval of
    integration among all available processors and use the trapezoidal
    rule to numerically evaluate the integral over the interval [a,b].

    Parameters:
        f (function handle): the function to evaluate
        a (float): the lower bound of integration
        b (float): the upper bound of integration
        n (int): the number of points to use; defaults to 200
    Returns:
        value (float): the approximate integral calculated by the
            trapezoidal rule
    """
    # Start the client
    client = Client()
    
    # Create a DirectView with all available engines
    dv = client[:]
    dv.block = True
    
    # Calculate list of numbers
    points = np.linspace(a, b, n)
       
    # Send in values
    dv.push({'f':f, 'a':a, 'b':b, 'n':n})
    dv.scatter("points", points[:-1])

    # Calculate the integral for each engine
    dv.execute("value = 0")
    dv.execute("for i in range(len(points)): value += (f(points[i]) + f(points[i] + ((b-a)/(n-1)) ))")
    dv.execute("value *= (b-a)/(2*(n-1))")
    results = sum(dv.gather("value"))

    client.close()
    # Gather the results and return the sum
    return results

if __name__ == '__main__':
    # Problem 1
    # dv = prob1()
    # print("Problem 1:")
    # print(dv)

    # Problem 2
    # dx = {'a': 10, 'b': 5, 'c': 2}
    # dv = variables(dx)
    # print("Problem 2:")
    # print(dv)

    # Problem 3
    # means, mins, maxs = prob3()
    # print("Problem 3:")
    # print(f"Means: {means}")
    # print(f"Mins: {mins}")
    # print(f"Maxs: {maxs}")

    # Problem 4
    #prob4()

    # Problem 5
    # a = 0
    # b = 1
    # f = lambda x: x
    # n = 200

    # print(parallel_trapezoidal_rule(f, a, b, n))
    pass
