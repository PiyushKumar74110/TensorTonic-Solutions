import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    sum = 0
    for num in x :
        sum += num

    mean = sum/len(x)

    sq_diff = 0

    for num in x :
        sq_diff += (num - mean)**2

    var = sq_diff/(len(x)-1)

    stddev = np.sqrt(var)
        
        

    ans_dict = {
        "variance" : var,
        "standard_deviation" : float(stddev)
    }

    return ans_dict