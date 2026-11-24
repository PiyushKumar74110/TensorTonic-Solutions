def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    # Write code here

    

    x_prev = x0


    while steps :
        
        df_dx = 2*a*x_prev + b

        x_next = x_prev - lr * df_dx

        x_prev = x_next

        steps = steps-1
        
    
        

    
    return x_prev