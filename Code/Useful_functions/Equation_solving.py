def newtons_method(a, b, c, d, guess, tol=1e-7, max_iter=100):
    """Find a real root using Newton's method."""
    for _ in range(max_iter):
        f = a * guess**3 + b * guess**2 + c * guess + d
        f_prime = 3 * a * guess**2 + 2 * b * guess + c

        if abs(f_prime) < tol:  # Avoid division by zero or very small derivative
            return None

        next_guess = guess - f / f_prime

        if abs(next_guess - guess) < tol:  # Check for convergence
            return next_guess

        guess = next_guess

    return None  # If it doesn't converge within max_iter
