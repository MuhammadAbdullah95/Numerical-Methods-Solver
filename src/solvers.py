# solvers.py
import numpy as np

def bisection_method(f, a, b, tol=1e-6, max_iter=100, tol_mode="Absolute"):
    try:
        fa, fb = float(f(a)), float(f(b))
    except Exception:
        return None, []

    if np.isnan(fa) or np.isnan(fb):
        return None, []

    if fa * fb > 0:
        return None, []

    iterations = []
    for i in range(1, max_iter + 1):
        c = (a + b) / 2.0
        try:
            fc = float(f(c))
        except Exception:
            return None, iterations
        iterations.append([i, float(a), float(b), float(c), float(fc), abs(b - a)])
        current_error = abs(b - a) if tol_mode == "Absolute" else abs(b - a) / abs(c) if c != 0 else np.inf
        
        if abs(fc) < tol or current_error < tol:
            return c, iterations

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    return (a + b) / 2.0, iterations

def fixed_point_iteration(g, x0, tol=1e-6, max_iter=100, tol_mode="Absolute"):
    try:
        x_prev = float(x0)
        iterations = [[0, x_prev, np.real(float(g(x_prev)))]]
        
        for i in range(1, max_iter + 1):
            x_curr = np.real(float(g(x_prev)))
            error = abs(x_curr - x_prev)
            iterations.append([i, x_curr, np.real(float(g(x_curr))), error])
            current_error = error if tol_mode == "Absolute" else error / abs(x_curr) if x_curr != 0 else np.inf
            
            if current_error < tol:
                return x_curr, iterations
            
            x_prev = x_curr
            
    except Exception:
        return None, []
        
    return x_prev, iterations

def auto_detect_intervals(f, start=-10, end=10, step=1.0):
    intervals = []
    xs = np.arange(start, end, step)
    for i in range(len(xs) - 1):
        a, b = float(xs[i]), float(xs[i+1])
        try:
            fa, fb = float(f(a)), float(f(b))
            if np.isnan(fa) or np.isnan(fb):
                continue
            if fa * fb < 0:
                intervals.append((a, b))
        except Exception:
            continue
    return intervals