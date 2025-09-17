# plots.py
import matplotlib.pyplot as plt
import numpy as np

def plot_function(f, a, b, root=None, label="f(x)"):
    """
    Plots a function and its root/interval.
    """
    x_vals = np.linspace(a - 1, b + 1, 400)
    y_vals = []
    for xv in x_vals:
        try:
            yv = float(f(xv))
        except Exception:
            yv = np.nan
        y_vals.append(yv)

    fig, ax = plt.subplots()
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(x_vals, y_vals, label=label)
    try:
        ax.scatter([a, b], [float(f(a)), float(f(b))], color="red", label="Interval")
    except Exception:
        pass
    if root is not None:
        try:
            ax.scatter(root, float(f(root)), color="green", label="Root")
        except Exception:
            pass
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title(f"{label} Visualization")
    fig.tight_layout()
    return fig

def plot_fixed_point_cobweb(g, x_range, iters):
    """
    Generates a cobweb plot for fixed-point iteration.
    """
    try:
        fig, ax = plt.subplots()
        x_vals = np.linspace(x_range[0], x_range[1], 400)
        y_vals = [np.real(float(g(val))) for val in x_vals]

        ax.plot(x_vals, x_vals, label="y=x", linestyle="--", color="gray")
        ax.plot(x_vals, y_vals, label="g(x)")

        x_points = [i[1] for i in iters]
        y_points = [i[2] for i in iters]
        
        for i in range(len(x_points) - 1):
            ax.plot([x_points[i], x_points[i]], [x_points[i], y_points[i]], color='red', linestyle=':', alpha=0.6)
            ax.plot([x_points[i], x_points[i+1]], [y_points[i], y_points[i]], color='red', linestyle=':', alpha=0.6)

        ax.scatter(x_points, y_points, color="blue", label="Iterations")
        ax.set_title("Fixed-Point Iteration (Cobweb Plot)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        return fig
    except Exception:
        return None