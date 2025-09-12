import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import uuid
import re

# --- Setup ---
DATA_FILE = "bisection_history.json"
GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

def load_history():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(DATA_FILE, "w") as f:
        json.dump(history, f, indent=4)

def save_graph(fig, uid):
    path = os.path.join(GRAPH_DIR, f"{uid}.png")
    fig.savefig(path)
    plt.close(fig)
    return path

# --- Bisection Method ---
def bisection_method(f, a, b, tol=1e-6, max_iter=100):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return None, []

    iterations = []
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        iterations.append([i, float(a), float(b), float(c), float(fc), abs(b - a)])

        if abs(fc) < tol or abs(b - a) < tol:
            return c, iterations

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return c, iterations

def plot_function(f, a, b, root=None):
    x_vals = np.linspace(a - 1, b + 1, 400)
    y_vals = [f(x) for x in x_vals]

    fig, ax = plt.subplots()
    ax.axhline(0, color="black", linewidth=1)
    ax.plot(x_vals, y_vals, label="f(x)")
    ax.scatter([a, b], [f(a), f(b)], color="red", label="Interval")
    if root:
        ax.scatter(root, f(root), color="green", label="Root")
    ax.legend()
    ax.set_title("Bisection Method Visualization")
    return fig

def auto_detect_intervals(f, start=-10, end=10, step=1):
    """Scan range to find intervals where function changes sign."""
    intervals = []
    x_vals = np.arange(start, end, step)
    for i in range(len(x_vals) - 1):
        a, b = x_vals[i], x_vals[i + 1]
        try:
            fa, fb = f(a), f(b)
            if fa * fb < 0:  # sign change → root exists
                intervals.append((a, b))
        except Exception:
            continue
    return intervals

# --- Streamlit UI ---
st.set_page_config(page_title="Bisection Method Solver", layout="wide")
st.title("🔢 Bisection Method Solver")

# Sidebar history
history = load_history()
st.sidebar.header("📚 History")

# Decimal places (global setting)
decimals = st.number_input("Decimal places:", min_value=2, max_value=12, value=6, step=1)
fmt = f"{{:.{decimals}f}}"

if history:
    for item in history:
        if st.sidebar.button(item["function"], key=item["id"]):
            st.subheader(f"History: {item['function']}")
            st.write(f"**Interval:** [{fmt.format(item['a'])}, {fmt.format(item['b'])}]")
            st.write(f"**Root:** {fmt.format(item['root'])}")
            
            df = pd.DataFrame(item["iterations"], 
                              columns=["Iter", "a", "b", "c", "f(c)", "Error"])
            st.dataframe(df.style.format(fmt))

            st.image(item["graph"])
else:
    st.sidebar.info("No history yet.")

# --- Input ---
raw_func_str = st.text_input("Enter function f(x):", "x**3 - x - 2")

# Preprocess input
func_str = raw_func_str.replace(" ", "")
func_str = func_str.replace("sinx", "sin(x)")
func_str = func_str.replace("cosx", "cos(x)")
func_str = func_str.replace("tanx", "tan(x)")
func_str = func_str.replace("logx", "log(x)")
func_str = func_str.replace("ex", "exp(x)")

# Handle implicit multiplication like 2x → 2*x
func_str = re.sub(r"(\d)(x)", r"\1*\2", func_str)
func_str = re.sub(r"(x)(\d)", r"\1*\2", func_str)

# Mode selection
mode = st.radio("Choose Interval Mode:", ["Manual", "Auto-Detect"])

if mode == "Manual":
    a = st.number_input("Enter interval start (a):", value=1.0)
    b = st.number_input("Enter interval end (b):", value=2.0)
else:
    start_range = st.number_input("Auto-detect start:", value=-10.0)
    end_range = st.number_input("Auto-detect end:", value=10.0)
    step = st.number_input("Step size:", value=1.0)

tol = st.number_input("Tolerance:", value=1e-6, format="%.1e")
max_iter = st.slider("Max Iterations:", 10, 200, 50)

if st.button("Run Bisection"):
    try:
        x = sp.symbols("x")
        f = sp.lambdify(x, sp.sympify(func_str), "numpy")

        if mode == "Manual":
            intervals = [(a, b)]
        else:
            intervals = auto_detect_intervals(f, start_range, end_range, step)
            if not intervals:
                st.error("⚠️ No sign-changing intervals found. Try expanding the range or reducing step size.")
        
        for interval in intervals:
            a, b = interval
            root, iterations = bisection_method(f, a, b, tol, max_iter)

            if root is None:
                st.error(f"No root found in interval [{a}, {b}]")
            else:
                st.success(f"✅ Root found in [{a}, {b}]: {fmt.format(root)}")

                df = pd.DataFrame(iterations, columns=["Iter", "a", "b", "c", "f(c)", "Error"])
                st.dataframe(df.style.format(fmt))

                fig = plot_function(f, a, b, root)
                st.pyplot(fig)

                # Save history (store raw floats)
                uid = str(uuid.uuid4())
                graph_path = save_graph(fig, uid)
                new_item = {
                    "id": uid,
                    "function": raw_func_str,
                    "a": float(a),
                    "b": float(b),
                    "root": float(root),
                    "iterations": iterations,
                    "graph": graph_path
                }
                history.append(new_item)
                save_history(history)
    except Exception as e:
        st.error(f"Error parsing function: {e}")
