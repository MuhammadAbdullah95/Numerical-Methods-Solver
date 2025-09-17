# main.py
import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import uuid
import re
from datetime import datetime

# -------------------------
# Setup paths and SymPy x
# -------------------------
DATA_FILE = "bisection_history.json"
GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

x = sp.symbols("x")

# -------------------------
# Helpers: history & graphs
# -------------------------
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
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

# -------------------------
# Numerical routines
# -------------------------
def bisection_method(f, a, b, tol=1e-6, max_iter=100, tol_mode="Absolute"):
    """Return (root or None, iterations list). iterations = [i,a,b,c,f(c),interval_length]"""
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

    # return last midpoint if loop ends
    return (a + b) / 2.0, iterations

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

# -------------------------
# Plot helpers
# -------------------------
def plot_function(f, a, b, root=None):
    x_vals = np.linspace(a - 1, b + 1, 400)
    # safe evaluate
    y_vals = []
    for xv in x_vals:
        try:
            yv = float(f(xv))
        except Exception:
            yv = np.nan
        y_vals.append(yv)

    fig, ax = plt.subplots()
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(x_vals, y_vals, label="f(x)")
    # endpoints (safe)
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
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.set_title("Bisection Method Visualization")
    fig.tight_layout()
    return fig

def animate_bisection_steps(f, iters):
    if not iters:
        return

    # Determine a reasonable range for plotting
    all_vals = [i[1] for i in iters] + [i[2] for i in iters]
    min_x = min(all_vals)
    max_x = max(all_vals)
    x_range = max_x - min_x
    padding = x_range * 0.1
    x_vals = np.linspace(min_x - padding, max_x + padding, 400)
    
    y_vals = [float(f(val)) if not np.isnan(float(f(val))) else np.nan for val in x_vals]

    figs = []
    for i, step_data in enumerate(iters):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(step_data[3], color="blue", linestyle='--', label=f'Midpoint $c_{i+1}$')
        
        # Plot the function
        ax.plot(x_vals, y_vals, label="f(x)")
        
        # Highlight the current interval
        ax.fill_between([step_data[1], step_data[2]], ax.get_ylim()[0], ax.get_ylim()[1], color='gray', alpha=0.3, label=f'Interval $[a_{i+1}, b_{i+1}]$')
        
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.set_title(f"Bisection Method - Iteration {i+1}")
        fig.tight_layout()
        figs.append(fig)

    return figs

def generate_report(data):
    report_content = f"""
# Bisection Method Report

**Function:** {data.get('raw','')}
**Interval:** [{data['a']:.6f}, {data['b']:.6f}]
**Date & Time:** {data.get('timestamp','')}

---

### Results

**Root Found:** {data['root']:.6f}
**Number of Iterations:** {len(data['iterations'])}
**Final Error Bound:** {data['iterations'][-1][5] if data['iterations'] else 'N/A'}

---

### Iteration Table

| Iter | a | b | c | f(c) | Error Bound |
|------|---|---|---|---|---|
"""
    
    for i in data['iterations']:
        report_content += f"| {i[0]} | {i[1]:.6f} | {i[2]:.6f} | {i[3]:.6f} | {i[4]:.6f} | {i[5]:.6f} |\n"
        
    return report_content

# -------------------------
# UI / Main
# -------------------------
st.set_page_config(page_title="Bisection Method Solver", layout="wide")
st.title("🔢 Bisection Method Solver — Step-by-Step + History")

# load history
history = load_history()

# Global controls
decimals = st.number_input("Decimal places:", min_value=2, max_value=12, value=6, step=1)
fmt = f"{{:.{decimals}f}}"

# Sidebar — history display & clear
st.sidebar.header("📚 Saved Problems")
if history:
    for item in history:
        root_disp = item.get("root")
        root_label = fmt.format(root_disp) if root_disp is not None else "—"
        label = f"`{item.get('raw','')}` → root: {root_label} ({item.get('timestamp','')})"
        if st.sidebar.button(label, key=item["id"]):
            st.subheader(f"History: {item.get('raw','')}")
            st.write(f"**Interval:** [{fmt.format(item['a'])}, {fmt.format(item['b'])}]")
            st.write(f"**Root:** {root_label}")
            df = pd.DataFrame(item.get("iterations", []), columns=["Iter","a","b","c","f(c)","Error"])
            if not df.empty:
                st.dataframe(df.style.format(fmt))
            if os.path.exists(item.get("graph","")):
                st.image(item["graph"])
else:
    st.sidebar.info("No saved problems yet.")

if st.sidebar.button("🗑️ Clear History"):
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    if os.path.exists(GRAPH_DIR):
        for f in os.listdir(GRAPH_DIR):
            try:
                os.remove(os.path.join(GRAPH_DIR, f))
            except Exception:
                pass
    st.rerun()

# Input & preprocessing
raw_func_str = st.text_input("Enter function f(x):", "x**3 - x - 2")

proc = raw_func_str.replace(" ", "")
proc = proc.replace("^", "**")
proc = proc.replace("sinx", "sin(x)")
proc = proc.replace("cosx", "cos(x)")
proc = proc.replace("tanx", "tan(x)")
proc = proc.replace("logx", "log(x)")
proc = proc.replace("ex", "exp(x)")
proc = re.sub(r"(\d)(x)", r"\1*\2", proc)
proc = re.sub(r"(x)(\d)", r"\1*\2", proc)

# interval mode
mode = st.radio("Interval mode:", ["Manual", "Auto-detect"], horizontal=True)
if mode == "Manual":
    a = st.number_input("Interval start a", value=1.0, format="%.6f")
    b = st.number_input("Interval end b", value=2.0, format="%.6f")
else:
    start_range = st.number_input("Scan start", value=-10.0, format="%.6f")
    end_range = st.number_input("Scan end", value=10.0, format="%.6f")
    step = st.number_input("Scan step", value=1.0, format="%.6f")

# Tolerance modes
tol_mode = st.radio("Tolerance mode:", ["Absolute", "Relative"], horizontal=True)
tol = st.number_input("Tolerance:", value=1e-6, format="%.1e")

max_iter = st.slider("Max iterations", min_value=10, max_value=500, value=100)
step_mode = st.checkbox("Enable Step-by-Step Mode", value=False)

# Run button
if st.button("Run Bisection", key="run"):
    # compile expression safely
    try:
        expr = sp.sympify(proc)
        f_callable = sp.lambdify(x, expr, "numpy")
    except Exception as e:
        st.error(f"Failed to parse function. Try valid Python/Sympy syntax. Error: {e}")
        st.stop()

    if mode == "Manual":
        intervals = [(float(a), float(b))]
    else:
        intervals = auto_detect_intervals(f_callable, float(start_range), float(end_range), float(step))
        if not intervals:
            st.error("No sign-changing intervals found. Try expanding range or reducing step size.")
            intervals = []

    results = []
    for ia, ib in intervals:
        root, iterations = bisection_method(f_callable, ia, ib, tol, max_iter, tol_mode)
        results.append({
            "a": float(ia),
            "b": float(ib),
            "root": (float(root) if root is not None else None),
            "iterations": iterations,
            "expr": proc,
            "raw": raw_func_str
        })

    st.session_state.bisection_results = results
    st.session_state.step_mode_active = bool(step_mode)
    st.session_state.selected_result_index = 0
    st.session_state.step_index = 0

    hist = load_history()
    for res in results:
        if res["root"] is None:
            continue
        try:
            f_for_graph = sp.lambdify(x, sp.sympify(res["expr"]), "numpy")
            fig = plot_function(f_for_graph, res["a"], res["b"], res["root"])
            gid = str(uuid.uuid4())
            graph_path = save_graph(fig, gid)
            hist.append({
                "id": gid,
                "raw": res["raw"],
                "expr": res["expr"],
                "a": res["a"],
                "b": res["b"],
                "root": res["root"],
                "iterations": res["iterations"],
                "graph": graph_path,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        except Exception:
            hist.append({
                "id": str(uuid.uuid4()),
                "raw": res["raw"],
                "expr": res["expr"],
                "a": res["a"],
                "b": res["b"],
                "root": res["root"],
                "iterations": res["iterations"],
                "graph": "",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    save_history(hist)
    st.success("Computation finished. Scroll down for results / Step-by-Step controls.")

# -------------
# Show results
# -------------
if st.session_state.get("bisection_results"):
    results = st.session_state.bisection_results
    st.markdown("---")

    # Symbolic Analysis Section
    st.header("Symbolic Analysis with SymPy")
    try:
        expr = sp.sympify(results[0]["expr"])
        
        st.subheader("Derivative")
        derivative = sp.diff(expr, x)
        st.write(f"The derivative of $f(x) = {sp.latex(expr)}$ is $f'(x) = {sp.latex(derivative)}$")
        
        st.subheader("Exact Roots")
        roots_symbolic = sp.solve(expr, x)
        if roots_symbolic:
            st.write("The exact symbolic roots are:")
            for root_sym in roots_symbolic:
                st.write(f"- ${sp.latex(root_sym)}$")
        else:
            st.info("SymPy could not find a simple symbolic root.")
    except Exception as e:
        st.error(f"Symbolic analysis failed: {e}")
    st.markdown("---")


    # Non-step (show full)
    if not st.session_state.get("step_mode_active", False):
        for res in results:
            st.markdown("---")
            if res["root"] is None:
                st.warning(f"No root found in interval [{res['a']}, {res['b']}]")
                continue
            st.subheader(f"Result for interval [{fmt.format(res['a'])}, {fmt.format(res['b'])}]")
            st.metric("Root", fmt.format(res["root"]))
            
            # Confidence Report
            report_data = {
                "raw": res['raw'],
                "a": res['a'],
                "b": res['b'],
                "root": res['root'],
                "iterations": res['iterations'],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            report_content = generate_report(report_data)
            st.subheader("Confidence Report")
            st.code(report_content, language="markdown")
            st.download_button(
                label="Download Report",
                data=report_content,
                file_name="bisection_report.txt",
                mime="text/plain"
            )
            
            st.subheader("Full Iteration Details")
            df = pd.DataFrame(res["iterations"], columns=["Iter","a","b","c","f(c)","Error"])
            st.dataframe(df.style.format(fmt))
            try:
                f_for_graph = sp.lambdify(x, sp.sympify(res["expr"]), "numpy")
                fig = plot_function(f_for_graph, res["a"], res["b"], res["root"])
                st.pyplot(fig)
            except Exception:
                pass

    # Step-by-step interactive mode
    else:
        st.markdown("## ▶ Step-by-step interactive mode")
        valid_indices = [i for i, r in enumerate(results) if r.get("root") is not None]
        if not valid_indices:
            st.warning("No intervals with roots to step through.")
        else:
            def fmt_interval(i):
                r = results[i]
                return f"[{fmt.format(r['a'])}, {fmt.format(r['b'])}]  root: {fmt.format(r['root'])}"
            sel = st.selectbox("Choose interval", options=valid_indices, format_func=fmt_interval, key="select_interval")
            st.session_state.selected_result_index = sel

            curr = results[st.session_state.selected_result_index]
            iters = curr.get("iterations", [])
            if not iters:
                st.warning("No iterations available for this interval.")
            else:
                si = st.session_state.get("step_index", 0)
                si = max(0, min(si, len(iters) - 1))
                st.session_state.step_index = si
                step_data = iters[si]  # [i,a,b,c,fc,error]

                st.markdown(f"### Iteration {step_data[0]}")
                st.write(f"a = {fmt.format(step_data[1])}  b = {fmt.format(step_data[2])}")
                st.write(f"c = {fmt.format(step_data[3])}  f(c) = {fmt.format(step_data[4])}")
                st.write(f"Interval length (error bound) = {fmt.format(step_data[5])}")

                try:
                    f_for_plot = sp.lambdify(x, sp.sympify(curr["expr"]), "numpy")
                    figs = animate_bisection_steps(f_for_plot, iters)
                    if figs:
                        st.pyplot(figs[si])
                except Exception as e:
                    st.error(f"Plot failed: {e}")

                col1, col2, col3 = st.columns([1,1,1])
                with col1:
                    if st.button("⏮ Previous", key="prev_step") and st.session_state.step_index > 0:
                        st.session_state.step_index -= 1
                        st.rerun()
                with col2:
                    if st.button("Next ▶", key="next_step") and st.session_state.step_index < len(iters) - 1:
                        st.session_state.step_index += 1
                        st.rerun()
                with col3:
                    if st.button("Finish / Clear Step Mode", key="finish_step"):
                        for k in ["bisection_results", "step_mode_active", "selected_result_index", "step_index"]:
                            if k in st.session_state:
                                del st.session_state[k]
                        st.rerun()

else:
    st.info("Enter function and interval (or auto-detect) and click Run Bisection. Enable Step-by-Step to step through iterations.")