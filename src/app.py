# app.py
import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import uuid
from datetime import datetime

from solvers import bisection_method, fixed_point_iteration, auto_detect_intervals
from utils import (
    load_history,
    save_run,
    generate_report,
    preprocess,
    clear_history_files,
    save_graph
)
from plots import plot_function, plot_fixed_point_cobweb

# -------------------------
# Setup paths and SymPy x
# -------------------------
x = sp.symbols("x")

# -------------------------
# Initialize Session State
# -------------------------
if 'bisection_results' not in st.session_state:
    st.session_state.bisection_results = None
if 'fixed_point_results' not in st.session_state:
    st.session_state.fixed_point_results = None
if 'step_mode_active' not in st.session_state:
    st.session_state.step_mode_active = False
if 'selected_result_index' not in st.session_state:
    st.session_state.selected_result_index = 0
if 'step_index' not in st.session_state:
    st.session_state.step_index = 0

# -------------------------
# UI / Main
# -------------------------
st.set_page_config(page_title="Numerical Method Solver", layout="wide")
st.title("🔢 Numerical Method Solver — Compare + History")

# Global controls
decimals = st.number_input("Decimal places:", min_value=2, max_value=12, value=6, step=1)
fmt = f"{{:.{decimals}f}}"

# Sidebar — history display & clear
st.sidebar.header("📚 Saved Problems")
history = load_history()
if history:
    for item in history:
        # Check if the item is a dictionary before proceeding
        if isinstance(item, dict):
            root_disp = item.get("root")
            root_label = fmt.format(root_disp) if root_disp is not None else "—"
            label = f"`{item.get('raw','')}` → root: {root_label} ({item.get('method','')})"
            if st.sidebar.button(label, key=item["id"]):
                st.subheader(f"History: {item.get('raw','')}")
                st.write(f"**Method:** {item.get('method','N/A')}")
                if item.get("a") is not None and item.get("b") is not None:
                     st.write(f"**Interval:** [{fmt.format(item['a'])}, {fmt.format(item['b'])}]")
                if item.get("x0") is not None:
                    st.write(f"**Initial Guess:** {fmt.format(item['x0'])}")
                st.write(f"**Root:** {root_label}")
                if 'comparison_results' in item:
                    st.subheader("Comparison Results")
                    df_bisection = pd.DataFrame(item["comparison_results"]["bisection"]["iterations"], columns=item["comparison_results"]["bisection"]["cols"])
                    st.write("Bisection Method:")
                    st.dataframe(df_bisection.style.format(fmt))
                    df_fixed_point = pd.DataFrame(item["comparison_results"]["fixed_point"]["iterations"], columns=item["comparison_results"]["fixed_point"]["cols"])
                    st.write("Fixed-Point Iteration:")
                    st.dataframe(df_fixed_point.style.format(fmt))
                else:
                    df = pd.DataFrame(item.get("iterations", []), columns=item.get("cols", []))
                    if not df.empty:
                        st.dataframe(df.style.format(fmt))
                if item.get("graph"):
                    st.image(item["graph"])
else:
    st.sidebar.info("No saved problems yet.")

if st.sidebar.button("🗑️ Clear History"):
    clear_history_files()
    st.rerun()

# --- Main UI Controls ---
method = st.radio("Choose Method:", ["Bisection Method", "Fixed-Point Iteration"], horizontal=True)
compare_mode = st.checkbox("Compare Both Methods", value=False)

raw_func_str = st.text_input("Enter function f(x):", "x**3 - x - 2")
if method == "Fixed-Point Iteration" or compare_mode:
    raw_g_str = st.text_input("Enter iteration function g(x):", "x**(1/3) + 2/x**2")
    st.info("The fixed-point method solves $g(x) = x$. You must provide a valid $g(x)$ for your $f(x)=0$ problem. For $x^3-x-2=0$, a good choice is $g(x)=(x+2)^{1/3}$.")

# Dynamic Inputs
if method == "Bisection Method" and not compare_mode:
    mode = st.radio("Interval mode:", ["Manual", "Auto-detect"], horizontal=True)
    if mode == "Manual":
        a = st.number_input("Interval start a", value=1.0, format="%.6f")
        b = st.number_input("Interval end b", value=2.0, format="%.6f")
    else:
        start_range = st.number_input("Scan start", value=-10.0, format="%.6f")
        end_range = st.number_input("Scan end", value=10.0, format="%.6f")
        step = st.number_input("Scan step", value=1.0, format="%.6f")
elif method == "Fixed-Point Iteration" and not compare_mode:
    x0 = st.number_input("Initial Guess x₀", value=1.5, format="%.6f")
elif compare_mode:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bisection Method Inputs")
        mode = st.radio("Interval mode:", ["Manual", "Auto-detect"], horizontal=True, key="comp_bisect_mode")
        if mode == "Manual":
            a = st.number_input("Interval start a", value=1.0, format="%.6f", key="comp_bisect_a")
            b = st.number_input("Interval end b", value=2.0, format="%.6f", key="comp_bisect_b")
        else:
            start_range = st.number_input("Scan start", value=-10.0, format="%.6f", key="comp_bisect_start")
            end_range = st.number_input("Scan end", value=10.0, format="%.6f", key="comp_bisect_end")
            step = st.number_input("Scan step", value=1.0, format="%.6f", key="comp_bisect_step")
    with col2:
        st.subheader("Fixed-Point Iteration Inputs")
        x0 = st.number_input("Initial Guess x₀", value=1.5, format="%.6f", key="comp_fp_x0")

tol_mode = st.radio("Tolerance mode:", ["Absolute", "Relative"], horizontal=True)
tol = st.number_input("Tolerance:", value=1e-6, format="%.1e")
max_iter = st.slider("Max iterations", min_value=10, max_value=500, value=100)
step_mode = st.checkbox("Enable Step-by-Step Mode", value=False)

if st.button("Run Solver", key="run"):
    st.session_state.bisection_results = None
    st.session_state.fixed_point_results = None
    
    proc_f = preprocess(raw_func_str)
    try:
        f_expr = sp.sympify(proc_f)
        f_callable = sp.lambdify(x, f_expr, "numpy")
    except Exception as e:
        st.error(f"Failed to parse f(x) function. Error: {e}")
        st.stop()

    if method == "Bisection Method" or compare_mode:
        if 'mode' in locals() and mode == 'Manual':
            intervals = [(float(a), float(b))]
        else:
            intervals = auto_detect_intervals(f_callable, float(start_range), float(end_range), float(step))
        
        bisection_results = []
        for ia, ib in intervals:
            root, iters = bisection_method(f_callable, ia, ib, tol, max_iter, tol_mode)
            bisection_results.append({
                "a": float(ia), "b": float(ib), "root": root, "iterations": iters,
                "expr": proc_f, "raw": raw_func_str, "method": "Bisection Method",
                "cols": ["Iter", "a", "b", "c", "f(c)", "Error"]
            })
        st.session_state.bisection_results = bisection_results

    if method == "Fixed-Point Iteration" or compare_mode:
        proc_g = preprocess(raw_g_str)
        try:
            g_expr = sp.sympify(proc_g)
            g_callable = sp.lambdify(x, g_expr, "numpy")
        except Exception as e:
            st.error(f"Failed to parse g(x) function. Error: {e}")
            st.stop()
            
        fp_root, fp_iters = fixed_point_iteration(g_callable, float(x0), tol, max_iter, tol_mode)
        fixed_point_results = [{
            "x0": float(x0), "root": fp_root, "iterations": fp_iters,
            "expr_f": proc_f, "raw_f": raw_func_str,
            "expr_g": proc_g, "raw_g": raw_g_str,
            "method": "Fixed-Point Iteration",
            "cols": ["Iter", "x_i", "g(x_i)", "Error"]
        }]
        st.session_state.fixed_point_results = fixed_point_results
    
    st.success("Computation finished. Scroll down for results.")

    # Save to history after computation
    if st.session_state.bisection_results or st.session_state.fixed_point_results:
        history_entry = {
            "id": str(uuid.uuid4()),
            "raw": raw_func_str,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if compare_mode:
            history_entry["method"] = "Comparison"
            history_entry["a"] = st.session_state.bisection_results[0].get('a')
            history_entry["b"] = st.session_state.bisection_results[0].get('b')
            history_entry["x0"] = st.session_state.fixed_point_results[0].get('x0')
            history_entry["root"] = st.session_state.bisection_results[0].get('root')
            history_entry["comparison_results"] = {
                "bisection": st.session_state.bisection_results[0],
                "fixed_point": st.session_state.fixed_point_results[0]
            }
        elif method == "Bisection Method":
            history_entry.update(st.session_state.bisection_results[0])
        elif method == "Fixed-Point Iteration":
            history_entry.update(st.session_state.fixed_point_results[0])

        fig = None
        if compare_mode:
            # We don't save a single graph to history in comparison mode anymore
            # We'll display both separately
            pass
        elif method == "Bisection Method":
            try:
                res = st.session_state.bisection_results[0]
                f_for_graph = sp.lambdify(x, sp.sympify(res["expr"]), "numpy")
                fig = plot_function(f_for_graph, res["a"], res["b"], res["root"])
            except Exception as e:
                st.error(f"Bisection plot saving failed: {e}")
        elif method == "Fixed-Point Iteration":
            try:
                res = st.session_state.fixed_point_results[0]
                g_expr = sp.sympify(res["expr_g"])
                g_callable = sp.lambdify(x, g_expr, "numpy")
                all_x = [i[1] for i in res["iterations"]]
                x_range = (min(all_x) - 1, max(all_x) + 1) if all_x else (-2, 2)
                fig = plot_fixed_point_cobweb(g_callable, x_range, res['iterations'])
            except Exception as e:
                st.error(f"Fixed-Point plot saving failed: {e}")

        if fig:
            history_entry["graph"] = save_graph(fig, history_entry["id"])
        else:
            history_entry["graph"] = ""
        
        save_run(history_entry)

# -------------------------
# Display Results
# -------------------------
if st.session_state.get("bisection_results") or st.session_state.get("fixed_point_results"):
    st.markdown("---")
    st.header("Results")

    if compare_mode:
        st.markdown("## 📊 Comparison of Methods")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Bisection Method Results")
            if st.session_state.bisection_results:
                res = st.session_state.bisection_results[0]
                if res["root"] is None:
                    st.warning("Bisection Method: No root found.")
                else:
                    st.metric("Root", fmt.format(res["root"]))
                    st.metric("Iterations", len(res["iterations"]))
                    df = pd.DataFrame(res["iterations"], columns=res["cols"])
                    st.dataframe(df.style.format(fmt))

        with col2:
            st.markdown("### Fixed-Point Iteration Results")
            if st.session_state.fixed_point_results:
                res = st.session_state.fixed_point_results[0]
                if res["root"] is None:
                    st.warning("Fixed-Point Iteration: No root found or did not converge.")
                else:
                    st.metric("Root", fmt.format(res["root"]))
                    st.metric("Iterations", len(res["iterations"]))
                    fp_cols = res["cols"]
                    df_data = res["iterations"]
                    if df_data and len(df_data[0]) < len(fp_cols):
                        df_data[0].append(np.nan)
                    df = pd.DataFrame(df_data, columns=fp_cols)
                    st.dataframe(df.style.format(fmt))
        
        st.markdown("---")
        
        st.header("Graphical Comparison")
        col1, col2 = st.columns(2)
        with col1:
            try:
                st.subheader("Bisection Method Graph")
                b_res = st.session_state.bisection_results[0]
                f_expr = sp.sympify(b_res["expr"])
                f_callable = sp.lambdify(x, f_expr, "numpy")
                fig = plot_function(f_callable, b_res['a'], b_res['b'], b_res['root'])
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Bisection plot failed: {e}")
        with col2:
            try:
                st.subheader("Fixed-Point Iteration Graph")
                fp_res = st.session_state.fixed_point_results[0]
                g_expr = sp.sympify(fp_res["expr_g"])
                g_callable = sp.lambdify(x, g_expr, "numpy")
                all_x = [i[1] for i in fp_res["iterations"]]
                x_range = (min(all_x) - 1, max(all_x) + 1) if all_x else (-2, 2)
                fig = plot_fixed_point_cobweb(g_callable, x_range, fp_res['iterations'])
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Fixed-Point plot failed: {e}")

    elif method == "Bisection Method":
        results = st.session_state.bisection_results
        if not st.session_state.get("step_mode_active", False) and results:
            for res in results:
                st.markdown("---")
                if res["root"] is None:
                    st.warning(f"No root found in interval [{res['a']}, {res['b']}]")
                    continue
                st.subheader(f"Result for interval [{fmt.format(res['a'])}, {fmt.format(res['b'])}]")
                st.metric("Root", fmt.format(res["root"]))
                report_content = generate_report(res, "Bisection Method")
                st.subheader("Confidence Report")
                st.code(report_content, language="markdown")
                st.download_button(
                    label="Download Report", data=report_content, file_name="bisection_report.txt", mime="text/plain"
                )
                st.subheader("Full Iteration Details")
                df = pd.DataFrame(res["iterations"], columns=res["cols"])
                st.dataframe(df.style.format(fmt))
                try:
                    f_for_graph = sp.lambdify(x, sp.sympify(res["expr"]), "numpy")
                    fig = plot_function(f_for_graph, res["a"], res["b"], res["root"])
                    st.pyplot(fig)
                except Exception:
                    pass
        elif st.session_state.get("step_mode_active", False) and results:
            # Step-by-step logic here
            pass
    elif method == "Fixed-Point Iteration":
        results = st.session_state.fixed_point_results
        if not st.session_state.get("step_mode_active", False) and results:
            res = results[0]
            st.markdown("---")
            if res["root"] is None:
                st.warning("No root found or did not converge.")
            else:
                st.subheader("Results for Fixed-Point Iteration")
                st.metric("Root", fmt.format(res["root"]))
                report_content = generate_report(res, "Fixed-Point Iteration")
                st.subheader("Confidence Report")
                st.code(report_content, language="markdown")
                st.download_button(
                    label="Download Report", data=report_content, file_name="fixed_point_report.txt", mime="text/plain"
                )
                st.subheader("Full Iteration Details")
                df_data = res["iterations"]
                if df_data and len(df_data[0]) < len(res["cols"]):
                    df_data[0].append(np.nan)
                df = pd.DataFrame(df_data, columns=res["cols"])
                st.dataframe(df.style.format(fmt))
                try:
                    g_expr = sp.sympify(res["expr_g"])
                    g_callable = sp.lambdify(x, g_expr, "numpy")
                    all_x = [i[1] for i in res["iterations"]]
                    x_range = (min(all_x) - 1, max(all_x) + 1) if all_x else (-2, 2)
                    fig = plot_fixed_point_cobweb(g_callable, x_range, res['iterations'])
                    if fig:
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Fixed-Point plot failed: {e}")
        elif st.session_state.get("step_mode_active", False) and results:
            # Step-by-step logic here
            pass
else:
    st.info("Enter function and other parameters, then click 'Run Solver' to get started.")