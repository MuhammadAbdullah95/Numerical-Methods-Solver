# utils.py
import os
import json
import re
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

DATA_FILE = "numerical_methods_history.json"
GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

def load_history():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_history(history):
    with open(DATA_FILE, "w") as f:
        json.dump(history, f, indent=4)

def save_graph(fig, uid):
    path = os.path.join(GRAPH_DIR, f"{uid}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def save_run(data):
    hist = load_history()
    hist.append(data)
    save_history(hist)

def clear_history_files():
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    if os.path.exists(GRAPH_DIR):
        for f in os.listdir(GRAPH_DIR):
            try:
                os.remove(os.path.join(GRAPH_DIR, f))
            except Exception:
                pass

def preprocess(s):
    s = s.replace(" ", "").replace("^", "**")
    s = re.sub(r"(\d)x", r"\1*x", s)
    s = re.sub(r"x(\d)", r"x*\1", s)
    s = s.replace("sinx", "sin(x)").replace("cosx", "cos(x)").replace("tanx", "tan(x)")
    s = s.replace("logx", "log(x)").replace("ex", "exp(x)")
    return s

def generate_report(data, method_name):
    if not data or not data.get('iterations'):
        return f"No data available for {method_name}."

    report_content = f"""
# {method_name} Report

**Function:** {data.get('raw','')}
**Initial Conditions:** {'Interval' if method_name == 'Bisection Method' else 'Initial Guess'}: {f"[{data['a']:.6f}, {data['b']:.6f}]" if method_name == 'Bisection Method' else f"{data['x0']:.6f}"}
**Date & Time:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

### Results

**Root Found:** {data['root']:.6f}
**Number of Iterations:** {len(data['iterations'])}
**Final Error Bound:** {data['iterations'][-1][-1] if data['iterations'] and len(data['iterations'][-1]) > 2 else 'N/A'}

---

### Iteration Table

"""
    if method_name == 'Bisection Method':
        report_content += "| Iter | a | b | c | f(c) | Error Bound |\n"
        report_content += "|------|---|---|---|---|---|\n"
        for i in data['iterations']:
            report_content += f"| {i[0]} | {i[1]:.6f} | {i[2]:.6f} | {i[3]:.6f} | {i[4]:.6f} | {i[5]:.6f} |\n"
    else:
        report_content += "| Iter | x_i | g(x_i) | Error |\n"
        report_content += "|------|---|---|---|\n"
        for i in data['iterations']:
            error_val = f"{i[3]:.6f}" if len(i) > 3 and not np.isnan(i[3]) else "N/A"
            report_content += f"| {i[0]} | {i[1]:.6f} | {i[2]:.6f} | {error_val} |\n"
    return report_content