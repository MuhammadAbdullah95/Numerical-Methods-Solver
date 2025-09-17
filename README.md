# Numerical Method Solver: Numerical Computing Toolkit
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


An interactive Streamlit-based web application for solving and visualizing **numerical computing methods**. Currently, it includes the **Bisection Method** and **Fixed-Point Iteration**, with plans to add more numerical techniques in the future (e.g., Newton-Raphson, Secant Method, etc.). This toolkit is designed for students, educators, and professionals who want to explore root-finding algorithms and other numerical methods interactively.

---

## ✨ Features

* **Bisection Method**: Solve equations with manual or automatic interval detection.
* **Fixed-Point Iteration**: Find roots using an initial guess and iteration function.
* **Comparison Mode**: Run both methods side-by-side to compare convergence speed and efficiency.
* **Interactive Plots**:

  * Function graph for the Bisection Method.
  * Cobweb plot for Fixed-Point Iteration.
* **Detailed Iteration Tables**: Step-by-step results including values of *a, b, c, f(c),* and error.
* **Problem History**: Automatically saves all solved problems to revisit later.
* **Confidence Reports**: Generate and download detailed reports of results, including iterations and final root.
* **Customizable Parameters**: Control tolerance, max iterations, and decimal precision.
* **Extensible Design**: Modular codebase allows easy integration of additional numerical methods in future.

---

## 📁 File Structure

```
├── src/                        
│   ├── app.py                      # Main Streamlit app (UI + session state)
│   ├── solvers.py                  # Core numerical algorithms (bisection, fixed-point, etc.)
│   ├── utils.py                    # Helper functions (history, reports, file handling)
│   ├── plots.py                    # Visualization functions (function and cobweb plots)
│   ├── __init__.py                 # Makes src a package
├── requirements.txt            # Required dependencies
├── numerical_methods_history.json # Saved problem history
└── graphs/                     # Directory for generated plots
```

---

## 🚀 Setup and Installation

### Option 1: Using pip

1. Clone the repository:

   ```bash
   git clone https://github.com/MuhammadAbdullah95/Numerical-Methods-Solver.git
   cd Numerical-Methods-Solver
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   ```

   * Windows: `venv\Scripts\activate`
   * macOS/Linux: `source venv/bin/activate`

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

### Option 2: Using [uv](https://github.com/astral-sh/uv)

1. Install dependencies directly with uv:

   ```bash
   uv pip install -r requirements.txt
   or 
   uv sync
   ```

2. Run the Streamlit app with uv:

   ```bash
   uv run streamlit run app.py
   ```

The app will open in your default browser.

---

## 💻 Usage

1. **Choose Method**: Select *Bisection* or *Fixed-Point Iteration*.
2. **Compare Mode**: Enable to run both methods simultaneously.
3. **Enter Functions**:

   * *Bisection*: Provide `f(x)` and interval `[a, b]`, or use auto-detect.
   * *Fixed-Point*: Provide iteration function `g(x)` and initial guess `x₀`.
4. **Run Solver**: Click *Run Solver* to see results including graphs, iteration tables, and final root.
5. **Check History**: Past problems are saved in the sidebar for quick reload.

---

## 🛠️ Technologies Used

* **Streamlit**: Interactive web UI.
* **NumPy**: Numerical operations.
* **SymPy**: Symbolic math & parsing user input.
* **Pandas**: Tabular display of iteration data.
* **Matplotlib**: Graphs & visualizations.
* **uv**: Fast Python package manager (alternative to pip).

---

## 👤 Author

**Muhammad Abdullah**
[LinkedIn](https://www.linkedin.com/in/muhammad-abdullah-3a8550255) | [GitHub](https://github.com/MuhammadAbdullah95)
