"""
Page to visualize the solution of a linear program with a 1D or 2D objective function
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable
import streamlit as st

st.title("Lagrangian Relaxer")

# Defined radio for LP or X set
type_of_lp = st.radio("Type of LP", ["Knapsack", "LP", "Assignment", "Uncapacitated Facility Location"])

if type_of_lp == "Knapsack":
    st.header("Input Parameters")

    b_col, n_col, u_col = st.columns(3)
    b = b_col.number_input("Capacity (b)", min_value=0, value=68)
    n = n_col.number_input("Number of items (n)", min_value=0, value=10, step=1)
    u_input = u_col.number_input("Lagrangian relaxation (u)", min_value=0.0, value=0.0)

    # Assuming profits and weights are fixed
    # Alternatively, you could allow users to input these as well
    ps = st.text_input("Profits (p)", value="100, 80, 77, 90, 70, 80, 4, 8, 20, 4")
    ps = np.array([int(p) for p in ps.split(",")])
    ws = st.text_input("Weights (w)", value="10, 10, 11, 15, 12, 20, 2, 8, 40, 16")
    ws = np.array([int(w) for w in ws.split(",")])
    # find lagrangian relaxation

    # Define the problem
    model = LpProblem(name="knapsack-problem", sense=LpMaximize)

    # Define the decision variables
    for i in range(n):
        globals()["x" + str(i)] = LpVariable(name="x" + str(i), lowBound=0, upBound=1, cat="Binary")
        model += globals()["x" + str(i)]

    # Add the constraints to the model
    # model += lpSum([ws[i] * globals()["x" + str(i)] for i in range(n)]) <= b

    # Add the objective function to the model
    model += lpSum([ps[i] * globals()["x" + str(i)] for i in range(n)]) + u_input * (b - lpSum([ws[i] * globals()["x" + str(i)] for i in range(n)]))

    # Write out symbolic lagraingian relaxation term
    st.text(
        f"Lagrangian Term: {u_input} * ({b} - {lpSum([ws[i] * globals()['x' + str(i)] for i in range(n)])})"
    )

    st.text(model)
    
    # Button to solve the problem
    if st.button("Solve"):
        # Solve the problem
        status = model.solve()

        # Output the results
        st.write(f"Status: {model.status}, {LpStatus[model.status]}")
        st.write(f"Objective: {model.objective.value()}")
        st.write(f"u: {u_input}")

        # insert table of results
        st.write("Results")
        df = pd.DataFrame(
            [[var.name, var.value()] for var in model.variables()],
            columns=["Variable", "Value"],
        )
        st.write(df.T)

elif type_of_lp == "Assignment":
    "General Assignment Problem"
    st.write("Firstly, the GAP is defined as follows:")
    st.latex(r"""
        \begin{aligned}
        & \mathrm{Z}=\min \sum_{i=1}^m \sum_{j=1}^n c_{i j} x_{i j} \\
        & \sum_{i=1}^m x_{i j}=1, \quad j=1, \ldots, n, \\
        & \sum_{j=1}^n a_{i j} x_{i j} \leq b_i, \quad i=1, \ldots, m, \\
        & x_{i j}=0 \text { or } 1, \quad \text { all } i \text { and } j .
        \end{aligned}
             """)
    # Radio for either relaxing the assignment constraints or the capacity constraints
    relax_assignment = st.radio("Relax Assignment or Capacity Constraints?", ["Assignment", "Capacity"])
    if relax_assignment == "Assignment":
        st.write("Relaxing Assignment Constraints we get:")
        st.latex(r"""
        \begin{gathered}
        Z_{D 1}(u)=\min \sum_{i=1}^m \sum_{j=1}^n c_{i j} x_{i j}+\sum_{j=1}^n u_j\left(\sum_{i=1}^m x_{i j}-1\right) \\
        \text { subject to (3) and (4) } \\
        =\min \sum_{i=1}^m \sum_{j=1}^n\left(c_{i j}+u_j\right) x_{i j}-\sum_{j=1}^n u_j \\
        \text { subject to (3) and (4). }
        \end{gathered}
            """)
    
    elif relax_assignment == "Capacity":
        st.write("Relaxing Capacity Constraints we get:")
        st.latex(r"""
        \begin{gathered}
        Z_{D 2}(v)=\min \sum_{i=1}^m \sum_{j=1}^n c_{i j} x_{i j}+\sum_{i=1}^m v_i\left(\sum_{j=1}^n a_{i j} x_{i j}-b_i\right) \\
        \text { subject to (2) and (4) } \\
        =\min \sum_{j=1}^n\left(\sum_{i=1}^m\left(c_{i j}+v_i a_{i j}\right) x_{i j}\right)-\sum_{i=1}^m v_i b_i \\
        \text { subject to (2) and (4). }
        \end{gathered}
        """)

    st.write("Solver is coming...")

elif type_of_lp == "Uncapacitated Facility Location":
    st.write("## General Uncapacitated Facility Location Problem")
    st.latex(r"""
        \begin{aligned}
        & z=\max \sum_{i \in M} \sum_{j \in N} c_{i j} x_{i j}-\sum_{j \in N} f_j y_j \\
        & \text { s.t. } \sum_{j \in N} x_{i j}=1 \quad i \in M \\
        & x_{i j} \leq y_j \quad i \in M, j \in N \\
        & 0 \leq x_{i j} \leq 1 \quad i \in M, j \in N \\
        & y_j \in\{0,1\} \quad j \in N \\
        &
        \end{aligned}
             """)
    st.write("Relaxing the binary constraints on $y_j$ we get:")
    st.latex(r"""
        \begin{gathered}
        Z(u)=\max _{i \in M, j \in N}\left(c_{i j}-u_i\right) y_{i j}-\sum_{j \in N} f_j x_j+\sum_{i \in M} u_i \\
        y_{i j} \leq x_j \text { for } i \in M, j \in N \\
        y \in \mathbb{R}_{+}^{|M| \times|N|}, x \in\{0,1\}^{|N|} .
        \end{gathered}
             """)

    st.write("## Input Parameters")
    m_col, n_col = st.columns(2)
    m = m_col.number_input("Number of facilities (m)", min_value=0, value=3, step=1)
    n = n_col.number_input("Number of customers (n)", min_value=0, value=3, step=1)

    # allow for delivery costs to be input
    c_col = st.columns(m)

    c = []
    for i in range(m-1):
        c.append(c_col[i].text_input(f"Delivery costs for facility {i}", value="10, 10, 11"))
        c[i] = np.array([int(c_i) for c_i in c[i].split(",")])

    # allow for fixed costs to be input
    f_col = st.columns(m)
    f = []
    for i in range(m-1):
        f.append(f_col[i].number_input(f"Fixed costs for facility {i}", min_value=0, value=10, step=1))

    c = pd.DataFrame(c).T

    st.write("**Delivery Costs, $c_{ij}$**")
    UFL_cost = st.dataframe(c)

    dual_vector = st.text_input("Dual Vector", value="0, 0, 0")
    dual_vector = np.array([int(d) for d in dual_vector.split(",")])

    # withdraw the dual vector from the UFL cost matrix columnwise
    UFL_cost_dual = c.sub(dual_vector, axis=0)

    st.write("**Dual Cost, $c_{ij} - u_i$**")
    st.dataframe(UFL_cost_dual)
    st.write("$\sum_{i \in M}u_i=$ ", sum(dual_vector))

    st.write("Write we can then solve by inspection since,")
    st.latex(r"""
        \begin{gathered}
        Z_j(u)=\max \sum_{i \in M}\left(c_{i j}-u_i\right) y_{i j}-f_j x_j \\
        \left.\operatorname{lP}_j(u)\right) \quad \text { for } i \in M \\
        y_{. j} \in \mathbb{R}_{+}^{|M|}, x_j \in\{0,1\} .
        \end{gathered}
        """)

    # solve by inspection
    st.write("Solving by inspection we get:")
    # for each column we use the objective value: z(u) = max{0, sum(max{cij - ui, 0}) - fj}
    z_u = []
    used_facilities = []
    routes = []
    for j in range(n):
        # Use a generator expression to apply max to each item in the column
        column_max = UFL_cost_dual.iloc[:, j].apply(lambda x: max(0, x))
        # get indecies of positive values
        z_u.append(max(0, column_max.sum() - f[j]))
    st.write("The optimal solution is then:")
    st.write("z(u) = ", sum(z_u) + sum(dual_vector))


