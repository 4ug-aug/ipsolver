import streamlit as st
import numpy as np
import pandas as pd
from pulp import LpMaximize, LpProblem, lpSum, LpVariable, LpMinimize, LpStatus

# Title of the app
st.title("0-1 Knapsack Problem Solver")

# Sidebar for input parameters
st.header("Input Parameters")

b_col, n_col = st.columns(2)
b = b_col.number_input("Capacity (b)", min_value=0, value=68)
n = n_col.number_input("Number of items (n)", min_value=0, value=10, step=1)

# Assuming profits and weights are fixed
# Alternatively, you could allow users to input these as well
ps = st.text_input("Profits (p)", value="100, 80, 77, 90, 70, 80, 4, 8, 20, 4")
ps = np.array([int(p) for p in ps.split(",")])
ws = st.text_input("Weights (w)", value="10, 10, 11, 15, 12, 20, 2, 8, 40, 16")
ws = np.array([int(w) for w in ws.split(",")])

print(ps)
print(ws)

minmax, contint = st.columns(2)
minmax = minmax.radio("Maximize or Minimize", ["Maximize", "Minimize"])
contint = contint.radio("Continuous, Integer or Binary", ["Continuous", "Integer", "Binary"])

if minmax == "Minimize":
    sense = LpMinimize
else:
    sense = LpMaximize

# Define the problem
model = LpProblem(name="knapsack-problem", sense=sense)

# Define the decision variables
for i in range(n):
    globals()["x" + str(i)] = LpVariable(name="x" + str(i), lowBound=0, upBound=1, cat=contint)
    model += globals()["x" + str(i)]

# Add the constraints to the model
model += lpSum([ws[i] * globals()["x" + str(i)] for i in range(n)]) <= b

# Add the objective function to the model
model += lpSum([ps[i] * globals()["x" + str(i)] for i in range(n)])

st.text(model)

# Button to solve the problem
if st.button("Solve"):
    # Solve the problem
    status = model.solve()

    # Output the results
    st.write(f"Status: {model.status}, {LpStatus[model.status]}")
    st.write(f"Objective: {model.objective.value()}")

    # insert table of results
    st.write("Results")
    df = pd.DataFrame(
        [[var.name, var.value()] for var in model.variables()],
        columns=["Variable", "Value"],
    )
    st.write(df.T)
