#應該要完全按照hack md上的constraints
import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
import os
os.environ['GRB_LICENSE_FILE'] = r"/Users/albert/gurobi/gurobi.lic"
# === Step 1: Read and format data ===
df = pd.read_csv("availability_4d_binary.csv")

dept_mapping = {'AC': 0, 'GPDA': 1, 'PR': 2, 'DM': 3}
ids = sorted(df['ID'].unique())
depts = sorted(df['dept'].unique())
k_values = sorted(df['k'].unique())
s_values = sorted(df['s'].unique())

# Index mappings
id_to_idx = {id_: i for i, id_ in enumerate(ids)}
dept_to_idx = {dept: j for j, dept in enumerate(depts)}
k_to_idx = {k: kk for kk, k in enumerate(k_values)}
s_to_idx = {s: ss for ss, s in enumerate(s_values)}
idx_to_id = {i: id_ for id_, i in id_to_idx.items()}
idx_to_dept = {j: dept for dept, j in dept_to_idx.items()}
idx_to_k = {kk: k for k, kk in k_to_idx.items()}
idx_to_s = {ss: s for s, ss in s_to_idx.items()}

# Dimensions
num_ids = len(ids)
num_depts = len(depts)
num_k = len(k_values)
num_s = len(s_values)

# Availability O[i,j,k,s]
O = np.zeros((num_ids, num_depts, num_k, num_s), dtype=int)
for _, row in df.iterrows():
    i = id_to_idx[row['ID']]
    j = dept_to_idx[row['dept']]
    k = k_to_idx[row['k']]
    s = s_to_idx[row['s']]
    O[i, j, k, s] = row['available']


# A[i,j] = 1 if applicant i applied for department j
A = {(i, j): 1 for (i, j, k, s), available in np.ndenumerate(O) if available == 1}

# Interview durations in minutes
T = {0: 15, 1: 20, 2: 15, 3: 25}
dur_slots = {j: int(np.ceil(T[j] / 15)) for j in T}  # Use 15-minute base slot

# Gurobi model
model = Model("InterviewScheduling")
model.setParam("OutputFlag", 1)

x = model.addVars(num_ids, num_depts, num_k, num_s, vtype=GRB.CONTINUOUS, name="x")
y = model.addVars(num_ids, num_depts, num_k, vtype=GRB.CONTINUOUS, name="y")
z = model.addVars(num_ids, num_depts, num_depts, num_k, vtype=GRB.CONTINUOUS, name="z")

# Objective
model.setObjective(quicksum(x[i,j,k,s] for i in range(num_ids)
                                         for j in range(num_depts)
                                         for k in range(num_k)
                                         for s in range(num_s)), GRB.MAXIMIZE)

# Constraint 1: Each applicant can have at most two interviews
for i in range(num_ids):
    for j in range(num_depts):
        model.addConstr(
            quicksum(x[i,j,k,s] for k in range(num_k) for s in range(num_s)) <= A.get((i,j), 0), name = f"con1_{i}_{j}"
        )

# Constraint 2: Only one interviewee per sub-slot per department
for j in range(num_depts):
    for k in range(num_k):
        for s in range(num_s):
            model.addConstr(quicksum(x[i,j,k,s] for i in range(num_ids)) <= 1, name=f"con2_{j}_{k}_{s}")

# Constraint 3: Within available time
for i in range(num_ids):
    for j in range(num_depts):
        for k in range(num_k):
            for s in range(num_s):
                model.addConstr(x[i,j,k,s] <= O[i,j,k,s], name=f"con3_{i}_{j}_{k}_{s}")
'''
# Constraint 4: Interviews must occupy the correct number of time units
for i in range(num_ids):
    for j in range(num_depts):
        d = dur_slots[j]
        for k in range(num_k):
            for s in range(num_s - d + 1):
                model.addConstr(
                    y[i,j,k] <= quicksum(x[i,j,k,s+t] for t in range(d)) - d + 1
                , name=f"con4_{i}_{j}_{k}_{s}")
'''

# Constraint 5: Avoid overlap on the same day
for i in range(num_ids):
    for j in range(num_depts):
        model.addConstr(quicksum(y[i, j, k] for k in range(num_k)) <= 1, name=f"con5_{i}_{j}")

for i in range(num_ids):
    for j in range(num_depts):
        for jp in range(num_depts):
            if j == jp: continue
            for k in range(num_k):
                model.addConstr(z[i, j, jp, k] <= y[i, j, k], name=f"con6_{i}_{j}_{jp}_{k}")
                model.addConstr(z[i, jp, j, k] <= y[i, jp, k], name=f"con7_{i}_{jp}_{j}_{k}")

for i in range(num_ids):
    for j in range(num_depts):
        for jp in range(num_depts):
            if j == jp: continue
            for k in range(num_k):
                model.addConstr(
                    y[i, j, k] + y[i, jp, k] <= z[i, j, jp, k] + z[i, jp, j, k] + 1
                , name=f"con8_{i}_{j}_{jp}_{k}")

M = 10000  # 足夠大的數字（一天分鐘數）

for i in range(num_ids):
    for j in range(num_depts):
        for jp in range(num_depts):
            if j == jp:
                continue
            for k in range(num_k):
                for s in range(num_s):
                    for sp in range(num_s):
                        start_j = s * T[j]
                        end_j = (s + 1) * T[j]

                        start_jp = sp * T[jp]
                        end_jp = (sp + 1) * T[jp]

                        # 如果結束早於開始，才有意義設這條 constraint
                        if end_j <= start_jp:
                            model.addConstr(
                                x[i, j, k, s] * end_j - x[i, jp, k, sp] * start_jp
                                <= M * (1 - z[i, j, jp, k]), name=f"con9_{i}_{j}_{jp}_{k}_{s}_{sp}"
                            )


model.optimize()



# Output solution
if model.status == GRB.OPTIMAL:
    results = []
    for i in range(num_ids):
        for j in range(num_depts):
            for k in range(num_k):
                for s in range(num_s):
                    if x[i,j,k,s].X > 0.5:
                        results.append({
                            "ID": idx_to_id[i],
                            "Department": idx_to_dept[j],
                            "Day": idx_to_k[k],
                            "Slot": idx_to_s[s]
                        })
    pd.DataFrame(results).to_csv("scheduled_interviews.csv", index=False)
    print("✅ Solution written to scheduled_interviews.csv")
else:
    print("❌ Model is infeasible or not optimal")
    model.computeIIS()
    model.write("infeasible.ilp")
