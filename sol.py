from gurobipy import Model, GRB, quicksum
import pandas as pd
import os
os.environ['GRB_LICENSE_FILE'] = r"/Users/albert/gurobi/gurobi.lic"
# 讀取 CSV
df = pd.read_csv("availability_4d_binary.csv")

# 部門編號對應
dept_map = {"AC": 1, "GPDA": 2, "PR": 3, "DM": 4}
T = {1: 15, 2: 20, 3: 15, 4: 25}  # duration for each department

# S_k 每日 slot 數（根據題目）
S_k = {
    1: list(range(1, 11)),
    2: list(range(1, 37)),
    3: list(range(1, 29)),
    4: list(range(1, 29)),
    5: list(range(1, 11)),
    6: list(range(1, 11)),
    7: list(range(1, 13)),
    8: list(range(1, 13)),
}

# 集合
I = df["ID"].unique().tolist()
J = [1, 2, 3, 4]
K = [1, 2, 3, 4, 5, 6, 7, 8]

# 可用 O[i,j,k,s]
O = {}
for _, row in df.iterrows():
    i = row["ID"]
    j = dept_map[row["dept"]]
    k = row["k"]
    s = row["s"]
    if row["available"] == 1:
        O[i, j, k, s] = 1

# 建立 A[i,j]：有填過任何 O[i,j,k,s]=1 就設為 1
A = {}
for (i, j, k, s) in O:
    A[i, j] = 1

# 對於沒出現在 O 的 i,j 組合，預設 A[i,j]=0（在 constraints 時用 get）

# 建立模型
model = Model("InterviewScheduling")
model.setParam("OutputFlag", 1)

# 建立變數 x[i,j,k,s]
x = model.addVars(O.keys(), vtype=GRB.BINARY, name="x")

# 目標：最大化總面試數
model.setObjective(quicksum(x[i, j, k, s] for (i, j, k, s) in x), GRB.MAXIMIZE)

# 限制條件 1：每個時段部門僅能安排一人
for j in J:
    for k in K:
        for s in S_k[k]:
            model.addConstr(quicksum(x[i, j, k, s]
                             for i in I if (i, j, k, s) in x) <= 1)

# 限制條件 2：面試僅能安排在可面時段
for key in x.keys():
    model.addConstr(x[key] <= 1)  # already filtered via O

# 限制條件 3：每位面試者只能面試他有申請的部門一次（y_ij 對應為是否面試 j 部門）
for i in I:
    for j in J:
        model.addConstr(
            quicksum(x[i, j, k, s] for k in K for s in S_k[k] if (i, j, k, s) in x) <= A.get((i, j), 0)
        )
# 求解
model.optimize()

# 輸出排程結果
for v in model.getVars():
    if v.X > 0.5:
        print(v.VarName, "= 1")

print("Optimal number of interviews scheduled:", model.ObjVal)