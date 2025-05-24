import random
import pandas as pd

def generate_scenario(applicant_count, avg_slots, dept_count, scenario_id):
    names = [f"Applicant_{i+1}" for i in range(applicant_count)]
    depts = [f"Dept_{i+1}" for i in range(dept_count)]
    dates = [f"2024-09-{str(d).zfill(2)}" for d in range(19, 27)]
    records = []
    for i, name in enumerate(names):
        id = i + 1
        slot_count = max(1, int(random.gauss(avg_slots, 1)))
        for _ in range(slot_count):
            dept = random.choice(depts)
            date = random.choice(dates)
            start_hour = random.choice(range(9, 20))
            duration = random.choice([1, 2])
            end_hour = min(start_hour + duration, 21)
            time_slot = f"{start_hour:02d}:00-{end_hour:02d}:00"
            records.append([id, name, dept, date, time_slot])
    df = pd.DataFrame(records, columns=["ID", "Name", "dept", "date", "time_slot"])
    df.to_csv(f"scenario_{scenario_id}.csv", index=False)

# Scenario 設定
scenarios = [
    (50, 4, 4),   # 1
    (20, 4, 4),   # 2
    (100, 4, 4),  # 3
    (50, 2, 4),   # 4
    (50, 8, 4),   # 5
    (50, 4, 2),   # 6
    (50, 4, 8),   # 7
]

for idx, (a, s, d) in enumerate(scenarios, 1):
    generate_scenario(a, s, d, idx)