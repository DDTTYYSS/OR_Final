import random
import pandas as pd

def generate_scenario(applicant_count, avg_slots, dept_count, scenario_id):
    names = [f"Applicant_{i+1}" for i in range(applicant_count)]
    depts = [f"Dept_{i+1}" for i in range(dept_count)]
    dates = [f"2024-09-{str(d).zfill(2)}" for d in range(19, 27)]
    # 隨機產生每個部門的面試時長（10~25分鐘）
    dept_duration = {dept: random.choice([10, 15, 20, 25]) for dept in depts}
    # 輸出部門時長資訊
    pd.DataFrame(
        [{"dept": dept, "duration": duration} for dept, duration in dept_duration.items()]
    ).to_csv(f"scenario_{scenario_id}_dept_duration.csv", index=False)
    # 為每個部門預先產生一組可用時段池
    dept_time_pool = {}
    for dept in depts:
        dept_time_pool[dept] = []
        for date in dates:
            for start_hour in range(9, 20):
                for duration in [1, 2]:
                    end_hour = min(start_hour + duration, 21)
                    if end_hour > start_hour:
                        time_slot = f"{start_hour:02d}:00-{end_hour:02d}:00"
                        dept_time_pool[dept].append((date, time_slot))
    records = []
    for i, name in enumerate(names):
        id = i + 1
        allowed_dept_count = max(1, dept_count // 2)
        allowed_depts = random.sample(depts, allowed_dept_count)
        slot_count = max(1, int(random.gauss(avg_slots, 1)))
        for _ in range(slot_count):
            dept = random.choice(allowed_depts)
            date, time_slot = random.choice(dept_time_pool[dept])
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

if __name__ == "__main__":
    for idx, (a, s, d) in enumerate(scenarios, 1):
        generate_scenario(a, s, d, idx)