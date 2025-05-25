import random
import pandas as pd

def generate_scenario(applicant_count, avg_slots, dept_count, scenario_id):
    names = [f"Applicant_{i+1}" for i in range(applicant_count)]
    depts = [f"Dept_{i+1}" for i in range(dept_count)]
    dates = [f"2024-09-{str(d).zfill(2)}" for d in range(19, 27)]
    # 假設 2024-09-21, 2024-09-22 是週末
    weekend = {"2024-09-21", "2024-09-22"}
    dept_duration = {dept: random.choice([10, 15, 20, 25]) for dept in depts}
    pd.DataFrame(
        [{"dept": dept, "duration": duration} for dept, duration in dept_duration.items()]
    ).to_csv(f"scenario_{scenario_id}_dept_duration.csv", index=False)
    dept_time_pool = {}
    for dept in depts:
        dept_time_pool[dept] = []
        for date in dates:
            if date in weekend:
                min_h, max_h = 6, 8  # 假日長一點
            else:
                min_h, max_h = 2, 4
            slot_length = random.choice([i * 0.5 for i in range(int(min_h*2), int(max_h*2)+1)])  # 1.5~4, step 0.5
            slot_length = round(slot_length, 1)
            # 隨機起始時間（保證不超過 21:00）
            possible_starts = [h + m for h in range(9, 21) for m in [0, 0.5] if h + m + slot_length <= 21]
            start = random.choice(possible_starts)
            end = start + slot_length
            # 轉成時間字串
            def time_str(t):
                h = int(t)
                m = int((t - h) * 60)
                return f"{h:02d}:{m:02d}"
            time_slot = f"{time_str(start)}-{time_str(end)}"
            dept_time_pool[dept].append((date, time_slot))

    
    # 彙整每個部門每天的 available time slot
    dept_avail_records = []
    for dept in depts:
        for date in dates:
            slots = [slot for d, slot in dept_time_pool[dept] if d == date]
            if slots:
                dept_avail_records.append({
                    "dept": dept,
                    "date": date,
                    "available_slots": ",".join(slots)
                })

    # 輸出成 CSV
    pd.DataFrame(dept_avail_records).to_csv(f"scenario_{scenario_id}_dept_available_time.csv", index=False)


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
    (100, 4, 4),   # 1
    (50, 4, 4),   # 2
    (200, 4, 4),  # 3
    (100, 2, 4),   # 4
    (100, 8, 4),   # 5
    (100, 4, 2),   # 6
    (100, 4, 8),   # 7
]

if __name__ == "__main__":
    for idx, (a, s, d) in enumerate(scenarios, 1):
        generate_scenario(a, s, d, idx)