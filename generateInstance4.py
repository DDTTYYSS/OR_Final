import random
import pandas as pd

def time_str(t):
    h = int(t)
    m = int(round((t - h) * 60))
    return f"{h:02d}:{m:02d}"

def generate_day_slots(date, is_weekend):
    # 週末三段，平日兩段，每段長度1.5或2小時，且不重疊
    slots = []
    if is_weekend:
        n_slots = 3
        start = 10.0
        end = 17.0
    else:
        n_slots = 2
        start = 18.0
        end = 22.0  # 你可以調整平日總長度
    # 產生所有可能的組合（每段1.5或2小時，總長度不能超過end-start）
    from itertools import product
    possible_lens = [1.5, 2]
    all_combos = [combo for combo in product(possible_lens, repeat=n_slots) if abs(sum(combo) - (end - start)) < 1e-6 or sum(combo) <= (end - start)]
    if not all_combos:
        # 若無法剛好填滿，則允許總長度小於最大長度
        all_combos = [combo for combo in product(possible_lens, repeat=n_slots) if sum(combo) <= (end - start)]
    slot_lens = random.choice(all_combos)
    cur = start
    for l in slot_lens:
        slot_start = cur
        slot_end = cur + l
        slots.append((slot_start, slot_end))
        cur = slot_end
    return slots

def generate_scenario(applicant_count, avg_slots, dept_count, scenario_id):
    names = [f"Applicant_{i+1}" for i in range(applicant_count)]
    depts = [f"Dept_{i+1}" for i in range(dept_count)]
    dates = [f"2024-09-{str(d).zfill(2)}" for d in range(19, 27)]
    date_labels = [
        "Thu, 9/19", "Fri, 9/20", "Sat, 9/21", "Sun, 9/22",
        "Mon, 9/23", "Tue, 9/24", "Wed, 9/25", "Thu, 9/26"
    ]
    weekend = {"2024-09-21", "2024-09-22"}

    # 產生部門面試時長
    dept_duration = {dept: random.choice([10, 15, 20, 25]) for dept in depts}
    pd.DataFrame(
        [{"dept": dept, "duration": duration} for dept, duration in dept_duration.items()]
    ).to_csv(f"scenario_{scenario_id}_dept_duration.csv", index=False)

    # 產生每天的時段與部門可用性
    avail_rows = []
    day_slot_dict = {}  # {date: [(start, end), ...]}
    for date, label in zip(dates, date_labels):
        is_weekend = date in weekend
        slots = generate_day_slots(date, is_weekend)
        day_slot_dict[date] = slots
        for slot_start, slot_end in slots:
            row = {
                "Date": label,
                "Time Slot": f"{time_str(slot_start)}-{time_str(slot_end)}"
            }
            for dept in depts:
                # 70% 機率 available
                row[dept] = "O" if random.random() < 0.5 else "X"
            avail_rows.append(row)
    pd.DataFrame(avail_rows).to_csv(f"scenario_{scenario_id}_dept_available_time.csv", index=False)

    # 產生面試者可用時段（依照部門 available slot 隨機分配）
    records = []
    for i, name in enumerate(names):
        id = i + 1
        allowed_dept_count = max(1, dept_count // 2)
        allowed_depts = random.sample(depts, allowed_dept_count)
        slot_count = max(1, int(random.gauss(avg_slots, 4)))
        for _ in range(slot_count):
            dept = random.choice(allowed_depts)
            # 隨機選一天有 available 的 slot
            possible = []
            for date in dates:
                for slot_start, slot_end in day_slot_dict[date]:
                    # 查這個部門這個 slot 是否 available
                    label = date_labels[dates.index(date)]
                    slot_str = f"{time_str(slot_start)}-{time_str(slot_end)}"
                    row = next((r for r in avail_rows if r["Date"] == label and r["Time Slot"] == slot_str), None)
                    if row and row[dept] == "O":
                        possible.append((date, slot_str))
            if possible:
                date, slot_str = random.choice(possible)
                records.append([id, name, dept, date, slot_str])
    df = pd.DataFrame(records, columns=["ID", "Name", "dept", "date", "time_slot"])
    df.to_csv(f"scenario_{scenario_id}.csv", index=False)

# Scenario 設定
scenarios = [
    (100, 4, 4),   # 1
    (50, 4, 4),   # 2
    (200, 4, 4),  # 3
    (100, 2, 4),   # 4
    (100, 8, 4),   # 5
    (100, 2, 2),   # 6
    (100, 8, 8),   # 7
]

if __name__ == "__main__":
    for idx, (a, s, d) in enumerate(scenarios, 1):
        generate_scenario(a, s, d, idx)
