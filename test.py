import os
import generateInstance3
from heuristic_OriginData import InterviewSchedulerOptimized

def main():
    # 先產生所有 scenario instance
    print("Generating all scenario instances...")
    for idx, (a, s, d) in enumerate(generateInstance3.scenarios, 1):
        generateInstance3.generate_scenario(a, s, d, idx)
    print("All scenario instances generated.\n")

    # 針對每個 scenario 檔案執行 heuristic algorithm
    for idx in range(1, len(generateInstance3.scenarios) + 1):
        csv_path = f"scenario_{idx}.csv"
        dept_duration_path = f"scenario_{idx}_dept_duration.csv"
        print(f"Running heuristic on {csv_path} ...")
        scheduler = InterviewSchedulerOptimized(csv_path, dept_duration_path)
        scheduler.solve_heuristic()
        scheduler.analyze_results()
        result_path = f"scenario_{idx}_result.csv"
        scheduler.export_schedule(result_path)

if __name__ == "__main__":
    main()