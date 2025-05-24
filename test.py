import os
import generateInstance2
from heuristic_OriginData import InterviewSchedulerOptimized

def main():
    # 先產生所有 scenario instance
    print("Generating all scenario instances...")
    for idx, (a, s, d) in enumerate(generateInstance2.scenarios, 1):
        generateInstance2.generate_scenario(a, s, d, idx)
    print("All scenario instances generated.\n")

    # 針對每個 scenario 檔案執行 heuristic algorithm
    for idx in range(1, len(generateInstance2.scenarios) + 1):
        csv_path = f"scenario_{idx}.csv"
        print(f"Running heuristic on {csv_path} ...")
        scheduler = InterviewSchedulerOptimized(csv_path)
        scheduler.solve_heuristic()
        scheduler.analyze_results()
        result_path = f"scenario_{idx}_result.csv"
        scheduler.export_schedule(result_path)
        print(f"Result exported to {result_path}\n")

if __name__ == "__main__":
    main()