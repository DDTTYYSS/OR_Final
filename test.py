import os
import sys
import generateInstance2
from heuristic_OriginData import InterviewSchedulerOptimized
from datetime import datetime

def main():
    # 準備輸出檔案
    results_file = "results.txt"
    summary_stats = []
    
    # 重定向輸出到檔案和控制台
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        
        def write(self, text):
            for file in self.files:
                file.write(text)
                file.flush()
        
        def flush(self):
            for file in self.files:
                file.flush()
    
    with open(results_file, 'w', encoding='utf-8') as f:
        # 設置同時輸出到檔案和控制台
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(original_stdout, f)
        
        try:
            print(f"GIS Taiwan Interview Scheduling Test Results")
            print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            print()
            
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