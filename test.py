import os
import sys
import generateInstance4
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
            for idx, (a, s, d) in enumerate(generateInstance4.scenarios, 1):
                generateInstance4.generate_scenario(a, s, d, idx)
            print("All scenario instances generated.\n")

            # 針對每個 scenario 檔案執行 heuristic algorithm
            for idx in range(1, len(generateInstance4.scenarios) + 1):
                csv_path = f"scenario_{idx}.csv"
                dept_duration_path = f"scenario_{idx}_dept_duration.csv"
                print(f"{'='*50}")
                print(f"SCENARIO {idx}: Running heuristic on {csv_path}")
                print(f"{'='*50}")
                
                scheduler = InterviewSchedulerOptimized(csv_path, dept_duration_path)
                scheduler.solve_heuristic()
                
                # 獲取統計資料
                total_scheduled = len(scheduler.schedule)
                unique_applicant_dept_combinations = set()
                for (applicant, dept, date_k) in scheduler.available_times.keys():
                    unique_applicant_dept_combinations.add((applicant, dept))
                max_possible = len(unique_applicant_dept_combinations)
                
                # 儲存統計資料
                summary_stats.append({
                    'scenario': idx,
                    'total_scheduled': total_scheduled,
                    'max_possible': max_possible,
                    'success_rate': (total_scheduled / max_possible * 100) if max_possible > 0 else 0
                })
                
                scheduler.analyze_results()
                result_path = f"scenario_{idx}_result.csv"
                scheduler.export_schedule(result_path)
                print(f"Result exported to {result_path}\n")
            
            # 輸出總結統計
            print("\n" + "=" * 80)
            print("SUMMARY STATISTICS")
            print("=" * 80)
            print(f"{'Scenario':<10} {'Scheduled':<12} {'Max Possible':<14} {'Success Rate':<12}")
            print("-" * 50)
            
            total_scheduled_all = 0
            total_max_possible_all = 0
            
            for stats in summary_stats:
                print(f"{stats['scenario']:<10} {stats['total_scheduled']:<12} {stats['max_possible']:<14} {stats['success_rate']:<11.1f}%")
                total_scheduled_all += stats['total_scheduled']
                total_max_possible_all += stats['max_possible']
            
            print("-" * 50)
            overall_success_rate = (total_scheduled_all / total_max_possible_all * 100) if total_max_possible_all > 0 else 0
            print(f"{'TOTAL':<10} {total_scheduled_all:<12} {total_max_possible_all:<14} {overall_success_rate:<11.1f}%")
            
            print(f"\nOverall Performance:")
            print(f"- Total interviews scheduled across all scenarios: {total_scheduled_all}")
            print(f"- Total possible interviews across all scenarios: {total_max_possible_all}")
            print(f"- Overall success rate: {overall_success_rate:.1f}%")
            
        finally:
            # 恢復原來的stdout
            sys.stdout = original_stdout
    
    print(f"\nAll results have been saved to {results_file}")

if __name__ == "__main__":
    main()
