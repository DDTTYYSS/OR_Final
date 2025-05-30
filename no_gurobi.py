import os
import sys
import pandas as pd
import numpy as np
# from gurobipy import Model, GRB, quicksum
from datetime import datetime
import generateInstance4
from heuristic_OriginData import InterviewSchedulerOptimized
from naive import NaiveInterviewScheduler
import time
# os.environ['GRB_LICENSE_FILE'] = r"/Users/albert/gurobi/gurobi.lic"

class ComparisonSolver:
    def __init__(self):
        self.results_file = "comparison_results.txt"
        self.summary_stats = []
        
        # Given session counts per day
        self.S_sizes = {1: 24, 2: 24, 3: 36, 4: 36, 5: 24, 6: 24, 7: 24, 8: 24}
        
        # Day start times
        self.day_start_map = {
            '2024-09-19': '18:00',
            '2024-09-20': '18:00',
            '2024-09-21': '10:00',
            '2024-09-22': '10:00',
            '2024-09-23': '18:00',
            '2024-09-24': '18:00',
            '2024-09-25': '18:00',
            '2024-09-26': '18:00'
        }
    
    def get_durations(self, dept_duration_path):
        """Read department durations from CSV file"""
        durations_df = pd.read_csv(dept_duration_path)
        return dict(zip(durations_df['dept'], durations_df['duration']))
    
    def process_availability_to_4d(self, csv_path, dept_duration_path):
        """Convert availability data to 4D binary format"""
        # Load the availability CSV
        avail_df = pd.read_csv(csv_path)
        
        # Get department durations
        durations = self.get_durations(dept_duration_path)
        
        # Map dates to day index k (1 to 8) in chronological order
        unique_dates = sorted(avail_df['date'].unique(), key=lambda d: datetime.fromisoformat(d))
        date_to_k = {date: idx+1 for idx, date in enumerate(unique_dates)}
        
        # Initialize availability sets
        I_list = sorted(avail_df['ID'].unique())
        J_list = sorted(avail_df['dept'].unique())
        K_list = list(range(1, len(unique_dates)+1))
        
        # Prepare a dict to collect available s for each (i,j,k)
        avail_sets = {(i, j, k): set() for i in I_list for j in J_list for k in K_list}
        
        # Populate availability based on each record
        for _, row in avail_df.iterrows():
            i = row['ID']
            j = row['dept']
            date = row['date']
            k = date_to_k[date]
            dur = durations[j]
            
            # Parse start and end times
            start_str, end_str = row['time_slot'].split('-')
            day_start = datetime.strptime(self.day_start_map[date], '%H:%M')
            start_ts = datetime.strptime(start_str, '%H:%M')
            end_ts = datetime.strptime(end_str, '%H:%M')
            
            offset_start = int((start_ts - day_start).total_seconds() // 60)
            offset_end = int((end_ts - day_start).total_seconds() // 60)
            
            s_start = offset_start // dur + 1
            s_end = offset_end // dur
            
            # Add to set
            for s in range(s_start, s_end+1):
                if 1 <= s <= self.S_sizes[k]:
                    avail_sets[(i, j, k)].add(s)
        
        # Build full 4D DataFrame
        records = []
        for i in I_list:
            for j in J_list:
                for k in K_list:
                    for s in range(1, self.S_sizes[k] + 1):
                        records.append({
                            'ID': i,
                            'dept': j,
                            'k': k,
                            's': s,
                            'available': 1 if s in avail_sets[(i, j, k)] else 0
                        })
        
        return pd.DataFrame(records)
    
    def run_naive(self, csv_path, dept_duration_path):
        """Run the naive algorithm on a scenario"""
        start_time = time.time()
        scheduler = NaiveInterviewScheduler(csv_path)
        # Set the interview durations from the department duration file
        durations_df = pd.read_csv(dept_duration_path)
        scheduler.interview_duration = dict(zip(durations_df['dept'], durations_df['duration']))
        scheduler.reset_solution()
        total_assigned = scheduler.naive_schedule()
        end_time = time.time()
        return total_assigned, end_time - start_time
    
    def run_heuristic(self, csv_path, dept_duration_path):
        """Run the heuristic algorithm on a scenario"""
        start_time = time.time()
        scheduler = InterviewSchedulerOptimized(csv_path, dept_duration_path)
        scheduler.solve_heuristic()
        end_time = time.time()
        return scheduler, end_time - start_time
    
    # def run_gurobi(self, df_4d, dept_duration_path):
    #     """Run the Gurobi optimization on a scenario using 4D data"""
    #     start_time = time.time()
        
    #     # Setup mappings and dimensions
    #     ids = sorted(df_4d['ID'].unique())
    #     depts = sorted(df_4d['dept'].unique())
    #     k_values = sorted(df_4d['k'].unique())
    #     s_values = sorted(df_4d['s'].unique())
        
    #     # Create index mappings
    #     id_to_idx = {id_: i for i, id_ in enumerate(ids)}
    #     dept_to_idx = {dept: j for j, dept in enumerate(depts)}
    #     k_to_idx = {k: kk for kk, k in enumerate(k_values)}
    #     s_to_idx = {s: ss for ss, s in enumerate(s_values)}
        
    #     # Create availability matrix
    #     num_ids = len(ids)
    #     num_depts = len(depts)
    #     num_k = len(k_values)
    #     num_s = len(s_values)
        
    #     O = np.zeros((num_ids, num_depts, num_k, num_s), dtype=int)
    #     for _, row in df_4d.iterrows():
    #         i = id_to_idx[row['ID']]
    #         j = dept_to_idx[row['dept']]
    #         k = k_to_idx[row['k']]
    #         s = s_to_idx[row['s']]
    #         O[i, j, k, s] = row['available']
        
    #     # A[i,j] = 1 if applicant i applied for department j
    #     A = {(i, j): 1 for (i, j, k, s), available in np.ndenumerate(O) if available == 1}
        
    #     # Read interview durations from CSV
    #     durations_df = pd.read_csv(dept_duration_path)
    #     T = dict(zip(durations_df['dept'], durations_df['duration']))
    #     # Convert department names to indices
    #     T = {dept_to_idx[dept]: dur for dept, dur in T.items()}
        
    #     # Create and solve Gurobi model
    #     model = Model("InterviewScheduling")
    #     model.setParam("OutputFlag", 0)  # Suppress output
        
    #     # Add variables
    #     x = model.addVars(num_ids, num_depts, num_k, num_s, vtype=GRB.CONTINUOUS, name="x")
    #     y = model.addVars(num_ids, num_depts, num_k, vtype=GRB.CONTINUOUS, name="y")
    #     z = model.addVars(num_ids, num_depts, num_depts, num_k, vtype=GRB.CONTINUOUS, name="z")
        
    #     # Set objective
    #     model.setObjective(quicksum(x[i,j,k,s] for i in range(num_ids)
    #                                          for j in range(num_depts)
    #                                          for k in range(num_k)
    #                                          for s in range(num_s)), GRB.MAXIMIZE)
        
    #     # Constraint 1: Each applicant can have at most two interviews
    #     for i in range(num_ids):
    #         for j in range(num_depts):
    #             model.addConstr(
    #                 quicksum(x[i,j,k,s] for k in range(num_k) for s in range(num_s)) <= A.get((i,j), 0),
    #                 name=f"con1_{i}_{j}"
    #             )
        
    #     # Constraint 2: Only one interviewee per sub-slot per department
    #     for j in range(num_depts):
    #         for k in range(num_k):
    #             for s in range(num_s):
    #                 model.addConstr(quicksum(x[i,j,k,s] for i in range(num_ids)) <= 1,
    #                               name=f"con2_{j}_{k}_{s}")
        
    #     # Constraint 3: Within available time
    #     for i in range(num_ids):
    #         for j in range(num_depts):
    #             for k in range(num_k):
    #                 for s in range(num_s):
    #                     model.addConstr(x[i,j,k,s] <= O[i,j,k,s],
    #                                   name=f"con3_{i}_{j}_{k}_{s}")
        
    #     # Constraint 5: Avoid overlap on the same day
    #     for i in range(num_ids):
    #         for j in range(num_depts):
    #             model.addConstr(quicksum(y[i, j, k] for k in range(num_k)) <= 1,
    #                           name=f"con5_{i}_{j}")
        
    #     # Additional constraints for z variables
    #     for i in range(num_ids):
    #         for j in range(num_depts):
    #             for jp in range(num_depts):
    #                 if j == jp: continue
    #                 for k in range(num_k):
    #                     model.addConstr(z[i, j, jp, k] <= y[i, j, k],
    #                                   name=f"con6_{i}_{j}_{jp}_{k}")
    #                     model.addConstr(z[i, jp, j, k] <= y[i, jp, k],
    #                                   name=f"con7_{i}_{jp}_{j}_{k}")
        
    #     for i in range(num_ids):
    #         for j in range(num_depts):
    #             for jp in range(num_depts):
    #                 if j == jp: continue
    #                 for k in range(num_k):
    #                     model.addConstr(
    #                         y[i, j, k] + y[i, jp, k] <= z[i, j, jp, k] + z[i, jp, j, k] + 1,
    #                         name=f"con8_{i}_{j}_{jp}_{k}"
    #                     )
        
    #     # Constraint 9: Time overlap between interviews
    #     M = 10000  # Large number (minutes in a day)
    #     for i in range(num_ids):
    #         for j in range(num_depts):
    #             for jp in range(num_depts):
    #                 if j == jp: continue
    #                 for k in range(num_k):
    #                     for s in range(num_s):
    #                         for sp in range(num_s):
    #                             start_j = s * T[j]
    #                             end_j = (s + 1) * T[j]
    #                             start_jp = sp * T[jp]
    #                             end_jp = (sp + 1) * T[jp]
                                
    #                             # Only add constraint if end_j <= start_jp
    #                             if end_j <= start_jp:
    #                                 model.addConstr(
    #                                     x[i, j, k, s] * end_j - x[i, jp, k, sp] * start_jp
    #                                     <= M * (1 - z[i, j, jp, k]),
    #                                     name=f"con9_{i}_{j}_{jp}_{k}_{s}_{sp}"
    #                                 )
        
    #     # Optimize
    #     model.optimize()
        
    #     end_time = time.time()
        
    #     if model.status == GRB.OPTIMAL:
    #         results = []
    #         for i in range(num_ids):
    #             for j in range(num_depts):
    #                 for k in range(num_k):
    #                     for s in range(num_s):
    #                         if x[i,j,k,s].X > 0.5:
    #                             results.append({
    #                                 "ID": ids[i],
    #                                 "Department": depts[j],
    #                                 "Day": k_values[k],
    #                                 "Slot": s_values[s]
    #                             })
    #         return pd.DataFrame(results), end_time - start_time
    #     else:
    #         return None, end_time - start_time

    def run_comparison(self):
        """Run comparison between heuristic and Gurobi approaches"""
        with open(self.results_file, 'w', encoding='utf-8') as f:
            original_stdout = sys.stdout
            sys.stdout = TeeOutput(original_stdout, f)
            
            try:
                print(f"GIS Taiwan Interview Scheduling Comparison Results")
                print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80)
                print()
                
                # Generate all scenario instances
                print("Generating all scenario instances...")
                for idx, (a, s, d) in enumerate(generateInstance4.scenarios, 1):
                    generateInstance4.generate_scenario(a, s, d, idx)
                print("All scenario instances generated.\n")
                
                # Compare approaches for each scenario
                for idx in range(1, len(generateInstance4.scenarios) + 1):
                    print(f"{'='*50}")
                    print(f"SCENARIO {idx}")
                    print(f"{'='*50}")
                    
                    # Process data for both approaches
                    csv_path = f"scenario_{idx}.csv"
                    dept_duration_path = f"scenario_{idx}_dept_duration.csv"
                    
                    # Convert to 4D format for Gurobi
                    print("\nProcessing availability data to 4D format...")
                    df_4d = self.process_availability_to_4d(csv_path, dept_duration_path)
                    
                    # Calculate total possible independent interviews
                    # Calculate maximum possible interviews (unique applicant-department combinations)
                    unique_applicant_dept_combinations = set()
                    for _, row in df_4d[df_4d['available'] == 1].iterrows():
                        unique_applicant_dept_combinations.add((row['ID'], row['dept']))
                    total_possible = len(unique_applicant_dept_combinations)
                    
                    # Run naive
                    print("\nRunning naive algorithm...")
                    naive_results, naive_time = self.run_naive(csv_path, dept_duration_path)
                    
                    # Run heuristic
                    print("\nRunning heuristic algorithm...")
                    heuristic_scheduler, heuristic_time = self.run_heuristic(csv_path, dept_duration_path)
                    heuristic_results = len(heuristic_scheduler.schedule)
                    
                    # # Run Gurobi
                    # print("\nRunning Gurobi optimization...")
                    # gurobi_results, gurobi_time = self.run_gurobi(df_4d, dept_duration_path)
                    # gurobi_count = len(gurobi_results) if gurobi_results is not None else 0
                    
                    # Compare results
                    print(f"\nResults for Scenario {idx}:")
                    print(f"Total possible interviews: {total_possible}")
                    print(f"Naive algorithm scheduled: {naive_results} interviews (Time: {naive_time:.2f}s)")
                    print(f"Heuristic algorithm scheduled: {heuristic_results} interviews (Time: {heuristic_time:.2f}s)")
                    #print(f"Gurobi optimization scheduled: {gurobi_count} interviews (Time: {gurobi_time:.2f}s)")
                    print(f"Success rate - Naive: {(naive_results/total_possible*100):.1f}%")
                    print(f"Success rate - Heuristic: {(heuristic_results/total_possible*100):.1f}%")
                    #print(f"Success rate - Gurobi: {(gurobi_count/total_possible*100):.1f}%")
                    
                    # Store comparison stats
                    self.summary_stats.append({
                        'scenario': idx,
                        'total_possible': total_possible,
                        'naive_scheduled': naive_results,
                        'heuristic_scheduled': heuristic_results,
                        # 'gurobi_scheduled': gurobi_count,
                        'naive_time': naive_time,
                        'heuristic_time': heuristic_time,
                        # 'gurobi_time': gurobi_time,
                        'naive_success_rate': naive_results/total_possible*100,
                        'heuristic_success_rate': heuristic_results/total_possible*100,
                        # 'gurobi_success_rate': gurobi_count/total_possible*100
                    })
                    
                    print("\n" + "-"*50 + "\n")
                
                # Print summary statistics
                print("\n" + "=" * 80)
                print("SUMMARY STATISTICS")
                print("=" * 80)
                print(f"{'Scenario':<10} {'Total':<10} {'Naive':<12} {'Heuristic':<12} {'Gurobi':<12} {'N-Time':<10} {'H-Time':<10} {'G-Time':<10} {'N-Rate':<10} {'H-Rate':<10} {'G-Rate':<10} {'N-Gap':<10} {'H-Gap':<10} {'N-Time%':<10} {'H-Time%':<10}")
                print("-" * 160)
                
                total_possible_all = 0
                total_naive = 0
                total_heuristic = 0
                total_gurobi = 0
                # total_naive_time = 0
                # total_heuristic_time = 0
                # total_gurobi_time = 0
                
                for stats in self.summary_stats:
                    # Calculate optimality gaps
                    # naive_gap = ((stats['gurobi_scheduled'] - stats['naive_scheduled']) / stats['gurobi_scheduled'] * 100) if stats['gurobi_scheduled'] > 0 else 0
                    # heuristic_gap = ((stats['gurobi_scheduled'] - stats['heuristic_scheduled']) / stats['gurobi_scheduled'] * 100) if stats['gurobi_scheduled'] > 0 else 0
                    
                    # # Calculate time percentages relative to Gurobi
                    # naive_time_pct = (stats['naive_time'] / stats['gurobi_time'] * 100) if stats['gurobi_time'] > 0 else 0
                    # heuristic_time_pct = (stats['heuristic_time'] / stats['gurobi_time'] * 100) if stats['gurobi_time'] > 0 else 0
                    
                    print(f"{stats['scenario']:<10} {stats['total_possible']:<10} "
                          f"{stats['naive_scheduled']:<12} {stats['heuristic_scheduled']:<12} "
                        #   {stats['gurobi_scheduled']:<12} "
                        #   f"{stats['naive_time']:<10.2f} {stats['heuristic_time']:<10.2f} {stats['gurobi_time']:<10.2f} "
                          f"{stats['naive_success_rate']:<10.1f} {stats['heuristic_success_rate']:<10.1f}")
                        #    {stats['gurobi_success_rate']:<10.1f} ")
                        #   f"{naive_gap:<10.1f} {heuristic_gap:<10.1f} "
                        #   f"{naive_time_pct:<10.1f} {heuristic_time_pct:<10.1f}")
                    total_possible_all += stats['total_possible']
                    total_naive += stats['naive_scheduled']
                    total_heuristic += stats['heuristic_scheduled']
                    # total_gurobi += stats['gurobi_scheduled']
                    # total_naive_time += stats['naive_time']
                    # total_heuristic_time += stats['heuristic_time']
                    # total_gurobi_time += stats['gurobi_time']
                
                # # Calculate total optimality gaps
                # total_naive_gap = ((total_gurobi - total_naive) / total_gurobi * 100) if total_gurobi > 0 else 0
                # total_heuristic_gap = ((total_gurobi - total_heuristic) / total_gurobi * 100) if total_gurobi > 0 else 0
                
                # # Calculate total time percentages
                # total_naive_time_pct = (total_naive_time / total_gurobi_time * 100) if total_gurobi_time > 0 else 0
                # total_heuristic_time_pct = (total_heuristic_time / total_gurobi_time * 100) if total_gurobi_time > 0 else 0
                
                print("-" * 160)
                print(f"{'TOTAL':<10} {total_possible_all:<10} {total_naive:<12} {total_heuristic:<12} {total_gurobi:<12} ")
                    #   f"{total_naive_time:<10.2f} {total_heuristic_time:<10.2f} {total_gurobi_time:<10.2f} "
                    #   f"{(total_naive/total_possible_all*100):<10.1f} {(total_heuristic/total_possible_all*100):<10.1f} {(total_gurobi/total_possible_all*100):<10.1f} "
                    #   f"{total_naive_gap:<10.1f} {total_heuristic_gap:<10.1f} "
                    #   f"{total_naive_time_pct:<10.1f} {total_heuristic_time_pct:<10.1f}")
                
            finally:
                sys.stdout = original_stdout
        
        print(f"\nAll results have been saved to {self.results_file}")

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

if __name__ == "__main__":
    solver = ComparisonSolver()
    solver.run_comparison() 