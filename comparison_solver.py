import os
import sys
import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
from datetime import datetime
import generateInstance4
from heuristic_OriginData import InterviewSchedulerOptimized
from naive import NaiveInterviewScheduler
import time
from collections import defaultdict
os.environ['GRB_LICENSE_FILE'] = r"/Users/albert/gurobi/gurobi.lic"

class ComparisonSolver:
    def __init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = f"comparison_results_{timestamp}.txt"
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
        start = time.perf_counter()  # Use perf_counter for precise timing
        
        # Initialize and run naive scheduler
        scheduler = NaiveInterviewScheduler(csv_path)
        scheduler.interview_duration = dict(zip(pd.read_csv(dept_duration_path)['dept'], 
                                             pd.read_csv(dept_duration_path)['duration']))
        scheduler.reset_solution()
        total_assigned = scheduler.naive_schedule()
        
        # Convert schedule to the format needed for comparison
        schedule = []
        for applicant, dept, day, start_time, end_time in scheduler.schedule:
            schedule.append({
                'ID': applicant,
                'Department': dept,
                'Day': day,
                'Start_Time': scheduler.minutes_to_time_str(start_time),
                'End_Time': scheduler.minutes_to_time_str(end_time),
                'Duration': end_time - start_time
            })
        
        execution_time = time.perf_counter() - start  # Calculate elapsed time in seconds
        return schedule, execution_time
    
    def run_heuristic(self, csv_path, dept_duration_path):
        """Run the heuristic algorithm on a scenario"""
        start_time = time.time()
        scheduler = InterviewSchedulerOptimized(csv_path, dept_duration_path)
        scheduler.solve_heuristic()
        end_time = time.time()
        return scheduler, end_time - start_time
    
    def run_gurobi(self, df_4d, dept_duration_path):
        """Run the Gurobi optimization on a scenario using 4D data"""
        start_time = time.time()
        
        # Setup mappings and dimensions
        ids = sorted(df_4d['ID'].unique())
        depts = sorted(df_4d['dept'].unique())
        k_values = sorted(df_4d['k'].unique())
        s_values = sorted(df_4d['s'].unique())
        
        # Create index mappings
        id_to_idx = {id_: i for i, id_ in enumerate(ids)}
        dept_to_idx = {dept: j for j, dept in enumerate(depts)}
        k_to_idx = {k: kk for kk, k in enumerate(k_values)}
        s_to_idx = {s: ss for ss, s in enumerate(s_values)}
        
        # Create availability matrix
        num_ids = len(ids)
        num_depts = len(depts)
        num_k = len(k_values)
        num_s = len(s_values)
        
        O = np.zeros((num_ids, num_depts, num_k, num_s), dtype=int)
        for _, row in df_4d.iterrows():
            i = id_to_idx[row['ID']]
            j = dept_to_idx[row['dept']]
            k = k_to_idx[row['k']]
            s = s_to_idx[row['s']]
            O[i, j, k, s] = row['available']
        
        # A[i,j] = 1 if applicant i applied for department j
        A = {(i, j): 1 for (i, j, k, s), available in np.ndenumerate(O) if available == 1}
        
        # Read interview durations from CSV
        durations_df = pd.read_csv(dept_duration_path)
        T = dict(zip(durations_df['dept'], durations_df['duration']))
        # Convert department names to indices
        T = {dept_to_idx[dept]: dur for dept, dur in T.items()}
        
        # Create and solve Gurobi model
        model = Model("InterviewScheduling")
        model.setParam("OutputFlag", 0)  # Suppress output
        
        # Add variables
        x = model.addVars(num_ids, num_depts, num_k, num_s, vtype=GRB.CONTINUOUS, name="x")
        y = model.addVars(num_ids, num_depts, num_k, vtype=GRB.CONTINUOUS, name="y")
        z = model.addVars(num_ids, num_depts, num_depts, num_k, vtype=GRB.CONTINUOUS, name="z")
        
        # Set objective
        model.setObjective(quicksum(x[i,j,k,s] for i in range(num_ids)
                                             for j in range(num_depts)
                                             for k in range(num_k)
                                             for s in range(num_s)), GRB.MAXIMIZE)
        
        # Constraint 1: Each applicant can have at most two interviews
        for i in range(num_ids):
            for j in range(num_depts):
                model.addConstr(
                    quicksum(x[i,j,k,s] for k in range(num_k) for s in range(num_s)) <= A.get((i,j), 0),
                    name=f"con1_{i}_{j}"
                )
        
        # Constraint 2: Only one interviewee per sub-slot per department
        for j in range(num_depts):
            for k in range(num_k):
                for s in range(num_s):
                    model.addConstr(quicksum(x[i,j,k,s] for i in range(num_ids)) <= 1,
                                  name=f"con2_{j}_{k}_{s}")
        
        # Constraint 3: Within available time
        for i in range(num_ids):
            for j in range(num_depts):
                for k in range(num_k):
                    for s in range(num_s):
                        model.addConstr(x[i,j,k,s] <= O[i,j,k,s],
                                      name=f"con3_{i}_{j}_{k}_{s}")
        
        # Constraint 5: Avoid overlap on the same day
        for i in range(num_ids):
            for j in range(num_depts):
                model.addConstr(quicksum(y[i, j, k] for k in range(num_k)) <= 1,
                              name=f"con5_{i}_{j}")
        
        # Additional constraints for z variables
        for i in range(num_ids):
            for j in range(num_depts):
                for jp in range(num_depts):
                    if j == jp: continue
                    for k in range(num_k):
                        model.addConstr(z[i, j, jp, k] <= y[i, j, k],
                                      name=f"con6_{i}_{j}_{jp}_{k}")
                        model.addConstr(z[i, jp, j, k] <= y[i, jp, k],
                                      name=f"con7_{i}_{jp}_{j}_{k}")
        
        for i in range(num_ids):
            for j in range(num_depts):
                for jp in range(num_depts):
                    if j == jp: continue
                    for k in range(num_k):
                        model.addConstr(
                            y[i, j, k] + y[i, jp, k] <= z[i, j, jp, k] + z[i, jp, j, k] + 1,
                            name=f"con8_{i}_{j}_{jp}_{k}"
                        )
        
        # Constraint 9: Time overlap between interviews
        M = 10000  # Large number (minutes in a day)
        for i in range(num_ids):
            for j in range(num_depts):
                for jp in range(num_depts):
                    if j == jp: continue
                    for k in range(num_k):
                        for s in range(num_s):
                            for sp in range(num_s):
                                start_j = s * T[j]
                                end_j = (s + 1) * T[j]
                                start_jp = sp * T[jp]
                                end_jp = (sp + 1) * T[jp]
                                
                                # Only add constraint if end_j <= start_jp
                                if end_j <= start_jp:
                                    model.addConstr(
                                        x[i, j, k, s] * end_j - x[i, jp, k, sp] * start_jp
                                        <= M * (1 - z[i, j, jp, k]),
                                        name=f"con9_{i}_{j}_{jp}_{k}_{s}_{sp}"
                                    )
        
        # Optimize
        model.optimize()
        
        end_time = time.time()
        
        if model.status == GRB.OPTIMAL:
            results = []
            for i in range(num_ids):
                for j in range(num_depts):
                    for k in range(num_k):
                        for s in range(num_s):
                            if x[i,j,k,s].X > 0.5:
                                results.append({
                                    "ID": ids[i],
                                    "Department": depts[j],
                                    "Day": k_values[k],
                                    "Slot": s_values[s]
                                })
            return pd.DataFrame(results), end_time - start_time
        else:
            return None, end_time - start_time

    def run_comparison(self):
        """Run comparison between heuristic and Gurobi approaches"""
        num_runs = 10
        all_runs_stats = []  # Store stats for all runs
        
        # Create timestamp for this comparison run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for run in range(num_runs):
            print(f"\n{'='*80}")
            print(f"RUN {run + 1} of {num_runs}")
            print(f"{'='*80}\n")
            
            # Create a unique filename for this run
            run_timestamp = f"{timestamp}_run_{run + 1}"
            with open(f"comparison_results_{run_timestamp}.txt", 'w', encoding='utf-8') as f:
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
                    run_stats = []  # Store stats for this run
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
                        
                        # Run Gurobi
                        print("\nRunning Gurobi optimization...")
                        gurobi_results, gurobi_time = self.run_gurobi(df_4d, dept_duration_path)
                        gurobi_count = len(gurobi_results) if gurobi_results is not None else 0
                        
                        # Calculate day distributions
                        naive_day_dist = self.calculate_day_distribution(naive_results)
                        heuristic_day_dist = self.calculate_day_distribution(heuristic_scheduler.schedule)
                        gurobi_day_dist = self.calculate_day_distribution(gurobi_results)
                        
                        # Store stats for this scenario
                        scenario_stats = {
                            'scenario': idx,
                            'total_possible': total_possible,
                            'naive_scheduled': len(naive_results),
                            'heuristic_scheduled': heuristic_results,
                            'gurobi_scheduled': gurobi_count,
                            'naive_time': naive_time,
                            'heuristic_time': heuristic_time,
                            'gurobi_time': gurobi_time,
                            'naive_day_dist': naive_day_dist,
                            'heuristic_day_dist': heuristic_day_dist,
                            'gurobi_day_dist': gurobi_day_dist
                        }
                        run_stats.append(scenario_stats)
                        
                        # Print results for this scenario
                        print(f"\nResults for Scenario {idx}:")
                        print(f"Total possible interviews: {total_possible}")
                        print(f"Naive algorithm scheduled: {len(naive_results)} interviews (Time: {naive_time:.2f}s)")
                        print(f"Heuristic algorithm scheduled: {heuristic_results} interviews (Time: {heuristic_time:.2f}s)")
                        print(f"Gurobi optimization scheduled: {gurobi_count} interviews (Time: {gurobi_time:.2f}s)")
                        
                        # Print day distribution for this scenario
                        print("\nDay Distribution (Number of applicants requiring X days):")
                        print(f"{'Days':<10} {'Naive':<12} {'Heuristic':<12} {'Gurobi':<12}")
                        print("-" * 50)
                        
                        all_day_counts = set()
                        for dist in [naive_day_dist, heuristic_day_dist, gurobi_day_dist]:
                            all_day_counts.update(dist.keys())
                        
                        for days in sorted(all_day_counts):
                            print(f"{days:<10} {naive_day_dist[days]:<12} {heuristic_day_dist[days]:<12} {gurobi_day_dist[days]:<12}")
                        
                        def calculate_avg_days(dist):
                            if not dist:
                                return 0
                            total_applicants = sum(dist.values())
                            total_days = sum(days * count for days, count in dist.items())
                            return total_days / total_applicants
                        
                        print("-" * 50)
                        print(f"{'Average':<10} {calculate_avg_days(naive_day_dist):<12.2f} {calculate_avg_days(heuristic_day_dist):<12.2f} {calculate_avg_days(gurobi_day_dist):<12.2f}")
                        
                        print("\n" + "-"*50 + "\n")
                    
                    all_runs_stats.append(run_stats)
                    
                finally:
                    sys.stdout = original_stdout
        
        # Save statistics to CSV
        stats_data = []
        
        # Process scenario statistics
        for scenario_idx in range(len(generateInstance4.scenarios)):
            scenario_data = {
                'Scenario': scenario_idx + 1,
                'Type': 'Scenario'
            }
            
            # Collect data for this scenario across all runs
            for metric in ['naive_scheduled', 'heuristic_scheduled', 'gurobi_scheduled',
                          'naive_time', 'heuristic_time', 'gurobi_time']:
                values = [run_stats[scenario_idx][metric] for run_stats in all_runs_stats]
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                std_dev = variance ** 0.5
                
                scenario_data[f'{metric}_mean'] = mean
                scenario_data[f'{metric}_std_dev'] = std_dev
            
            # Calculate average days statistics
            for approach in ['naive', 'heuristic', 'gurobi']:
                values = []
                for run_stats in all_runs_stats:
                    day_dist = run_stats[scenario_idx][f'{approach}_day_dist']
                    if day_dist:
                        total_applicants = sum(day_dist.values())
                        total_days = sum(days * count for days, count in day_dist.items())
                        avg_days = total_days / total_applicants if total_applicants > 0 else 0
                    else:
                        avg_days = 0
                    values.append(avg_days)
                
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                std_dev = variance ** 0.5
                
                scenario_data[f'{approach}_avg_days_mean'] = mean
                scenario_data[f'{approach}_avg_days_std_dev'] = std_dev
            
            stats_data.append(scenario_data)
        
        # Process overall statistics
        overall_data = {
            'Scenario': 'Overall',
            'Type': 'Overall'
        }
        
        for metric in ['naive_scheduled', 'heuristic_scheduled', 'gurobi_scheduled',
                      'naive_time', 'heuristic_time', 'gurobi_time']:
            values = []
            for run_stats in all_runs_stats:
                for stats in run_stats:
                    values.append(stats[metric])
            
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5
            
            overall_data[f'{metric}_mean'] = mean
            overall_data[f'{metric}_std_dev'] = std_dev
        
        # Calculate overall average days statistics
        for approach in ['naive', 'heuristic', 'gurobi']:
            values = []
            for run_stats in all_runs_stats:
                for stats in run_stats:
                    day_dist = stats[f'{approach}_day_dist']
                    if day_dist:
                        total_applicants = sum(day_dist.values())
                        total_days = sum(days * count for days, count in day_dist.items())
                        avg_days = total_days / total_applicants if total_applicants > 0 else 0
                    else:
                        avg_days = 0
                    values.append(avg_days)
            
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5
            
            overall_data[f'{approach}_avg_days_mean'] = mean
            overall_data[f'{approach}_avg_days_std_dev'] = std_dev
        
        stats_data.append(overall_data)
        
        # Save to CSV
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(f"comparison_statistics_{timestamp}.csv", index=False)
        
        print(f"\nStatistics have been saved to comparison_statistics_{timestamp}.csv")

    def calculate_day_distribution(self, schedule):
        """Calculate the distribution of interview days per applicant."""
        if schedule is None:
            return {}
            
        # Track days per applicant
        applicant_days = defaultdict(set)
        
        if isinstance(schedule, list):  # For naive and heuristic results
            for record in schedule:
                if isinstance(record, tuple):  # heuristic format
                    applicant, _, day, _, _ = record
                    applicant_days[applicant].add(day)
                elif isinstance(record, dict):  # naive format
                    applicant = record['ID']
                    day = record['Day']
                    applicant_days[applicant].add(day)
        elif isinstance(schedule, pd.DataFrame):  # For Gurobi results
            for _, row in schedule.iterrows():
                applicant = row['ID']
                day = row['Day']
                applicant_days[applicant].add(day)
        
        # Count distribution
        day_dist = defaultdict(int)
        for days in applicant_days.values():
            day_dist[len(days)] += 1
            
        return day_dist

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