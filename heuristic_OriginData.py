#!/usr/bin/env python3
"""
GIS Taiwan Interview Scheduling Heuristic Algorithm - Optimized Version
======================================================================

This version uses a more efficient approach:
- Only stores actual available time ranges (not pre-calculated slots)
- Dynamically checks feasibility when needed
- Much lower memory usage and faster execution

Author: Claude AI Assistant  
Date: May 2025
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import time
from datetime import datetime, timedelta
import os
from itertools import combinations

class InterviewSchedulerOptimized:
    def __init__(self, csv_file='availability_records.csv', dept_duration_file=None):
        """Initialize the optimized scheduler."""
        self.csv_file = csv_file
        self.dept_duration_file = dept_duration_file
        self.load_data()
        self.setup_parameters()
        self.reset_solution()
        
    def parse_time_to_minutes(self, time_str):
        """Convert time string like '18:00' to minutes from start of day."""
        hour, minute = map(int, time_str.split(':'))
        return hour * 60 + minute
    
    def parse_date_to_k(self, date_str):
        """Convert date string to k value (1-8 for 9/19-9/26)."""
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            day = date_obj.day
            if 19 <= day <= 26:
                return day - 18  # 9/19->1, 9/20->2, ..., 9/26->8
            return None
        except:
            print(f"Warning: Could not parse date '{date_str}'")
            return None
    
    def load_data(self):
        """Load and parse availability data efficiently."""
        print("Loading availability data...")
        
        self.df = pd.read_csv(self.csv_file)
        print(f"Raw data: {len(self.df)} records")
        
        # Extract basic info
        self.applicants = sorted(self.df['ID'].unique())
        self.departments = sorted(self.df['dept'].unique())
        self.applicant_names = dict(zip(self.df['ID'], self.df['Name']))
        
        # Store available time ranges for each (applicant, department, date)
        # Structure: {(applicant, dept, date_k): [(start_minutes, end_minutes), ...]}
        self.available_times = defaultdict(list)
        
        valid_records = 0
        for _, row in self.df.iterrows():
            applicant = row['ID']
            dept = row['dept']
            k = self.parse_date_to_k(row['date'])
            
            if k is None:
                continue
                
            # Parse time range
            try:
                start_str, end_str = row['time_slot'].split('-')
                start_minutes = self.parse_time_to_minutes(start_str)
                end_minutes = self.parse_time_to_minutes(end_str)
                
                # Store the time range
                self.available_times[(applicant, dept, k)].append((start_minutes, end_minutes))
                valid_records += 1
                
            except Exception as e:
                print(f"Warning: Could not parse time slot '{row['time_slot']}': {e}")
        
        # Get unique dates
        self.dates = sorted(set(k for (_, _, k) in self.available_times.keys()))
        
        print(f"Processed {valid_records} valid availability records")
        print(f"Applicants: {len(self.applicants)}, Departments: {len(self.departments)}")
        print(f"Interview dates (k): {self.dates}")
        
        # Show statistics
        stats_by_dept = defaultdict(int)
        for (_, dept, _), time_ranges in self.available_times.items():
            stats_by_dept[dept] += len(time_ranges)
        
        print("Available time ranges per department:")
        for dept in self.departments:
            print(f"  {dept}: {stats_by_dept[dept]} time ranges")
    
    def setup_parameters(self):
        """Set up interview durations and other parameters."""
        # Interview durations in minutes
        self.interview_duration = {}
        if self.dept_duration_file and os.path.exists(self.dept_duration_file):
            df = pd.read_csv(self.dept_duration_file)
            for _, row in df.iterrows():
                self.interview_duration[row['dept']] = int(row['duration'])
            print("Interview durations loaded from", self.dept_duration_file, ":", self.interview_duration)
        else:
            # 若無檔案則給預設值
            self.interview_duration = {
                'AC': 15,
                'GPDA': 20,
                'PR': 15,
                'DM': 25
            }
            print("Interview durations (default):", self.interview_duration)

        # 若有新部門但沒在interview_duration裡，給預設15分鐘
        for dept in self.departments:
            if dept not in self.interview_duration:
                self.interview_duration[dept] = 15
        
        print("Interview durations (minutes):", self.interview_duration)
    
    def reset_solution(self):
        """Reset solution state."""
        # Current schedule: [(applicant, dept, date_k, start_time_minutes, end_time_minutes)]
        self.schedule = []
        
        # Track occupied time for each department on each date
        # Structure: {(dept, date_k): [(start_minutes, end_minutes), ...]}
        self.dept_occupied_times = defaultdict(list)
        
        # Track applicant schedules by date
        # Structure: {(applicant, date_k): [(start_minutes, end_minutes, dept), ...]}
        self.applicant_occupied_times = defaultdict(list)
        
        # Counters
        self.applicant_interview_count = defaultdict(int)
        self.applicant_dept_count = defaultdict(lambda: defaultdict(int))
    
    def times_overlap(self, start1, end1, start2, end2):
        """Check if two time ranges overlap."""
        return start1 < end2 and start2 < end1
    
    def can_schedule_interview(self, applicant, dept, date_k, start_time, end_time):
        """Check if we can schedule an interview at the given time."""
        # Check if applicant is available for this time
        available_ranges = self.available_times.get((applicant, dept, date_k), [])
        is_available = any(
            av_start <= start_time and end_time <= av_end 
            for av_start, av_end in available_ranges
        )
        
        if not is_available:
            return False
          # Check department conflicts (only for the same department and date)
        dept_schedule = self.dept_occupied_times[(dept, date_k)]
        for occupied_start, occupied_end in dept_schedule:
            if self.times_overlap(start_time, end_time, occupied_start, occupied_end):
                return False
        
        # Check applicant conflicts on same date
        applicant_schedule = self.applicant_occupied_times[(applicant, date_k)]
        for occupied_start, occupied_end, _ in applicant_schedule:
            if self.times_overlap(start_time, end_time, occupied_start, occupied_end):
                return False
        return True
    
    def consolidation_optimization(self):
        """Dedicated consolidation phase to reduce multi-day applicants."""
        print("Starting consolidation optimization...")
        
        # Get multi-day applicants
        applicant_days = defaultdict(set)
        applicant_interviews = defaultdict(list)
        for a, dept, k, start_time, end_time in self.schedule:
            applicant_days[a].add(k)
            applicant_interviews[a].append((dept, k, start_time, end_time))
        
        multi_day_applicants = [a for a, days in applicant_days.items() if len(days) > 1]
        consolidated_count = 0
        
        print(f"Found {len(multi_day_applicants)} multi-day applicants to consolidate")
        
        for applicant in multi_day_applicants:
            name = self.applicant_names.get(applicant, f"ID_{applicant}")
            interviews = applicant_interviews[applicant]
            days = sorted(list(applicant_days[applicant]))
            
            # Try to consolidate: move interviews from less populated days to more populated days
            # Count interviews per day for this applicant
            interviews_per_day = defaultdict(list)
            for dept, k, start_time, end_time in interviews:
                interviews_per_day[k].append((dept, start_time, end_time))
            
            # Sort days by number of interviews (ascending), so we try to move from days with fewer interviews
            day_priority = sorted(days, key=lambda d: len(interviews_per_day[d]))
            
            consolidated = False
            for source_day in day_priority:
                if consolidated:
                    break
                    
                for target_day in day_priority:
                    if source_day == target_day or consolidated:
                        continue
                    
                    # Try to move all interviews from source_day to target_day
                    source_interviews = interviews_per_day[source_day]
                    if len(source_interviews) > len(interviews_per_day[target_day]):
                        continue  # Skip if source has more interviews than target
                    
                    # Check if we can move all interviews from source to target
                    can_move_all = True
                    move_plans = []
                    
                    for dept, old_start, old_end in source_interviews:
                        # Check availability for this department on target day
                        available_ranges = self.available_times.get((applicant, dept, target_day), [])
                        if not available_ranges:
                            can_move_all = False
                            break
                        
                        # Temporarily remove the interview from source day
                        temp_schedule_backup = self.schedule.copy()
                        temp_dept_backup = {k: v.copy() for k, v in self.dept_occupied_times.items()}
                        temp_app_backup = {k: v.copy() for k, v in self.applicant_occupied_times.items()}
                        
                        try:
                            self.remove_interview(applicant, dept, source_day, old_start)
                            
                            # Find feasible times on target day
                            feasible_times = self.find_feasible_times(applicant, dept, target_day)
                            
                            if feasible_times:
                                # Choose the best feasible time (earliest available)
                                new_start = feasible_times[0]
                                move_plans.append((dept, source_day, old_start, target_day, new_start))
                            else:
                                can_move_all = False
                                
                        finally:
                            # Restore state for next check
                            self.schedule = temp_schedule_backup
                            self.dept_occupied_times = temp_dept_backup
                            self.applicant_occupied_times = temp_app_backup
                        
                        if not can_move_all:
                            break
                    
                    # If we can move all interviews, do it
                    if can_move_all and move_plans:
                        for dept, old_day, old_start, new_day, new_start in move_plans:
                            self.remove_interview(applicant, dept, old_day, old_start)
                            self.assign_interview(applicant, dept, new_day, new_start)
                        
                        consolidated_count += 1
                        consolidated = True
                        print(f"  ✓ Consolidated {name}: Moved {len(move_plans)} interview(s) from day {source_day} to day {target_day}")
                        break
            
            if not consolidated:
                # Try a different strategy: move single interviews to days with existing interviews
                for source_day in day_priority:
                    if consolidated:
                        break
                        
                    source_interviews = interviews_per_day[source_day]
                    if len(source_interviews) != 1:  # Only try to move single interviews
                        continue
                    
                    dept, old_start, old_end = source_interviews[0]
                    
                    for target_day in day_priority:
                        if source_day == target_day or len(interviews_per_day[target_day]) == 0:
                            continue  # Only move to days that already have interviews
                        
                        # Check if we can move this interview
                        available_ranges = self.available_times.get((applicant, dept, target_day), [])
                        if not available_ranges:
                            continue
                        
                        # Try to move
                        temp_schedule_backup = self.schedule.copy()
                        temp_dept_backup = {k: v.copy() for k, v in self.dept_occupied_times.items()}
                        temp_app_backup = {k: v.copy() for k, v in self.applicant_occupied_times.items()}
                        
                        try:
                            self.remove_interview(applicant, dept, source_day, old_start)
                            feasible_times = self.find_feasible_times(applicant, dept, target_day)
                            
                            if feasible_times:
                                new_start = feasible_times[0]
                                self.assign_interview(applicant, dept, target_day, new_start)
                                consolidated_count += 1
                                consolidated = True
                                print(f"  ✓ Consolidated {name}: Moved {dept} from day {source_day} to day {target_day}")
                                break
                            else:
                                # Restore state if we can't move
                                self.schedule = temp_schedule_backup
                                self.dept_occupied_times = temp_dept_backup
                                self.applicant_occupied_times = temp_app_backup
                        except:
                            # Restore state on any error
                            self.schedule = temp_schedule_backup
                            self.dept_occupied_times = temp_dept_backup
                            self.applicant_occupied_times = temp_app_backup
        
        print(f"Consolidation optimization completed: {consolidated_count} applicants consolidated")
        return consolidated_count
    
    
    def find_feasible_times(self, applicant, dept, date_k):
        """Find all feasible start times for an interview."""
        available_ranges = self.available_times.get((applicant, dept, date_k), [])
        duration = self.interview_duration[dept]
        feasible_times = []
        
        # Get already occupied times for this department on this date
        occupied_times = self.dept_occupied_times[(dept, date_k)]
        
        for av_start, av_end in available_ranges:
            # First, try to start immediately at the beginning of available range
            current_time = av_start
            while current_time + duration <= av_end:
                end_time = current_time + duration
                
                if self.can_schedule_interview(applicant, dept, date_k, current_time, end_time):
                    feasible_times.append(current_time)
                
                # Move to next possible start time
                # Check if there's an occupied slot that we need to skip
                next_time = current_time + 5  # 5-minute increments for finer granularity
                
                # Skip over any occupied time periods
                for occ_start, occ_end in occupied_times:
                    if current_time < occ_end and next_time > occ_start:
                        next_time = max(next_time, occ_end)  # Start right after occupied time
                
                current_time = next_time
        
        # Sort feasible times to prioritize earlier times and times right after other interviews
        def time_priority(start_time):
            # Check if this start time is right after another interview ends
            immediately_after_bonus = 0
            for occ_start, occ_end in occupied_times:
                if start_time == occ_end:  # Starts exactly when another interview ends
                    immediately_after_bonus = -10000  # High priority bonus
                    break
            
            return start_time + immediately_after_bonus
        
        feasible_times.sort(key=time_priority)
        return feasible_times
    
    def assign_interview(self, applicant, dept, date_k, start_time):
        """Assign an interview and update tracking structures."""
        duration = self.interview_duration[dept]
        end_time = start_time + duration
        
        # Add to schedule
        self.schedule.append((applicant, dept, date_k, start_time, end_time))
        
        # Update occupied times
        self.dept_occupied_times[(dept, date_k)].append((start_time, end_time))
        self.applicant_occupied_times[(applicant, date_k)].append((start_time, end_time, dept))
        
        # Update counters
        self.applicant_interview_count[applicant] += 1
        self.applicant_dept_count[applicant][dept] += 1
    
    def remove_interview(self, applicant, dept, date_k, start_time):
        """Remove an interview and update tracking structures."""
        duration = self.interview_duration[dept]
        end_time = start_time + duration
        
        # Remove from schedule
        self.schedule = [
            (a, d, k, st, et) for (a, d, k, st, et) in self.schedule
            if not (a == applicant and d == dept and k == date_k and st == start_time)
        ]
        
        # Update occupied times
        self.dept_occupied_times[(dept, date_k)] = [
            (st, et) for (st, et) in self.dept_occupied_times[(dept, date_k)]
            if not (st == start_time and et == end_time)
        ]
        
        self.applicant_occupied_times[(applicant, date_k)] = [
            (st, et, d) for (st, et, d) in self.applicant_occupied_times[(applicant, date_k)]
            if not (st == start_time and et == end_time and d == dept)
        ]
        
        # Update counters
        self.applicant_interview_count[applicant] -= 1
        self.applicant_dept_count[applicant][dept] -= 1

    def calculate_applicant_dept_priority(self, applicant, dept):
        """Calculate priority for (applicant, department) combination."""
        # Calculate available time for this specific applicant-department combination
        dept_minutes = 0
        for date_k in self.dates:
            available_ranges = self.available_times.get((applicant, dept, date_k), [])
            for start, end in available_ranges:
                dept_minutes += (end - start)
        
        if dept_minutes == 0:
            return float('inf')  # No availability, lowest priority
        
        dept_hours = dept_minutes / 60
        
        # Calculate urgency score - less time = higher urgency = lower score (higher priority)
        urgency_score = dept_minutes * 1000  # Base score
        
        # Apply urgency bonuses
        if dept_hours < 0.5:  # Less than 30 minutes
            urgency_score -= 100000  # Extremely high priority
        elif dept_hours < 1:  # Less than 1 hour
            urgency_score -= 50000   # Very high priority
        elif dept_hours < 2:  # Less than 2 hours
            urgency_score -= 20000   # High priority
        elif dept_hours < 4:  # Less than 4 hours
            urgency_score -= 10000   # Medium priority
        
        return urgency_score

    def greedy_assignment(self):
        """Main greedy assignment algorithm - dynamically update priorities after each assignment."""
        print("Starting greedy assignment with dynamic urgency-based prioritization...")
        
        assignments_made = 0
        scheduled_pairs = set()  # (applicant, dept) 已排過的組合
        
        while True:
            # 動態計算所有還沒排過的 applicant-dept priority
            applicant_dept_priorities = []
            for applicant in self.applicants:
                for dept in self.departments:
                    if (applicant, dept) in scheduled_pairs:
                        continue
                    if self.applicant_dept_count[applicant][dept] > 0:
                        continue
                    priority = self.calculate_applicant_dept_priority(applicant, dept)
                    if priority != float('inf'):
                        applicant_dept_priorities.append((priority, applicant, dept))
            if not applicant_dept_priorities:
                break
            applicant_dept_priorities.sort()  # Lower score = higher priority
            # 只取最緊急的那一組
            priority, applicant, dept = applicant_dept_priorities[0]
            name = self.applicant_names.get(applicant, f"ID_{applicant}")
            available_hours = sum(
                (end - start) for date_k in self.dates
                for start, end in self.available_times.get((applicant, dept, date_k), [])
            ) / 60
            print(f"  Processing {name} ({applicant}) - {dept}: {available_hours:.1f}h available")
            # 找最佳時段
            best_assignment = None
            best_score = float('-inf')
            for date_k in self.dates:
                feasible_times = self.find_feasible_times(applicant, dept, date_k)
                for start_time in feasible_times:
                    time_preference = -start_time // 60
                    date_preference = -date_k * 10
                    existing_interviews_on_date = len(self.applicant_occupied_times[(applicant, date_k)])
                    same_day_bonus = existing_interviews_on_date * 100
                    score = time_preference + date_preference + same_day_bonus
                    if score > best_score:
                        best_score = score
                        best_assignment = (dept, date_k, start_time)
            # 排入
            if best_assignment:
                dept, date_k, start_time = best_assignment
                duration = self.interview_duration[dept]
                end_time = start_time + duration
                if self.can_schedule_interview(applicant, dept, date_k, start_time, end_time):
                    self.assign_interview(applicant, dept, date_k, start_time)
                    scheduled_pairs.add((applicant, dept))
                    assignments_made += 1
                    bonus_str = "(same-day)" if best_score % 100 >= 50 else ""
                    print(f"    ✓ Assigned {dept} on day {date_k} at {self.minutes_to_time_str(start_time)} {bonus_str}")
                else:
                    print(f"    ✗ Failed to assign (conflict detected)")
            else:
                # 沒有可行時段，標記為已排過，避免無限迴圈
                scheduled_pairs.add((applicant, dept))
        scheduled_applicants = len(set(a for a, _, _, _, _ in self.schedule))
        print(f"Phase 1 completed: {assignments_made} interviews assigned, {scheduled_applicants}/{len(self.applicants)} applicants have interviews")
        return assignments_made 
      
    def solve_heuristic(self):
        """Main solving function with full coverage priority."""
        print("=" * 50)
        print("OPTIMIZED HEURISTIC SCHEDULING - FULL COVERAGE")
        print("=" * 50)
        
        start_time = time.time()
        
        self.reset_solution()
        
        # Phase 1: Greedy assignment with coverage priority
        assignments = self.greedy_assignment()
        # Check coverage
        scheduled_applicants = len(set(a for a, _, _, _, _ in self.schedule))
        print(f"\nAfter greedy assignment: {scheduled_applicants}/{len(self.applicants)} applicants scheduled")
        
        # Phase 2: Consolidation optimization (try to reduce multi-day applicants)
        final_scheduled = len(set(a for a, _, _, _, _ in self.schedule))
        if final_scheduled >= len(self.applicants) * 0.9:  # If 90%+ coverage
            print("Good coverage achieved, starting consolidation optimization...")
            consolidation_improvements = self.consolidation_optimization()
            print(f"Consolidation completed: {consolidation_improvements} applicants consolidated")
        else:
            print("Coverage still insufficient, skipping optimization phases")
        
        solve_time = time.time() - start_time
        print(f"\nHeuristic completed in {solve_time:.2f} seconds")
        
        final_coverage = len(set(a for a, _, _, _, _ in self.schedule))
        success_rate = final_coverage / len(self.applicants) * 100
        print(f"Final coverage: {final_coverage}/{len(self.applicants)} ({success_rate:.1f}%)")
        
        return len(self.schedule) > 0
    
    def minutes_to_time_str(self, minutes):
        """Convert minutes from start of day to HH:MM format."""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    def analyze_results(self):
        """Analyze and display results."""
        print("\n" + "="*50)
        print("SOLUTION ANALYSIS")
        print("="*50)        
        total_interviews = len(self.schedule)
        print(f"Total scheduled interviews: {total_interviews}")
        
        # Calculate maximum possible interviews (unique applicant-department combinations)
        unique_applicant_dept_combinations = set()
        for (applicant, dept, date_k) in self.available_times.keys():
            unique_applicant_dept_combinations.add((applicant, dept))
        max_possible_interviews = len(unique_applicant_dept_combinations)
        print(f"Maximum possible interviews: {max_possible_interviews}")
        
        if total_interviews == 0:
            print("No interviews were scheduled!")
            return
        
        # Department statistics
        dept_counts = defaultdict(int)
        for _, dept, _, _, _ in self.schedule:
            dept_counts[dept] += 1
        
        print("\nInterviews per department:")
        for dept in self.departments:
            print(f"  {dept}: {dept_counts[dept]}")
        
        # Applicant statistics
        applicants_with_interviews = set(a for a, _, _, _, _ in self.schedule)
        print(f"\nApplicants with interviews: {len(applicants_with_interviews)} / {len(self.applicants)}")
        
        # Day distribution
        applicant_days = defaultdict(set)
        for a, _, k, _, _ in self.schedule:
            applicant_days[a].add(k)
        
        days_distribution = defaultdict(int)
        for applicant, days in applicant_days.items():
            days_distribution[len(days)] += 1
        
        print("\nDays distribution:")
        for num_days in sorted(days_distribution.keys()):
            count = days_distribution[num_days]
            print(f"  {num_days} day(s): {count} applicants")
        
        # Sample interviews
        print("\nSample scheduled interviews:")
        for i, (applicant, dept, date_k, start_time, end_time) in enumerate(self.schedule[:8]):
            name = self.applicant_names.get(applicant, f"ID_{applicant}")
            start_str = self.minutes_to_time_str(start_time)
            end_str = self.minutes_to_time_str(end_time)
            print(f"  {name} ({applicant}) - {dept} - Day {date_k} - {start_str}-{end_str}")
        
        if len(self.schedule) > 8:
            print(f"  ... and {len(self.schedule) - 8} more interviews")
          # Average days per applicant
        if applicants_with_interviews:
            total_days = sum(len(days) for days in applicant_days.values())
            avg_days = total_days / len(applicants_with_interviews)
            print(f"\nAverage days per applicant: {avg_days:.2f}")
          # Multi-day analysis
        self.analyze_multi_day_applicants()
    
    def export_schedule(self, filename='optimized_schedule.csv'):
        """Export schedule to CSV."""
        if not self.schedule:
            print("No schedule to export!")
            return
        
        schedule_data = []
        for applicant, dept, date_k, start_time, end_time in self.schedule:
            name = self.applicant_names.get(applicant, f"ID_{applicant}")
            start_str = self.minutes_to_time_str(start_time)
            end_str = self.minutes_to_time_str(end_time)
            
            schedule_data.append({
                'Applicant_ID': applicant,
                'Name': name,
                'Department': dept,
                'Date_K': date_k,
                'Start_Time': start_str,
                'End_Time': end_str,
                'Duration_Minutes': end_time - start_time
            })
        
        df_schedule = pd.DataFrame(schedule_data)
        df_schedule = df_schedule.sort_values(['Date_K', 'Start_Time', 'Department'])
        df_schedule.to_csv(filename, index=False)
        print(f"\nSchedule exported to {filename}")

    def analyze_multi_day_applicants(self):
        """Analyze and report applicants with interviews on multiple days."""
        # Day distribution by applicant
        applicant_days = defaultdict(set)
        for a, _, k, _, _ in self.schedule:
            applicant_days[a].add(k)
        
        # Count applicants with multiple days
        multi_day_applicants = []
        for applicant, days in applicant_days.items():
            if len(days) > 1:
                multi_day_applicants.append((applicant, days))
        
        print(f"\nMulti-day Interview Analysis:")
        print(f"Applicants with interviews on multiple days: {len(multi_day_applicants)}")
        
        if multi_day_applicants:
            print("\nDetailed multi-day interview breakdown:")
            for applicant, days in sorted(multi_day_applicants, key=lambda x: len(x[1]), reverse=True):
                name = self.applicant_names.get(applicant, f"ID_{applicant}")
                days_list = sorted(list(days))
                print(f"  {name} ({applicant}): {len(days)} days - Days {days_list}")
                
                # Show which departments on which days
                dept_by_day = defaultdict(list)
                for a, dept, k, start_time, end_time in self.schedule:
                    if a == applicant:
                        start_str = self.minutes_to_time_str(start_time)
                        end_str = self.minutes_to_time_str(end_time)
                        dept_by_day[k].append(f"{dept} ({start_str}-{end_str})")
                
                for day in sorted(dept_by_day.keys()):
                    print(f"    Day {day}: {', '.join(dept_by_day[day])}")
        
        return len(multi_day_applicants)


def main():
    """Main function."""
    print("GIS Taiwan Interview Scheduling - Optimized Heuristic")
    print("=" * 50)
    
    try:
        scheduler = InterviewSchedulerOptimized('availability_records.csv', 'departmentDuration.csv')
        
        if scheduler.solve_heuristic():
            scheduler.analyze_results()
            scheduler.export_schedule()
        else:
            print("Could not find any feasible schedule!")
            
    except FileNotFoundError:
        print("Error: availability_records.csv not found!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()