#!/usr/bin/env python3
"""
Naive Interview Scheduling Algorithm
===================================

A very simple naive approach for interview scheduling:
- Read all availability data
- Process availability records in order as they appear
- Try to assign each interview if possible
- Each applicant gets at most one interview per department

Author: GitHub Copilot
Date: May 2025
"""

import pandas as pd
import time
from datetime import datetime
from collections import defaultdict
from heuristic_OriginData import InterviewSchedulerOptimized

class NaiveInterviewScheduler(InterviewSchedulerOptimized):
    def __init__(self, csv_file='availability_records.csv'):
        """Initialize the naive scheduler."""
        # Use parent class methods for data loading and setup
        super().__init__(csv_file)
        
    def reset_solution(self):
        """Reset solution state - override parent to add tracking for naive algorithm."""
        super().reset_solution()
        
        # Additional tracking for naive algorithm
        # Track which (applicant, dept) combinations have been assigned
        self.assigned_combinations = set()    
    
    def naive_schedule(self):
        """
        Simple naive scheduling algorithm:
        - Process availability records in order
        - Try to schedule each one if possible
        - Each applicant gets at most one interview per department
        """
        print("Starting naive scheduling...")
        
        total_assigned = 0
        
        for _, row in self.df.iterrows():
            applicant = row['ID']
            dept = row['dept']
            k = self.parse_date_to_k(row['date'])
            
            if k is None:
                continue
                
            # Skip if this (applicant, dept) combination is already assigned
            if (applicant, dept) in self.assigned_combinations:
                continue
            
            try:
                start_str, end_str = row['time_slot'].split('-')
                available_start = self.parse_time_to_minutes(start_str)
                available_end = self.parse_time_to_minutes(end_str)
                
                # Try to schedule at the earliest possible time in this slot
                duration = self.interview_duration[dept]
                
                # Try every minute within the available time range
                for start_time in range(available_start, available_end - duration + 1):
                    end_time = start_time + duration
                    
                    # Check if we can schedule here
                    if self.can_schedule_interview(applicant, dept, k, start_time, end_time):
                        self.assign_interview(applicant, dept, k, start_time)
                        self.assigned_combinations.add((applicant, dept))
                        total_assigned += 1
                        print(f"Assigned: Applicant {applicant} ({self.applicant_names[applicant]}) "
                              f"for {dept} on day {k} at {self.minutes_to_time_str(start_time)}")
                        break  # Move to next availability record
                        
            except Exception as e:
                print(f"Error processing record: {e}")
                continue
        
        print(f"\nNaive scheduling completed: {total_assigned} interviews assigned total")
        return total_assigned
    
    def analyze_results(self):
        """Analyze and display the scheduling results."""
        total_interviews = len(self.schedule)
        
        print("\n" + "="*50)
        print("NAIVE SCHEDULING RESULTS")
        print("="*50)
        print(f"Total interviews scheduled: {total_interviews}")
        
        # Calculate maximum possible interviews (unique applicant-department combinations)
        unique_applicant_dept_combinations = set()
        for (applicant, dept, date_k) in self.available_times.keys():
            unique_applicant_dept_combinations.add((applicant, dept))
        max_possible_interviews = len(unique_applicant_dept_combinations)
        print(f"Maximum possible interviews: {max_possible_interviews}")
        
        # Count by department
        dept_counts = defaultdict(int)
        for _, dept, _, _, _ in self.schedule:
            dept_counts[dept] += 1
        
        print("\nInterviews by department:")
        for dept in self.departments:
            print(f"  {dept}: {dept_counts[dept]} interviews")
        
        # Count by date
        date_counts = defaultdict(int)
        for _, _, date_k, _, _ in self.schedule:
            date_counts[date_k] += 1
            
        print("\nInterviews by date:")
        for date_k in sorted(date_counts.keys()):
            print(f"  Day {date_k}: {date_counts[date_k]} interviews")
        
        # Count unique applicants
        unique_applicants = len(set(applicant for applicant, _, _, _, _ in self.schedule))
        print(f"\nUnique applicants with interviews: {unique_applicants}")
        
        # Check for duplicate (applicant, dept) combinations
        applicant_dept_combinations = [(applicant, dept) for applicant, dept, _, _, _ in self.schedule]
        unique_combinations = len(set(applicant_dept_combinations))
        print(f"Total (applicant, dept) combinations: {unique_combinations}")
        if len(applicant_dept_combinations) != unique_combinations:
            print("WARNING: Found duplicate (applicant, dept) combinations!")
        else:
            print("✓ No duplicate (applicant, dept) combinations found")
        
        # Analyze multi-day applicants (using inherited method from heuristic)
        multi_day_count = self.analyze_multi_day_applicants()
        
        return total_interviews
    
    def export_schedule(self, filename='naive_schedule.csv'):
        """Export the schedule to a CSV file."""
        if not self.schedule:
            print("No schedule to export")
            return
            
        # Convert schedule to DataFrame
        schedule_data = []
        for applicant, dept, date_k, start_time, end_time in self.schedule:
            schedule_data.append({
                'Applicant_ID': applicant,
                'Applicant_Name': self.applicant_names[applicant],
                'Department': dept,
                'Date_K': date_k,
                'Start_Time': self.minutes_to_time_str(start_time),
                'End_Time': self.minutes_to_time_str(end_time),
                'Duration_Minutes': end_time - start_time
            })
        
        df_schedule = pd.DataFrame(schedule_data)
        df_schedule.to_csv(filename, index=False)
        print(f"Schedule exported to: {filename}")

def main():
    """Main function."""
    print("Naive Interview Scheduling Algorithm")
    print("=" * 40)
    
    try:
        # Initialize scheduler
        scheduler = NaiveInterviewScheduler('availability_records.csv')
        
        # Reset and run naive scheduling
        scheduler.reset_solution()
        total_assigned = scheduler.naive_schedule()
        
        # Analyze results
        scheduler.analyze_results()
        
        # Export schedule
        scheduler.export_schedule('naive_schedule.csv')
        
        print(f"\nFinal result: {total_assigned} interviews successfully assigned")
        
    except FileNotFoundError:
        print("Error: availability_records.csv not found!")
        print("Please make sure the data file exists in the current directory.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
