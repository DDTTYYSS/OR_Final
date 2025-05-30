import pandas as pd
import datetime

# Load the cleaned availability CSV
avail_df = pd.read_csv('availability_records.csv')

# Given session counts per day
S_sizes = {1: 10, 2: 36, 3: 28, 4: 28, 5: 10, 6: 10, 7: 12, 8: 12}

# Department interview durations in minutes
durations = {'GPDA': 20, 'AC': 15, 'PR': 15, 'DM': 25}

# Map dates to day index k (1 to 8) in chronological order
unique_dates = sorted(avail_df['date'].unique(), key=lambda d: datetime.datetime.fromisoformat(d))
date_to_k = {date: idx+1 for idx, date in enumerate(unique_dates)}

# Manually specify day's earliest start time based on the schedule
day_start_map = {
    '2024-09-19': '18:00',
    '2024-09-20': '18:00',
    '2024-09-21': '10:00',
    '2024-09-22': '10:00',
    '2024-09-23': '18:00',
    '2024-09-24': '18:00',
    '2024-09-25': '18:00',
    '2024-09-26': '18:00'
}

# Initialize availability sets
I_list = sorted(avail_df['ID'].unique())
J_list = ['AC', 'GPDA', 'PR', 'DM']
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
    day_start = datetime.datetime.strptime(day_start_map[date], '%H:%M')
    start_ts = datetime.datetime.strptime(start_str, '%H:%M')
    end_ts = datetime.datetime.strptime(end_str, '%H:%M')
    
    offset_start = int((start_ts - day_start).total_seconds() // 60)
    offset_end = int((end_ts - day_start).total_seconds() // 60)
    
    s_start = offset_start // dur + 1
    # Exclude a session if it would start exactly at end time
    # if offset_end % dur == 0:
    #     s_end = offset_end // dur - 1
    # else:
    s_end = offset_end // dur
    
    # Add to set
    for s in range(s_start, s_end+1):
        if 1 <= s <= S_sizes[k]:
            avail_sets[(i, j, k)].add(s)

# Build full 4D DataFrame
records = []
for i in I_list:
    for j in J_list:
        for k in K_list:
            for s in range(1, S_sizes[k] + 1):
                records.append({
                    'ID': i,
                    'dept': j,
                    'k': k,
                    's': s,
                    'available': 1 if s in avail_sets[(i, j, k)] else 0
                })

df_4d_binary = pd.DataFrame(records)

# Save to CSV
output_path = 'availability_4d_binary.csv'
df_4d_binary.to_csv(output_path, index=False)

output_path
