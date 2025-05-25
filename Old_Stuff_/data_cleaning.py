import pandas as pd
import re

# Load the raw CSV file
df = pd.read_csv('OR_Final 資料 - 工作表3.csv')

# Define department mappings
dept_map = {'學術': 'AC', '公關': 'PR', '國際': 'GPDA', '行銷': 'DM'}
dept_keys = ['學術', '公關', '國際', '行銷']

# Identify availability columns for first and second choices
avail_cols_first = list(df.columns[3:7])
avail_cols_second = list(df.columns[8:12])
col_to_chinese_first = dict(zip(avail_cols_first, dept_keys))
col_to_chinese_second = dict(zip(avail_cols_second, dept_keys))

# Parse availability strings into long format
records = []
for _, row in df.iterrows():
    applicant_id = row['ID']
    applicant_name = row['姓名 Name']
    
    # First and second choice availabilities
    for col_map, cols in [(col_to_chinese_first, avail_cols_first), (col_to_chinese_second, avail_cols_second)]:
        for col in cols:
            avail_str = row[col]
            if pd.isna(avail_str):
                continue
                
            dept_key = col_map[col]
            dept_code = dept_map[dept_key]
            
            if dept_code == 'DM':
                # Parse DM format: "9/19（四 Thu） 19:00-21:00, 9/20（五 Fri） 19:00-21:00"
                pattern = r'(\d{1,2}/\d{1,2})（[^）]+）\s*(\d{1,2}:\d{2}-\d{1,2}:\d{2})'
                matches = re.findall(pattern, str(avail_str))
                for date, time in matches:
                    date_iso = pd.to_datetime('2024/' + date).date().isoformat()
                    records.append({
                        'ID': applicant_id,
                        'Name': applicant_name,
                        'dept': dept_code,
                        'date': date_iso,
                        'time_slot': time
                    })
            else:
                # Parse AC/GPDA/PR format: "六  Saturday 9/21, 13:00-17:00"
                pattern = r'[^,]+?(\d{1,2}/\d{1,2}),\s*(\d{1,2}:\d{2}-\d{1,2}:\d{2})'
                matches = re.findall(pattern, str(avail_str))
                for date, time in matches:
                    date_iso = pd.to_datetime('2024/' + date).date().isoformat()
                    records.append({
                        'ID': applicant_id,
                        'Name': applicant_name,
                        'dept': dept_code,
                        'date': date_iso,
                        'time_slot': time
                    })

# Build the cleaned availability DataFrame
avail_df = pd.DataFrame(records)

# Save to CSV
output_path = 'availability_records.csv'
avail_df.to_csv(output_path, index=False)

print(f"Saved cleaned CSV to {output_path}")

# Print some sample records to verify
print("\nSample records:")
print(avail_df.head())
