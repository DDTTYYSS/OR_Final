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
            segments = [seg.strip() for seg in re.split('[,;]', str(avail_str)) if seg.strip()]
            for i in range(0, len(segments), 2):
                date_seg = segments[i]
                time_seg = segments[i+1] if i+1 < len(segments) else ''
                date_match = re.search(r'(\d{1,2}/\d{1,2})', date_seg)
                time_match = re.search(r'(\d{1,2}:\d{2}-\d{1,2}:\d{2})', time_seg)
                if date_match and time_match:
                    date_iso = pd.to_datetime('2024/' + date_match.group(1)).date().isoformat()
                    time_range = time_match.group(1)
                    records.append({
                        'ID': applicant_id,
                        'Name': applicant_name,
                        'dept': dept_code,
                        'date': date_iso,
                        'time_slot': time_range
                    })

# Build the cleaned availability DataFrame
avail_df = pd.DataFrame(records)

# Save to CSV
output_path = 'availability_records.csv'
avail_df.to_csv(output_path, index=False)

print(f"Saved cleaned CSV to {output_path}")
