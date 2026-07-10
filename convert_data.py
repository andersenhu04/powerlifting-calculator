import pandas as pd

# Load only the columns we absolutely need
cols = ['Sex', 'Equipment', 'BodyweightKg', 'Best3SquatKg', 
        'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg', 'Federation', 'Tested', 'Dots']

df = pd.read_csv('C:\\Users\\a_hu2\\OneDrive\\Desktop\\powerlifting-app\\openpowerlifting-2026-07-04-9acfa1cf.csv', usecols=cols, low_memory=False)

# Filter for FULL POWER (SBD) competitors only to kill that spike on the left
df = df.dropna(subset=['TotalKg', 'Sex', 'Equipment', 'BodyweightKg', 
                       'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg'])

# Save the lightweight version
df.to_parquet('optimized_data.parquet')
print(f"Data saved! Row count: {len(df)}")