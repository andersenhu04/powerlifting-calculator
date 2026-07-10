import pandas as pd

# Load CSV
# Make sure to adjust the filename if it has changed!
print("Reading CSV...")
df = pd.read_csv('openpowerlifting-2026-07-04-9acfa1cf.csv', low_memory=False)

# Save as a compressed Parquet file
print("Saving as Parquet...")
df.to_parquet('powerlifting_data.parquet', compression='snappy')

print("Done! You can now use 'powerlifting_data.parquet' in your app.")