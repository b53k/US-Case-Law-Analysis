"""
This script processes dataset from huggingface and aggregates case counts per U.S. jurisdiction (State or Federal)
and year, storing the results in a CSV file.

Output:
    A csv file with the following structure (Example only):
    ```
    Jurisdiction, Year, Cases
    Georgia, 1999, 490
    Wyoming, 1878, 114
    ```
"""

import os
import gc
import csv
import warnings
import pandas as pd
from tqdm.rich import tqdm

warnings.simplefilter("ignore", category = FutureWarning)
warnings.simplefilter("ignore", category = UserWarning)

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.dirname(script_dir)
data_loc = os.path.join(cwd, 'Processed Data')
output_file = os.path.join(data_loc, "case_count.csv")
raw_data_path = os.path.join(cwd, 'Raw Data')


if not os.path.exists(path = data_loc):
    # Make Folder 'Processed Data'
    os.makedirs(data_loc)

if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
else:
    with open(output_file, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['Jurisdiction', 'Year', 'Case Count'])


# Data to append
rows_to_append = []
files = os.listdir(path = raw_data_path)
batch_rows = 0

try:
    for parquet in tqdm(iterable = files, desc = 'Processing Files', total = len(files), colour = 'green'):
        if not parquet.endswith(".parquet"):
            continue

        loc = os.path.join(raw_data_path, parquet)

        # Parquet to DataFrame
        df = pd.read_parquet(loc, columns = ['jurisdiction', 'decision_date'], engine = 'pyarrow')
        df = df.dropna(subset=['decision_date']) # Remove Missing value
        jurisdiction = df.iloc[0]['jurisdiction'] # Get the current Jurisdiction

        df['year'] = df['decision_date'].astype(str).str.split('-').str[0]  # Create a new 'year' column
        year_counts = df['year'].value_counts().to_dict() 

        for year, count in year_counts.items():
            rows_to_append.append([jurisdiction, year, count])

        # write to csv
        print (f'\U0001f5d2  Logging {jurisdiction}...')

        with open(output_file, 'a', newline = "") as f:
            writer = csv.writer(f)
            writer.writerows(rows_to_append)

        print (f'Finished Logging. {len(df)} cases processed. {len(rows_to_append)} rows logged.\n')

        batch_rows += len(rows_to_append)

        del df
        del rows_to_append 
        gc.collect()

        rows_to_append = []
        

    print (f'\u2705  Done. {batch_rows} rows added for this batch.')

except Exception as e:
    print (f'\u274c Error occured: {e}')