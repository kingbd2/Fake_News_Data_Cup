import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def log_model_metrics(parquet_file, metrics=1):
    try:
        table = pq.read_table(parquet_file)
        print(table.schema)
    except:
        print("There is no parquet file with that name")
    finally:
        print("Finished logging")