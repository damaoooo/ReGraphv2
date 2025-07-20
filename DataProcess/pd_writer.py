from multiprocessing import Queue
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

class PDWriter:
    def __init__(self, queue: Queue, output_path: str, bin_size: int = 1000, start_index: int = 0):
        self.queue = queue
        self.writer = None
        self.output_path = output_path
        self.bin_size = bin_size
        self.processed_count = 0
        self.file_index = start_index // bin_size
        
        self.progress_file_path = os.path.join(output_path, "progress.txt")
        self.temp_progress = []
        self.last_file_name = None

        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def start(self):
        try:
            while True:
                data = self.queue.get()
                if isinstance(data, str) and data == "STOP":
                    # Write the last file if there is any data left
                    if self.writer is not None:
                        self.writer.close()
                    last_file_name = f"data_{self.file_index}.parquet.tmp"
                    if os.path.exists(last_file_name):
                        os.rename(last_file_name, os.path.join(self.output_path, f"data_{self.file_index}.parquet"))
                    # Write the progress file
                    with open(self.progress_file_path, 'a') as f:
                        f.write("\n".join(self.temp_progress))
                    break
                self.write_data(data)
        finally:
            if self.writer is not None:
                self.writer.close()


            
    def write_data(self, data):
        file_name = f"data_{self.file_index}.parquet"
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        
        if self.writer is None:
            self.writer = pq.ParquetWriter(file_name+".tmp", table.schema, compression='snappy')
        
        self.writer.write_table(table)
        self.processed_count += len(data)
        self.temp_progress.extend(table['file_path'].to_pylist())
        
        if self.processed_count >= self.bin_size:
            self.writer.close()
            self.writer = None
            self.processed_count = 0
            
            # Save progress
            self.last_file_name = file_name
            os.rename(file_name + ".tmp", os.path.join(self.output_path, file_name))
            # Write the file_path into progress file
            with open(self.progress_file_path, 'a') as f:
                f.write("\n".join(self.temp_progress) + "\n")
            self.temp_progress = []
            self.file_index += 1
