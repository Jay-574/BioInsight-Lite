# notebooks/0_data_explore.py
import pandas as pd
import h5py
import numpy as np
import pickle
import sys

# Note: pytables (required for pd.HDFStore) is not available for Python 3.14 on Windows.
# Using h5py to explore the file instead.

FILE_PATH = 'chembl_36.h5'

def explore_h5():
    try:
        print(f"Opening '{FILE_PATH}'...")
        with h5py.File(FILE_PATH, 'r') as f:
            keys = list(f.keys())
            print("Keys available in HDF5 file:", keys)

            # 1. Decode Configuration/Metadata
            if 'config' in keys:
                print("\n--- File Configuration (Metadata) ---")
                try:
                    conf_data = f['config'][()]
                    for i, item in enumerate(conf_data):
                        # item is likely a numpy array of bytes (pickled data)
                        if hasattr(item, 'tobytes'):
                            b = item.tobytes()
                            # Check for pickle protocol 5 header (0x80 ...)
                            if len(b) > 0 and b[0] == 128:
                                try:
                                    obj = pickle.loads(b)
                                    print(f"  Config Item {i}: {obj}")
                                except:
                                    pass
                except Exception as e:
                    print(f"Could not read 'config': {e}")
            
            # 2. Inspect 'fps' dataset (Fingerprints)
            if 'fps' in keys:
                ds = f['fps']
                print("\n--- Fingerprint Dataset ('fps') ---")
                print(f"Shape: {ds.shape}")
                print(f"Dtype: {ds.dtype}")
                print(f"Chunking: {ds.chunks}")
                print(f"Compression: {ds.compression}")
                
                try:
                    print("Attempting to read first element of 'fps'...")
                    # Try reading a single element
                    print("First element:", ds[0])
                except Exception as e:
                    print(f"Error reading 'fps' data: {e}")
                    print("Note: This error usually indicates an issue with file integrity or platform-specific HDF5 driver limitations.")

            # 3. Inspect Index Group
            if '_i_fps' in keys:
                print("\n--- Index Group ('_i_fps') ---")
                grp = f['_i_fps']
                print(f"Subkeys: {list(grp.keys())}")
                if 'popcnt' in grp:
                    pop_obj = grp['popcnt']
                    if isinstance(pop_obj, h5py.Group):
                        print(f"  'popcnt' is a Group containing: {list(pop_obj.keys())}")
                        # Example: Read 'ranges'
                        if 'ranges' in pop_obj:
                            try:
                                rng = pop_obj['ranges']
                                print(f"  Sample 'ranges' data:\n{rng[:5]}")
                            except:
                                pass

    except FileNotFoundError:
        print(f"Error: '{FILE_PATH}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    explore_h5()
