import sqlite3
import pandas as pd
import os

# Define path to the SQLite database
# Structure found: chembl_36/chembl_36/chembl_36_sqlite/chembl_36.db
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Go up from bioinsight
db_path = os.path.join(base_dir, "chembl_36", "chembl_36", "chembl_36_sqlite", "chembl_36.db")

print(f"Connecting to database at: {db_path}")

if not os.path.exists(db_path):
    print("Error: Database file not found!")
    exit(1)

try:
    conn = sqlite3.connect(db_path)
    print("Connection successful.")

    query = """
    SELECT activity_id, molregno, pchembl_value, standard_type
    FROM activities
    WHERE pchembl_value IS NOT NULL
    LIMIT 5
    """
    
    print("Executing query...")
    df = pd.read_sql(query, conn)
    
    print("\nSample Data:")
    print(df)
    
    # Check for other required tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    required_tables = ['activities', 'assays', 'molecule_dictionary', 'compound_properties', 'target_dictionary']
    print("\nVerifying required tables:")
    for table in required_tables:
        if table in tables:
            print(f"[OK] {table} found")
        else:
            print(f"[MISSING] {table} NOT found")
            
    conn.close()
    print("\nVerification complete. Ready for bioactivity modeling.")

except Exception as e:
    print(f"An error occurred: {e}")
