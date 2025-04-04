import pandas as pd

def load_and_clean_data(data_path):
    try:
        # Import the dataset
        df = pd.read_csv(data_path)
        print(f"✅ Dataset loaded successfully from: {data_path}")
    except FileNotFoundError:
        print(f"❌ File not found at: {data_path}")
        raise
    except pd.errors.EmptyDataError:
        print("❌ The file is empty.")
        raise
    except Exception as e:
        print(f"❌ An error occurred while loading the data: {e}")
        raise

    try:
        # Replace missing basement values to 0
        df['basement'] = df['basement'].fillna(0)

        # Change the basement type to integer
        df['basement'] = df['basement'].astype(int)

        # Drop the row with lot_size = 1220551
        df = df[df.lot_size < 500000]

        print("✅ Data cleaned successfully.")
        return df

    except KeyError as ke:
        print(f"❌ Missing expected column: {ke}")
        raise
    except Exception as e:
        print(f"❌ Error during data cleaning: {e}")
        raise
