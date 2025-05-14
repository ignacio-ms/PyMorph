import pandas as pd


def csv2excel(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Save to Excel file
    df.to_excel(file_path.replace('.csv', '.xlsx'), index=False)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Convert CSV files to Excel format.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    args = parser.parse_args()

    # Check if the input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: The file {args.input_file} does not exist.")
        exit(1)

    # Convert CSV to Excel
    csv2excel(args.input_file)
    print(f"Converted {args.input_file} to Excel format.")
