import pandas as pd

def remove_duplicates(input_file, output_file):
    """Remove duplicate entries from a CSV file and save to a new file."""
    # Load data from the input CSV file
    data = pd.read_csv(input_file)
    
    # Remove duplicate rows
    cleaned_data = data.drop_duplicates()
    
    # Save the cleaned data to a new CSV file
    cleaned_data.to_csv(output_file, index=False)
    print(f"Removed duplicates. Cleaned data saved to '{output_file}'.")

# Specify your input and output file paths
input_file_path = "combined_file.csv"  # Replace with your input file path
output_file_path = "cleaned_output.csv"  # Replace with your desired output file path

# Run the function to remove duplicates
remove_duplicates(input_file_path, output_file_path)
