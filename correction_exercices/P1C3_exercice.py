import os
import subprocess
import sys

def convert_documents_to_markdown(input_dir, output_dir):
    """
    Converts all the documents in an input directory to Markdown
    in an output directory using the Docling tool via subprocess.

    Args:
        input_dir (str): The path to the directory containing the source documents.
        output_dir (str): The path to the directory where the Markdown files will be saved.
    """
    # Check whether the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: The input directory '{input_dir}' does not exist.")
        return

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Output directory '{output_dir}' created.")
        except OSError as e:
            print(f"Error while creating the output directory '{output_dir}': {e}")
            return

    print(f"Starting the conversion of the documents from '{input_dir}' to '{output_dir}'...")

    # Walk all the files in the input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        # Make sure it is a file (and not a subdirectory)
        if os.path.isfile(input_path):
            # Build the output file name with the .md extension
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}.md"
            output_path = os.path.join(output_dir, output_filename)

            print(f"\nProcessing '{filename}'...")

            # Build the docling command
            # Syntax: docling <input_file> -o <output_file> -f <format>
            # The output format for Markdown is 'md'
            cmd = ['docling', input_path, '-o', output_path, '-f', 'md']
            print(f"Running the command: {' '.join(cmd)}")

            try:
                # Run the docling command
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
                print(f"Success: '{filename}' converted to '{output_filename}'.")
                # Optional: display the docling standard output if needed
                # if result.stdout:
                #     print(f"Docling output: {result.stdout}")

            except subprocess.CalledProcessError as e:
                # Error while running docling
                print(f"Error while converting '{filename}'.")
                print(f"Failed command: {' '.join(e.cmd)}")
                print(f"Return code: {e.returncode}")
                print(f"Error output (stderr): {e.stderr}")
            except FileNotFoundError:
                # Error if the 'docling' command is not found
                print(f"Error: The 'docling' command was not found.")
                print("Check that Docling is correctly installed and that its executable is in the system PATH.")
                print("Installation: pip install docling")
                return # Stop the script if docling is not found
            except Exception as e:
                # Catch any other potential exception
                print(f"An unexpected error occurred while processing '{filename}': {e}")

    print(f"\nConversion finished. The Markdown files are in '{output_dir}'.")

# --- Configuration ---
# Replace these paths with your own
# INPUT_DIRECTORY: The folder where you unzipped 'inputs.zip'
# OUTPUT_DIRECTORY: The folder where you want to save the .md files

INPUT_DIRECTORY = 'inputs'  # Adapt if you unzipped elsewhere
OUTPUT_DIRECTORY = 'markdown_outputs' # Output folder name

# --- Running the script ---
if __name__ == "__main__":
    # Check whether Python 3 is used (subprocess.run has different arguments in Python 2)
    if sys.version_info < (3, 5):
        print("Error: This script requires Python 3.5 or a later version.")
    else:
        # Give the absolute path to avoid ambiguities
        abs_input_dir = os.path.abspath(INPUT_DIRECTORY)
        abs_output_dir = os.path.abspath(OUTPUT_DIRECTORY)
        convert_documents_to_markdown(abs_input_dir, abs_output_dir)
