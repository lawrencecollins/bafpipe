from bafpipe.ms_processing import *


def main():
    if len(sys.argv) < 2:
        print("Usage: python file.py <excel_file>")
        sys.exit(1)

    path = sys.argv[1]cr

    # load the Excel input file
    try:
        df = pd.read_excel(path)
        print("File loaded successfully!")
        
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    eng = BafPipe() # load engine

    try: 
        eng.load_input_file(path, unzip=False, clearhdf5=True, var_ids=True) # load input file and run deconvolution
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        eng.on_unidec() # run deconvolution
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()