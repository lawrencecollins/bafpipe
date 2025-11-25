from bafpipe.ms_processing import *
from bafpipe import ms_plotter_tools as msp



def pipe(path,unzip=False, clearhdf5=True, var_ids=True, plot = True):
    eng = BafPipe()
    eng.load_input_file(path, unzip=unzip, clearhdf5=clearhdf5, var_ids=var_ids)
    eng.on_unidec() 
    if plot:

        msp.plot_spectra_separate(eng.eng.data.spectra, eng.species, attr = "massdat", xlabel = "Mass [Da]", 
                        export=True, c='black',lw=0.7,window=[None, None],
                        show_peaks=False,legend=False, directory =eng.directory, fmt='png')
    return eng


def main():
    if len(sys.argv) < 2:
        print("Usage: python file.py <excel_file>")
        sys.exit(1)

    path = sys.argv[1]
    eng = BafPipe()
    # load the Excel input file
    try:
            
        eng.load_input_file(path, unzip=True, clearhdf5=True, var_ids=False)
     
        print("File loaded successfully!")
        
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    try: 
        eng.on_unidec() # load input file and run deconvolution
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    main()