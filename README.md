# BafPipe
Automated deconvolution of mass spectra datasets using [UniDec](https://github.com/michaelmarty/UniDec).

Cite Marty et al. Anal. Chem. 2015. DOI: 10.1021/acs.analchem.5b00140

Automates the UniDec deconvolution algorithms for large Bruker mass spectra datasets and quantifies relative peak intensity. Converts Bruker .baf to numpy arrays using baf2sql and then automates unidec to deconvolute protein mass spectra and quantify species. 

`pip install BafPipe`

The software uses an input excel file containing deconvolution parameters, species identities/masses and data plotting parameters. 

### Setting up the input file

Sheet 1 of input file contains the directory folder of your mass spectra and the unidec configuration parameters. 

You can also add masses to detect and a corresponding colour. These are done by stipulating **Species** + *name* and **Color** + *name* in the Parameter column, followed by the desired mass or colour in the Input column.


**Example input directory**

| Parameter | Input | Comments |
| --- | --- |--- |
|Directory|D:\mass spec\protein labelling|
|Species Protein| 48127|
|Color Protein| orange|
|Start Scan|	490	|
|End Scan|540	|
|Tolerance (Da)	|10	|Peak matching tolerance |
|Config masslb	|15000|	Deconvolution window low mass|
|Config massub	|50000|	Deconvolution window high mass|
|Config massbins|	1|Mass bins for deconvolution - sample mass every|
|Config peakwindow|	10|	|
|Config minmz|	700|	m/z lower bounds (defaults to 0)|
|Config maxmz|		|m/z upper bounds (defaults to 10e12)|
|Config startz|	1	| |
|Config endz|	100	| |
|Config numz|	100	| |
|Config numit|	60	|number of iterations of deconvolution algorithm|

Each file within the directory can be linked to custom variables defined in a second sheet of the input directory. This comes in handy if wanting to filter data or perform analyses/comparison on subsets of your experiment. 

Any column names can be defined aside from 'Name' in column 0. Name corresponds to your filename (can be partial match).

Make sure var_ids=True ```load_input_file(var_ids = True)```  

**Example variables table**

|Name|	var1	|var2|var3|
| --- | --- | --- | --- |
|Nexp 1	|1	|1	|1|
|Nexp 2	|2	|13	|3|
|Nexp 3	|1	|25	|3|
|Nexp 4	|3	|25	|1|
|Nexp 5	|2	|13	|5|

### Running the code
```

from bafpipe.meta_processing import *
from bafpipe import ms_plotter_tools as msp

path = r"C:\users\input_file.xlsx

eng = Meta2()

eng.load_input_file(path, unzip=False, clearhdf5=True, var_ids=True) # load input file and run deconvolution

eng.on_unidec()

spectra = eng.eng.data.spectra

msp.plot_spectra_separate(spectra,
attr = "massdat",
xlabel = "Mass [Da]", 
export=True,
c='black',
lw=0.7,
window=[None, None],
show_peaks=True,
legend=True,
directory=eng.directory,
fmt='png') # fmt = 'svg' for vector format

```




