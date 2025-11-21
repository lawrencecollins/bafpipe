from bafpipe.baf2sql2unidec import *
import matplotlib.pyplot as plt
import os
import unidec
from unidec.metaunidec.mudeng import MetaUniDec
from unidec import tools as ud
import pandas as pd
import zipfile
from scipy.signal import find_peaks
from bafpipe import ms_plotter_tools as msp



def unzip_from_dir(directory):
    """Unzips zip folders in dir and deletes zip"""
    zip_files = [x for x in os.listdir(directory) if x[-4:] == ".zip"]
    all_files = [f for f in os.listdir(directory)]
    # unzips into same folder
    for file in zip_files:
        path = os.path.join(directory, file)
        # check for unzipped file and pass
        if not os.path.exists(path[:-4]):
            with zipfile.ZipFile(path,"r") as zip_ref:
                zip_ref.extractall(directory)
            print("Unzipped {}".format(file))
        # os.remove(path)

def filter_df(df, filter_by, column):
    flt=df[column].str.contains(filter_by, na=False)
    return df[flt]

def df_partial_str_merge(df1, df2, on):
    r = '({})'.format('|'.join(df2.Name))
    merge_df = df1.Name.str.extract(r, expand=False).fillna(df1.Name)
    df2=df2.merge(df1.drop('Name', axis=1), left_on='Name', right_on=merge_df, how='outer')
    return df2

class BafPipe():
    def __init__(self):
        self.spectra = []
        self.eng = MetaUniDec()
        self.species = None
        self.tolerance = 10
        self.vars=False
        self.colors_dict=None


    def load_input_file(self, path, unzip = True, getscans=True, clearhdf5=True,
                        var_ids = False):
        self.params=pd.read_excel(path, sheet_name=0)

        try:
            self.conditions = pd.read_excel(path, sheet_name=0)
            if var_ids:
                self.var_ids = pd.read_excel(path, sheet_name=1)
                self.vars = True
        except Exception as e:
            print(e, "no conditions?")
        # try:
        self.get_directory()
        if unzip:
            unzip_from_dir(self.directory)

        if getscans:
            scanstart, scanend = self.get_scans()
        self.upload_spectra(scanstart=scanstart, scanend=scanend)
        self.load_hdf5(clear=clearhdf5)
        self.to_unidec()
        self.update_config()
        self.get_species()
        try:
            self.get_colors()
        except Exception as e:
            print(e)


    def update_config(self, config_table = None):
        self.eng.open(self.hdf5_path)
        self.default_config()
        if config_table is None:
            config_table = filter_df(self.params, 'Config', 'Parameter')
            config_table.loc[:, 'Parameter'] = config_table.loc[:, 'Parameter'].str.replace("Config ", "")
        for i, row in config_table.iterrows():
            print(row[0], row[1])
            if row[1] is not np.nan:
                setattr(self.eng.config, row[0], float(row[1]))
                # print(getattr(self.eng.config, row[0]))


        self.eng.config.write_hdf5()
        return config_table

    def get_scans(self):
        self.scanstart = filter_df(self.params, 'Start Scan', 'Parameter').iloc[0, 1]
        self.scanend = filter_df(self.params, 'End Scan', 'Parameter').iloc[0, 1]
        return self.scanstart, self.scanend

    def get_times(self):
        pass

    def get_species(self, param_table=None):
        if param_table is None:
            param_table=self.params

        seqs = filter_df(param_table, 'Species', 'Parameter')
        seqs.loc[:, 'Parameter']=seqs.loc[:, 'Parameter'].str.replace("Species ", "")
        self.species=seqs.rename(columns={"Parameter":"Species", "Input":"Mass"})

        return self.species

    def get_colors(self):
        param_table = self.params
        seqs = filter_df(param_table, 'Color', 'Parameter')
        seqs.loc[:, 'Parameter']=seqs.loc[:, 'Parameter'].str.replace("Color ", "")
        self.colors_df=seqs.rename(columns={"Parameter":"Species", "Input":"Color"})
        self.colors_df.drop('Comments',axis=1,inplace=True)
        self.colors_dict = pd.Series(self.colors_df.Color.values,index=self.colors_df.Species.values, ).to_dict()
        try:
            self.species = self.species.merge(self.colors_df, on='Species', how='outer')
            self.species = self.species[['Species', 'Mass', 'Color']].dropna(subset=['Mass'])
        except Exception as e:
            print(e)

    def get_directory(self, param_table=None):
        if param_table is None:
            param_table=self.params
        dr=filter_df(param_table, 'Directory', 'Parameter', )
        self.directory=dr.iloc[0, 1]

        return self.directory

    def upload_spectra(self, directory = None, scanstart = None, scanend = None, filetype= '.baf',
                       plot = False, show_scans=False):
        """_summary_

        Args:
            directory (_type_): _description_
            scanstart (_type_, optional): _description_. Defaults to None.
            scanend (_type_, optional): _description_. Defaults to None.
            filetype (str, optional): _description_. Defaults to 'baf'.
        """
        if directory is None:
            directory = self.directory
        else:
            self.directory = directory

        if filetype == '.baf' or filetype ==".d":

            spectra_names = [x for x in os.listdir(directory) if x[-2:] =='.d']

            for s in spectra_names:
                path = os.path.join(directory, s)
                spectrum = BafSpectrum()
                spectrum.export_scans_from_file(path, scanstart, scanend)
                self.spectra.append(spectrum)

                if plot is True:
                    spectrum.plot_tic(show_scans=show_scans)
        if filetype == ".mzml":
            pass


        return self.spectra

    def load_hdf5(self, directory=None, hdf5_name = None, clear = False,
                     ):
        """Generates hdf5 either using name of directory or defined hdf5_name.
            If hdf5 already exists either deletes or directly uploads to UniDec.

        Args:
            directory (_type_): _description_
            hdf5_name (_type_, optional): _description_. Defaults to None.
            clear (bool, optional): _description_. Defaults to False.
        """
        if directory is None:
            directory = self.directory
        if hdf5_name is None:
            hdf5_name = os.path.split(directory)[1]+".hdf5"
        hdf5_path = os.path.join(directory, hdf5_name)
        if clear:
            try:
                os.remove(hdf5_path)
            except Exception as error:
                print(error)
        self.eng.data.new_file(hdf5_path)
        self.hdf5_path = hdf5_path


    def to_unidec(self, spectra = None):
        """_summary_

        Args:
            spectra (_type_, optional): _description_. Defaults to None.
        """
        if spectra is None:
            spectra = self.spectra
        for s in spectra:
            self.eng.data.add_data(s.data2, name = s.name, export=False)
        self.eng.data.export_hdf5()

    def default_config(self, massub = 20000, masslb = 10000, minmz = 600,
                      numit = 50, peakwindow = 10, peaknorm = 0,
                      peakplotthresh = 0.1, peakthresh = 0.01,
                      datanorm = 0, startz = 1, endz = 100, numz = 100):
        """Standard UniDec configuration parameters for AccMass deconvolution.
        Added in variables that require specification e.g. mass window etc.

        Args:
            massub (int, optional): _description_. Defaults to 20000.
            masslb (int, optional): _description_. Defaults to 10000.
            minmz (int, optional): _description_. Defaults to 600.
            numit (int, optional): _description_. Defaults to 50.
        """
        # Parameters
        # UniDec
        self.eng.config.minmz=minmz
        self.eng.config.numit = numit
        self.eng.config.zzsig = 1
        self.eng.config.psig = 1
        self.eng.config.beta = 1
        self.eng.config.startz = startz
        self.eng.config.endz = endz # charge pretty essential to clean deconvolution
        self.eng.config.numz = numz
        self.eng.config.mzsig = 0.85
        self.eng.config.automzsig = 0
        self.eng.config.psfun = 0
        self.eng.config.psfunz = 0
        self.eng.config.autopsfun = 0
        self.eng.config.massub = massub
        self.eng.config.masslb = masslb
        self.eng.config.msig = 0
        self.eng.config.molig = 0
        self.eng.config.massbins = 1
        self.eng.config.adductmass = 1.007276467
        self.eng.config.baselineflag = 1
        self.eng.config.aggressiveflag = 0
        self.eng.config.noiseflag = 0
        self.eng.config.isotopemode = 0
        self.eng.config.orbimode = 0

        # Other
        self.eng.config.mtabsig = 0
        self.eng.config.poolflag = 2
        self.eng.config.nativezub = 1000
        self.eng.config.nativezlb = -1000
        self.eng.config.inflate = 1
        self.eng.config.linflag = 2
        self.eng.config.integratelb = ""
        self.eng.config.integrateub = ""
        self.eng.config.filterwidth = 20
        self.eng.config.zerolog = -12

        self.eng.config.datanorm = 1
        self.eng.config.subbuff=100
        self.eng.config.subtype=2

        # peak picking
        self.eng.config.peakwindow = peakwindow
        self.eng.config.peaknorm = peaknorm
        self.eng.config.peakplotthresh = peakplotthresh
        self.eng.config.peakthresh = peakthresh


        self.eng.config.datanorm = datanorm
        self.eng.config.exnorm = 0

        # update hdf5
        self.eng.config.write_hdf5()

    def on_unidec(self, hdf5_path = None, export_data=True, background_threshold = True, match = True):


        if hdf5_path is None:
            hdf5_path = self.hdf5_path

        try:
            self.eng.open(hdf5_path)
            self.eng.process_data()
            self.eng.run_unidec()
            self.eng.pick_peaks()
        except Exception as error:
            print(error)

        if self.species is not None and match:
            try:
                masslist = list(self.species['Mass'])
                names = list(self.species['Species'])
                self.match_spectra(masslist, names, self.tolerance, background=background_threshold)
                self.export_data()
            except Exception as e:
                print(e)
        # masslist = list(self.species['Mass'])
        # names = list(self.species['Species'])
        # self.match_spectra(masslist, names, self.tolerance, background=background_threshold)
        # self.export_data()

    def generate_peaks(self, spectrum, threshold = 0, tolerance = 10):
        """generates 2D x, y array of found peaks for a 2D x, y array"""

        x,y = spectrum[:,0], spectrum[:,1]
        peaks, _ = find_peaks(y, height = threshold, distance = tolerance)
        peaksx = [x[p] for p in peaksi]
        return np.array([np.array(peaksx), np.array(peaks)])
    
    def get_spectra_peaks(self, spectra = None, threshold = True, tolerance = 10):
        """"""
        if spectra == None: 
            spectra = self.eng.data.spectra

        if threshold:
            threshold = self.background_threshold(s)
        
        for s in spectra:
            if threshold:
                try:
                    threshold = self.background_threshold(s)
                except Exception as error:
                    print(error) 
            else:
                threshold = 0

            try:
                peaks = self.generate_peaks(s, threshold, tolerance)
                s.peaks2 = peaks
            
            except Exception as error:
                print("{} Peak picking error".format(s.name), error)

            




    def background_threshold(self, spectrum, binsize = 10):

        # flatten spectrum into bins of binsize (e.g. 20)
        new_size = (spectrum.massdat[:,1].flatten().shape[0]//binsize+1)*binsize
        resamp = np.resize(spectrum.massdat[:,1],new_size)
        resamp =resamp.reshape((binsize,-1))

        # calculate noise threshold from mean of the median peak within each bin
        peak_thresh = np.mean(np.median(np.max(resamp,axis=0)))

        return peak_thresh

    def background_threshold_spectra(self, threshold = 'background_threshold', binsize = 10):

        for s in self.eng.data.spectra:
            if threshold == 'background_threshold':
                s.background_threshold = self.background_threshold(s, binsize)
                print("Done thresholds")
            else :
                try:
                    s.background_threshold = threshold
                    print("Done thresholds")
                except Exception as error:
                    print("No threshold", error)

    def match(self,pks, masslist, names, tolerance):
        """matches peaks (y axis) to corresponding mass (x axis)

        Args:
            pks (_type_): _description_
            masslist (_type_): _description_
            names (_type_): _description_
            tolerance (_type_): _description_

        Returns:
            _type_: _description_
        """
        matches = []
        errors = []
        peaks = []
        nameslist = []

        for p in pks:
            target = p.mass
        #     print(target)
            nearpt = ud.nearestunsorted(masslist, target)

            match = masslist[nearpt]
            error = target-match
            if np.abs(error) < tolerance:
                name = names[nearpt]
                p.error = error
            else:
                name = ""
            p.label = name
            p.match = match
            p.matcherror = error

            if self.colors_dict is not None:
                if p.label in self.colors_dict.keys():
                    p.color = self.colors_dict[p.label]

            matches.append(match)
            errors.append(error)
            peaks.append(target)
            nameslist.append(name)

        matchlist = [peaks, matches, errors, nameslist]
        return matchlist

    def match_spectra(self, masslist, names, tolerance, background = True):

        if background:
            self.background_threshold_spectra()
        else:
            for s in self.eng.data.spectra:
                s.background_threshold = 0

        for s in self.eng.data.spectra:
            self.match(s.pks.peaks, masslist, names, tolerance,
                        background_threshold = s.background_threshold)



    def group_spectra(self, groupby):
        df1 = pd.DataFrame([s.name for s in self.eng.data.spectra], columns=['Name'])
        df2 = self.var_ids

        r = '({})'.format('|'.join(df2.Name))
        merge_df = df1.Name.str.extract(r, expand=False).fillna(df1.Name)
        merge_df
        df2=df2.merge(df1, left_on='Name', right_on=merge_df, how='outer')

        grouped = {}
        for n, d in df2.groupby(groupby):
            grouped[n] = [s for s in self.eng.data.spectra if s.name in list(d.Name_y)]

        return grouped


    def export_data(self, export = True, conditions_input = "", name = None):
        dfs = []
        for s in self.eng.data.spectra:

        # export massdat to unidec folder
            arraypath = os.path.join(self.directory, "UniDec_Figures_and_Files", s.name+"_massdat.txt")
            np.savetxt(arraypath, s.massdat)
            # export figs to unidec folder
            # msp.plot_spectra()

            counter = 0
            label = []
            mass = []
            height = []
            color = []
            for p in s.pks.peaks:

                if p.label !="" :
                    label.append(p.label)
                    mass.append(p.mass)
                    height.append(p.height)
                    color.append(p.color)
                    counter = counter+1
            s_name = [s.name]*counter

            dct = {"Label":label, "Mass":mass, "Height":height, "Name":s_name, "Color":color}
            df = pd.DataFrame(dct)
            df['Percentage_Labelling'] = (df.Height/df.Height.sum())*100
            dfs.append(df)
        results_df = pd.concat(dfs)

        if self.vars:
            try:
                results_df = df_partial_str_merge(results_df,self.var_ids,'Name')
            except Exception as e:
                print("check vars", e)

        self.results1 = results_df

        results2 = pd.pivot(results_df, index='Name', columns='Label', values = ['Height', 'Percentage_Labelling']).fillna(0)
        results2.reset_index(inplace=True)

        if self.vars:
            try:
                results2.columns = results2.columns.droplevel(0)
                results2.reset_index(inplace=True, drop=True)
                results2.rename(columns = {"" : "Name"}, inplace = True)
                results2 = df_partial_str_merge(results2,self.var_ids,'Name')
            except Exception as e:
                print("check vars", e)

        if name is None:
            name = os.path.split(self.directory)[1]+"_results.xlsx"
        path = os.path.join(self.directory,name)
        results2.to_excel(path)

        self.results_df = results2
        return results2





    def plot_spectra(self, export = True, combine = False, data = 'massdat',
                    window = [None, None], cmap='gray',title=None,show_titles=False,
                    show_peaks=False, xlabel='Mass (Da)',c='black',
                    lw=0.7,groupby=None
                     ):
        spectra = self.eng.data.spectra
        if combine and groupby is None:
            if title is None:
                title = os.path.split(self.directory)[-1]
            msp.plot_spectra_combined(spectra, directory = self.directory,
                                      cmap=cmap,title=title,show_titles=show_titles,
                                      show_peaks=show_peaks, window=window)
        elif combine and groupby is not None:
            if type(groupby) != list:
                groupby = [groupby]
            grouped = self.group_spectra(groupby)
            for var, spectra in grouped.items():
                title = " ,".join([name + " " + str(v) for name, v in zip(groupby, var)])
                msp.plot_spectra_combined(spectra, directory = self.directory,
                                      cmap=cmap,title=title,show_titles=show_titles,
                                      show_peaks=show_peaks, window=window)
        else:
            msp.plot_spectra_separate(spectra, directory=self.directory, export=export, attr=data, window=window, xlabel=xlabel,
                                      c=c, lw=lw
                                       )



if name == "__main__":

    # run test 
    input_file = r"c:\Users\chmlco\OneDrive - University of Leeds\RESEARCH\BafPipe test\Input file\BafPipe test.xlsx"
    eng = BafPipe()
    eng.load_input_file(input_file, unzip=False, clearhdf5=True, var_ids=True)
    eng.on_unidec()
    
