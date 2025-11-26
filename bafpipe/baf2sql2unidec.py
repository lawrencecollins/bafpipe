import sqlite3
from ctypes import *
import sys
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from bafpipe import baf2sql



class BafSpectrum():

    """Contains data from a baf file
    """
    def __init__(self):
        self.spec_vals = {'profile_mz':[], 'profile_int':[],
                          'line_mz':[], 'line_int':[]}
        self.path = ""
        self.xml_parameters_dict = {}
    def open_baf_tic(self, path):
        """_summary_

        Args:
            path (_type_): _description_

        Returns:
            _type_: _description_
        """
        # add checks for .d

        self.path = path
        if sys.version_info.major == 2:
        # note: assuming a european Windows here...
            self.path = path.decode('cp1252')

        self.baf_fn = os.path.join(path, "analysis.baf")
        sqlite_fn = baf2sql.getSQLiteCacheFilename(self.baf_fn)
        self.conn = sqlite3.connect(sqlite_fn)

        # --- Count spectra
        q = self.conn.execute("SELECT COUNT(*) FROM Spectra "
                        "WHERE LineMzId NOT NULL AND ProfileMzId NOT NULL")
        row = q.fetchone()
        N = row[0]
        print("Specified BAF has {} spectra with line and profile data.".format(N))

        # --- Plot TIC and BPC over MS^1 spectra
        q = self.conn.execute("SELECT Rt, SumIntensity, MaxIntensity FROM Spectra s "
                        "JOIN AcquisitionKeys ak ON s.AcquisitionKey = ak.Id "
                        "WHERE ak.MsLevel = 0 "
                        "ORDER BY s.ROWID")
        self.data = [ row for row in q ]
        self.rt = [ row[0] for row in self.data ]
        self.tic = [ row[1] for row in self.data ]
        self.bpc = [ row[2] for row in self.data ]

        return self.rt, self.tic, self.bpc
    def rt_iter(self, baf_fn=None, conn = None, rt = None, scanstart = None, scanend = None):
        if conn == None:
            conn = self.conn
        if baf_fn is None:
            baf_fn = self.baf_fn
        if rt == None:
            rt = self.rt
        if scanstart is not None and scanend is not None:
            rt = rt[scanstart:scanend]
        elif scanstart is not None:
            rt = rt[scanstart:]

        alldata = []
        scans = []
        for n, i in enumerate(rt):
            q = conn.execute("SELECT LineMzId, LineIntensityId, ProfileMzId, ProfileIntensityId FROM Spectra "
                        "WHERE ABS(Rt - {}) < 1e-8".format(i))
            row = q.fetchone()

            bs = baf2sql.BinaryStorage(baf_fn)

            if not all(row) == False: # check for None values

                bs = baf2sql.BinaryStorage(baf_fn)

                profile_mz = np.array(bs.readArrayDouble(row[2]))
                profile_int = np.array(bs.readArrayDouble(row[3]))

                scan = np.transpose([profile_mz, profile_int])
                alldata.append(scan)
                scans.append(n)
        return alldata, scans







    def extract_scans(self, scanstart=None, scanend=None, rt = None,
                      conn = None, baf_fn = None, mean=True):
        if rt is None:
            rt = self.rt
        if conn is None:
            conn = self.conn
        if baf_fn is None:
            baf_fn = self.baf_fn
        if scanstart is not None:
            rt = rt[scanstart:scanend]
        # if scanend is not None:
        #     rt = rt[:scanend]

        for i in rt:
            q = conn.execute("SELECT LineMzId, LineIntensityId, ProfileMzId, ProfileIntensityId FROM Spectra "
                        "WHERE ABS(Rt - {}) < 1e-8".format(i))
            row = q.fetchone()

            bs = baf2sql.BinaryStorage(baf_fn)

            if not all(row) == False: # check for None values

                bs = baf2sql.BinaryStorage(baf_fn)

                profile_mz = bs.readArrayDouble(row[2])
                profile_int = bs.readArrayDouble(row[3])

                self.spec_vals['profile_mz'].append(profile_mz)
                self.spec_vals['profile_int'].append(profile_int)


                line_mz = bs.readArrayDouble(row[0])
                line_int = bs.readArrayDouble(row[1])

                self.spec_vals['line_mz'].append(line_mz)
                self.spec_vals['line_int'].append(line_int)

        # convert spectra into arrays and average
        self.profile_mz = np.array(self.spec_vals['profile_mz']).mean(axis=0)
        self.profile_int = np.array(self.spec_vals['profile_int']).mean(axis=0)
        # self.line_mz = np.array(self.spec_vals['line_mz']).mean(axis=0)
        # self.line_int = np.array(self.spec_vals['line_int']).mean(axis=0)

        # transpose profile spectra
        self.data2 = np.transpose([self.profile_mz, self.profile_int])

        return self.data2

    def export_scans_from_file(self, path, scanstart=None, scanend=None, name = None):

        self.scanstart = scanstart
        self.scanend = scanend
        if name is None:
            directory, name = os.path.split(path)
        self.name = name

        self.open_baf_tic(path)
        data = self.extract_scans(scanstart=scanstart, scanend=scanend)
        self.data = data
        return self.name, data
    
    def extract_xml_parameters(self, subdirectory_path = None, parameter_names="CreationDateTime", UserColumnNames = True):
        """
        Extracts parameters, including custom UserColumnNames and their values,
        from XML file(s) in a single subdirectory and updates the instance
        attribute self.xml_parameters_dict.

        The extracted data is stored as:
        {subdirectory_name: {param1: value1, param2: value2, custom_col1: value, ...}}

        Args:
            subdirectory_path (str): Path to the specific subdirectory containing XML file(s).
            parameter_names (list): List of XML element or attribute names to extract.

        Returns:
            bool: True if parameters were extracted and the object was updated, False otherwise.
        """
        if subdirectory_path is None:

            subdirectory_path = self.path

        # 1. Initialize result dictionary for the single subdirectory
        subdirectory_name = os.path.basename(subdirectory_path)
        row = {}

        if not os.path.isdir(subdirectory_path):
            print(f"Error: Path is not a valid directory: {subdirectory_path}")
            for param_name in parameter_names:
                row[param_name] = None
            self.xml_parameters_dict[subdirectory_name] = row
            return False

        xml_found = False

        # 2. Look for XML files in the single subdirectory
        for file in os.listdir(subdirectory_path):
            if file.endswith('.xml'):
                xml_path = os.path.join(subdirectory_path, file)
                xml_found = True

                try:
                    if UserColumnNames:
                        # Parse the XML file
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
        
                        # --- NEW LOGIC: Extract Custom UserColumnNames and Values ---
                        custom_columns = {}
                        user_column_names_element = root.find('.//UserColumnNames')
                        user_data_element = root.find('.//UserData')
        
                        if user_column_names_element is not None and user_data_element is not None:
                            # Map ColumnNumber to Name from UserColumnNames
                            column_map = {}
                            for col_elem in user_column_names_element.findall('Column'):
                                col_num = col_elem.get('ColumnNumber')
                                col_name = col_elem.get('Name')
                                if col_num is not None and col_name is not None:
                                    column_map[col_num] = col_name
        
                            # Map Value to Name using the map from UserData
                            for data_elem in user_data_element.findall('Column'):
                                col_num = data_elem.get('ColumnNumber')
                                col_value = data_elem.get('Value')
        
                                if col_num in column_map:
                                    col_name = column_map[col_num]
                                    custom_columns[col_name] = col_value
                        
                        # Add the extracted custom columns to the row dictionary
                        row.update(custom_columns)
                    # ----------------- END OF NEW LOGIC ------------------------

                    # Search for each specified (non-custom) parameter
                    for param_name in parameter_names:
                        if param_name not in row: # Only search if it hasn't been found/set yet
                            # First, search for the parameter as an element (path-based)
                            param_elem = root.find(f'.//{param_name}')
                            if param_elem is not None and param_elem.text:
                                row[param_name] = param_elem.text
                                continue

                            # If not found as element, search for it as an attribute
                            # This checks all elements for an attribute named param_name
                            for elem in root.iter():
                                if param_name in elem.attrib:
                                    row[param_name] = elem.attrib[param_name]
                                    break

                    # Check if we have found all *mandatory* parameters (original list + custom columns)
                    # NOTE: This part of the logic is kept from your original code but will only work
                    # if you can define a complete list of all expected parameters (including custom ones)
                    # in `parameter_names` or if you adjust the check.
                    # For robustness, I'll remove the `break` condition as the custom columns
                    # are dynamically named and might not be in `parameter_names`.

                except ET.ParseError as e:
                    print(f"Warning: Could not parse {xml_path}: {e}")
                except Exception as e:
                    print(f"Warning: Error reading {xml_path}: {e}")

        if not xml_found:
            print(f"Error: No XML files found in {subdirectory_path}")
            return False

        # 3. Add None for any originally requested parameters not found (excluding custom ones)
        for param_name in parameter_names:
            if param_name not in row:
                row[param_name] = None

        # 4. Update the class object's attribute
        self.xml_parameters_dict[subdirectory_name] = row

        return self.xml_parameters_dict is not None

    def plot_tic(self, rt=None, tic=None, name = None, show_scans = False):
        if rt == None:
            rt = self.rt
        if tic == None:
            tic = self.tic
        if name == None:
            name = self.name
        plt.plot(rt, tic)
        if show_scans == True:
            plt.axvspan(rt[self.scanstart], rt[self.scanend])
        plt.title(name)
        plt.xlabel("Time")
        plt.show()
