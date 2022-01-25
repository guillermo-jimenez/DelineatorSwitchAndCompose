import os
import os.path

from typing import Tuple
from numpy import ndarray
from numpy import loadtxt
from numpy import fromstring
from numpy import zeros
from numpy import arange
from numpy import array
import numpy as np
import struct
import numpy as np
import pandas as pd

def readNHS(file, ftype="ecg"):
    # ftype to lower
    ftype = ftype.lower()
    
    # Determine nuubo's sampling frequency
    fs = {"ecg": 250., "aac": 50., "seq": 3.57, "ppm": 250., "ia": 250., "ang": 250.}
    fs = fs.get(ftype,None)
    if fs is None: raise ValueError("Data type not allowed")

    # Read whole file
    with open(file,"rb") as f:
        fileinfo = f.read(512)
        bheader  = f.read(512)
    content = np.fromfile(file,">u2",offset=1024)

    # Dict for outputs
    header = {}

    assert fileinfo[:9].decode('utf-8') == "nuubofile", "Not a nuubo file"

    # Get header information
    header["version"] = fileinfo[10:13].decode("utf-8")
    header["name"] = bheader[20:34].decode("utf-8")
    header["date"] = "/".join([str(bheader[41]),str(bheader[42]),str(bheader[43]*256+bheader[44])]) + " " + ":".join([str(bheader[45]),str(bheader[46]),str(bheader[47])])
    header["GPS"]  = (bheader[50] == 3)
    header["fw"]   = struct.unpack("BBB",bheader[85:88])
    header["filters"] = (bheader[83]/100,bheader[84])
    header["SigGain"] = struct.unpack(">f",bheader[73:77])
    header["AccGain"] = struct.unpack(">f",bheader[79:83])
    header["SigGain"] = -2.4267e-6 if (header["SigGain"] == 0) else header["SigGain"]
    header["AccGain"] = -2.8061e-3 if (header["AccGain"] == 0) else header["AccGain"]
    header["nFrames"] = len(content)//256
    header["fs"]      = fs
    assert len(content)//256 == len(content)/256, "Problem with file, possibly bad write/copy; EOF earlier than expected"

    if   ftype == "ecg":
        # Get content as frames
        allinfo  = np.reshape(content,((header["nFrames"],256)))

        # frame ID stored in a int64 in the first 2 positions
        frame_id = allinfo[:,0]*(2**16)+allinfo[:,1]

        # Discard some bits - ECG is stored in those positions (????)
        signal  = allinfo[:,3:-43]

        # Reshape to flatten as final ECG
        signal = np.reshape(signal,((signal.shape[0],3,signal.shape[1]//3)))
        signal = np.swapaxes(signal,1,0)
        signal = np.reshape(signal,(signal.shape[0],signal.shape[1]*signal.shape[2]))

        # Correct baseline (WHY???)
        signal = (signal - np.array([[2**13,],[2**13+2**12,],[2**14,]])).astype(float)

        # Apply ECG gain
        signal *= header["SigGain"]
    else: 
        raise NotImplementedError("Not yet implemented for non-ECG data")
        
    return signal,fs

def ECG_reader_ADAS(fname: str, default_fs: float = 1000., **kwargs) -> Tuple[ndarray, ndarray, dict]:
    """ECG reader in ADAS format. 
    
    Inputs:
        fname (str): absolute path to file
        default_fs (float): default value for when the sampling frequency has not been found
    
    Outputs:
        x_vector: time vector in format N_LEADS x N_SAMPLES
        y_vector: ECG vector in format N_LEADS x N_SAMPLES
        header: dictionary containing the following fields:
         -> 'channels' : names of the channels (list)
         -> 'x_units' : units of the time vector per channel (list)
         -> 'y_units' : units of the ECG per channel (list)
         -> 'samples' : number of samples per channel (list)
         -> 'start_time' : starting time of the ECG per channel (list)
         -> 'fs' : sampling frequency per channel (list)
         -> 'annotations' : Free form annotations per channel (list)
    
    Information:
        All files are composed of 16 lines per signal which follow the format:
        HEADER
        SIGNAL_NAME
        X UNIT
        Y UNIT
        NUMBER OF RECORDS
        SAMPLES PER RECORD
        SAMPLES PER CHANNEL (duration.numerator)
        SAMPLES RATE (duration.denominator)
        START TIME
        X VECTOR
        Y VECTOR
        ANNOTATIONS -> (NUM_ANNOTATIONS, TYPE_ANNOTATION, NAME_ANNOTATION, 
                        POS_ANNOTATION, Y_AT_ANNOTATION, DURATION_ANNOTATION)
        EMPTY LINE (\\n)
        EMPTY LINE (\\n)
    """

    # Set defaults
    kwargs["delimiter"] = kwargs.get("delimiter",",")

    # Read file
    with open(fname) as f:
        file_info = f.read().splitlines()
    
    # Prepare output structures
    x_vector = []
    y_vector = []
    header = {
        'channels': [],
        'x_units': [],
        'y_units': [],
        'samples': [],
        'start_time': [],
        'fs': [],
        'ann_number': [],
        'ann_type_id': [],
        'ann_type_name': [],
        'ann_x_value': [],
        'ann_y_value': [],
        'ann_duration': [],
    }
    
    # Iterate over file. Header always same size and the same order: 
    for k in range(0,len(file_info),14):
        # Retrieve header data
        header['channels'].append(file_info[k+1])
        header['x_units'].append(file_info[k+2])
        header['y_units'].append(file_info[k+3])
        header['samples'].append(int(file_info[k+6]))
        header['start_time'].append(int(file_info[k+8]))
        # Annotations
        ann = file_info[k+11].split(sep=';')[:-1]
        header['ann_number'].append(ann[0::6]) # Repeats every 6 fields
        header['ann_type_id'].append(ann[1::6]) # Repeats every 6 fields
        header['ann_type_name'].append(ann[2::6]) # Repeats every 6 fields
        header['ann_x_value'].append(ann[3::6]) # Repeats every 6 fields
        header['ann_y_value'].append(ann[4::6]) # Repeats every 6 fields
        header['ann_duration'].append(ann[5::6]) # Repeats every 6 fields
        # Sampling frequency
        try:
            header['fs'].append(float(file_info[k+7]))
        except:
            header['fs'].append(default_fs)

        # Retrieve signal info
        y_vector.append(fromstring(file_info[k+10],sep=';'))
        if file_info[k+9] == '':
            x_vector.append(arange(header['samples'][-1]))
        else:
            x_vector.append(fromstring(file_info[k+9],sep=';'))
        
        # Check data
        if y_vector[-1].size != x_vector[-1].size:
            raise ValueError(("Corrupt file. Dimensions of X VECTOR and Y VECTOR "+
                              "do not coincide in signal {}".format(header['channels'][-1])))

    # Convert to ndarray
    x_vector = array(x_vector)
    y_vector = array(y_vector)
    
    return x_vector, y_vector, header

def ECGReader(file: str, **kwargs) -> Tuple[ndarray, list, float]:
    """    
    ECG_Reader Reads the ECG data (file delimited by commas, with a header 
    containing the names and order of the electrodes) into a matrix. Allows
    for EPTracer data or data directly exported from BioSense Webster's CARTO
      INPUTS:
        * file: relative or absolute path of the file to be read
        * kwargs: different per-file keyword arguments: e.g. decimal comma.
      
      OUTPUTS:
        * signal: matrix of doubles containing the data, in the shape of MxN, 
                  being M the number of samples and N the number of 
                  electrodes.
        * header: header of the file, in the shape of a cell of strings. Each
                  element of the cell a string containing the name of an 
                  electrode.
        * Fs:     Sampling frequency."""

    # Read the first line to know which reader to use:
    with open(file,'r') as f:
        line                        = f.readline().strip().split(' ')

    if "[Header]" in line:
        (signal, header, Fs)        = Read_CARTO_ECG_Data(file, **kwargs)
    elif "ID:" in line:
        (signal, header)            = Read_CARTO_3V6_ECG_Data(file, **kwargs)
        Fs                          = 2000
    else:
        fname,ext                   = os.path.splitext(file)
        if os.path.isfile(fname+".inf"):
            # Check if .inf is in fact a header
            with open(os.path.join(fname+".inf")) as fheader:
                headerinfo          = fheader.read().splitlines()
            
            # Check if header-like
            if any([s.startswith('Data Sampling Rate = ') for s in headerinfo]):
                (signal, header, Fs)= Read_General_Electric_data(file, **kwargs)
            else:
                # If no sampling frequency returned, assume 1000Hz (EPTracer data)
                (signal, header)    = Read_EPTracer_Data(file, **kwargs)
                Fs                  = 1000
        else:
            # If no sampling frequency returned, assume 1000Hz (EPTracer data)
            (signal, header)        = Read_EPTracer_Data(file, **kwargs)
            Fs                      = 1000

    if signal.ndim == 2:
        if signal.shape[0] > signal.shape[1]:
            signal                  = signal.T
            
    return (signal,header,Fs)



def SegmentationReader(filename: str, **kwargs) -> Tuple[ndarray, list]:
    # Set defaults
    kwargs["delimiter"] = kwargs.get("delimiter",",")

    return loadtxt(filename, dtype=int, **kwargs)

def Read_EPTracer_Data(filename: str, **kwargs):
    """Read_EPTracer_Data - Reads the EPTracer data (file delimited by spaces or
    commas, with a header containing the names and order of the electrodes)
    into a matrix.
      INPUTS:
        * file: relative or absolute path of the file to be read
        * delimiter: used delimiter. Usually a space (' ') or a comma (',').
      
      OUTPUTS:
        * signal: matrix of doubles containing the data, in the shape of MxN, 
          being M the number of samples and N the number of electrodes.
        * header: header of the file, in the shape of a cell of strings. Each
          element of the cell a string containing the name of an electrode.
        * Fs:     Sampling frequency."""

    # Set defaults
    kwargs["delimiter"] = kwargs.get("delimiter",",")

    with open(filename, 'r') as f:
        header = f.readline().strip()
        if " P 1" in header:
            header = header.replace(" P 1", " P1")
            header = header.replace(" P 2", " P2")
            header = header.replace(" P 3", " P3")
            header = header.replace(" P 4", " P4")
        if " I 1" in header:
            header = header.replace(" I 1", " I1")
            header = header.replace(" I 2", " I2")
            header = header.replace(" I 3", " I3")
            header = header.replace(" I 4", " I4")
        header = header.split(' ')
        signal = loadtxt(f, **kwargs)
        header = list(filter(None, header))

    return (signal,header)

def Read_CARTO_3V6_ECG_Data(filename: str, **kwargs) -> Tuple[ndarray, list]:
    # Set defaults
    kwargs["delimiter"] = kwargs.get("delimiter",",")

    with open(filename,'r') as fid:
        lines = fid.readlines()

    nxtline = False
    values = False
    for line in lines:
        if "ECG:" in line.strip():
            nxtline = True
            continue
        if nxtline:
            header = line.strip().split(' ')
            header = list(filter(None, header))
            nxtline = False
            values = True
            signal = zeros((len(lines),len(header)))
            counter = 0
            continue
        if values:
            line = line.replace('\n','')
            chars = len(line)//12
            mod = 0
            for i in range(12):
                if i == 0:
                    num = line[-chars:]
                    if (line[-chars-1]) == '-':
                        num = line[-chars-1] + num
                        mod = mod + 1
                else:
                    num = line[-(i+1)*chars-mod:-i*chars-mod]
                    if i != 11:
                        if (line[-(i+1)*chars-mod-1]) == '-':
                            num = line[-(i+1)*chars-mod-1] + num
                            mod = mod + 1

                signal[counter,-1-i] = float(num)
            counter = counter + 1
            
    signal = signal[:counter,:]
    
    return (signal, header)


def Read_CARTO3_ECG(filename: str) -> Tuple[ndarray, list]:
    # Read file
    with open(filename,'r') as fid:
        lines = fid.read().splitlines()
        
    (fmt,gain,chn,hea),lines = lines[:4],lines[4:]
    assert fmt == "ECG_Export_4.0"
    # Recover header information
    chn = [c.split(" ")[0] for c in chn.split("=")[1:]]
    N_sigs  = len(hea.split("("))-1
    N_chars = len(hea)//N_sigs
    assert len(hea)%N_sigs == 0, "CHECK HEADER FORMAT, NOT EVENLY DISTRIBUTED"
    header = {
        "gain": float(gain.split(" = ")[1]),
        "UnipolarChannel":  chn[0],
        "BipolarChannel":   chn[1],
        "ReferenceChannel": chn[2],
        "Channels":         [hea[i:i+N_chars].replace(" ","") for i in range(0,len(hea),N_chars)]
    }

    signal = np.zeros((N_sigs,len(lines)),dtype=int)
    for i,line in enumerate(lines):
        mod = 0
        for j in range(N_sigs):
            if j == 0:
                num = line[-N_chars:]
                if (line[-N_chars-1]) == '-':
                    num = line[-N_chars-1] + num
                    mod = mod + 1
            else:
                num = line[-(j+1)*N_chars-mod:-j*N_chars-mod]
                if j != N_sigs-1:
                    if (line[-(j+1)*N_chars-mod-1]) == '-':
                        num = line[-(j+1)*N_chars-mod-1] + num
                        mod = mod + 1

            signal[j,-1-i] = int(num)

    signal = signal[::-1,::-1]
            
    return (signal, header)


def Read_CARTO_ECG_Data(filename: str, **kwargs) -> Tuple[ndarray, list, float]:
    """Read_CARTO_ECG_Data - Reads the CARTO ECG data (file delimited by spaces or
    commas, with a header containing basic information of the file being read.
      INPUTS:
        * file: relative or absolute path of the file to be read
        * delimiter: used delimiter. Usually a space (' ') or a comma (',').
      
      OUTPUTS:
        * signal: matrix of doubles containing the data, in the shape of MxN, 
          being M the number of samples and N the number of electrodes.
        * header: header of the file, in the shape of a cell of strings. Each
          element of the cell a string containing the name of an electrode.
        * Fs: Sampling frequency of the recording"""

    # Set defaults
    kwargs["delimiter"] = kwargs.get("delimiter",",")

    with open(filename, 'r') as f:
        header  = []
        Fs      = 0
        line    = f.readline().strip().split(' ')
        
        while not("[Data]" in line):
            if "Label:" in line:
                header.append(' '.join(line[1:]))

            if ("Sample" in line) and ("rate:" in line):
                Fs = float(line[2].strip("Hz"))
            
            line = f.readline().strip().split(' ')
            
        signal = loadtxt(f, **kwargs)

    return (signal,header,Fs)


def Read_Polygraph(filename: str, **kwargs) -> Tuple[np.ndarray, list, float]:
    """Read_Polygraph - Reads the polygraph data (file delimited by spaces or
    commas, with a header containing basic information of the file being read.
      INPUTS:
        * file: relative or absolute path of the file to be read
        * delimiter: used delimiter. Usually a space (' ') or a comma (',').
      
      OUTPUTS:
        * signal: matrix of doubles containing the data, in the shape of MxN, 
          being M the number of samples and N the number of electrodes.
        * header: header of the file, in the shape of a cell of strings. Each
          element of the cell a string containing the name of an electrode.
        * Fs: Sampling frequency of the recording"""

    # Set defaults
    kwargs["delimiter"] = kwargs.get("delimiter",",")

    with open(filename, 'r') as f:
        header  = []
        Fs      = []
        line    = f.readline().strip().split(' ')
        
        while not("[Data]" in line):
            if "Label:" in line:
                header.append(' '.join(line[1:]))

            if ("Sample" in line) and ("rate:" in line):
                Fs.append(float(line[2].strip("Hz")))
            
            line = f.readline().strip().split(' ')
            
        signal = np.loadtxt(f, **kwargs)

    return (signal,header,Fs)


def Read_General_Electric_data(filename: str, **kwargs) -> Tuple[np.ndarray, list, float]:
    """Read_General_Electric_data - Reads polygraph data from General Electric
       (.inf header with channel information and .txt as space separated values)
      INPUTS:
        * filename: path of the file to be read
      
      OUTPUTS:
        * signal: matrix of doubles containing the data, in the shape of MxN, 
          being M the number of samples and N the number of electrodes.
        * header: header of the file, in the shape of a list of strings. Each
          element of the list a string containing the name of an electrode.
        * Fs: Sampling frequency of the recording"""

    # Set defaults
    kwargs["sep"] = kwargs.get("sep"," ")
    kwargs["decimal"] = kwargs.get("decimal",",")
    kwargs["header"] = None

    # Split filename in file and extension
    fname,ext = os.path.splitext(filename)
    with open(os.path.join(fname+".inf")) as fheader:
        headerinfo = fheader.read().splitlines()

    # Read header
    starts_channels,header,fs = False,[],None
    for j,ln in enumerate(headerinfo):
        if ln.startswith("Data Sampling Rate"): 
            fs = float(ln.split("=")[1].replace("points/second",""))
        if ln.startswith("Units:"): 
            units = "mV" if " mV " in ln else "V"
        if ln.startswith("Channel Number"):
            starts_channels = True
            continue
        if starts_channels:
            channel = [s for s in ln.split(" ") if s != ""]
            header.append(" ".join(channel[1:]))

    if fs is None: # Yeah, for some reason (????)
        fs = 977.0

    # Read signal
    signal = pd.read_csv(filename,**kwargs)
    signal = signal.values[:,:-1].T

    return (signal,header,fs)


