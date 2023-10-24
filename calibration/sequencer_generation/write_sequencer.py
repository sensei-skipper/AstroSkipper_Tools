import numpy as np
import sys
import re
import argparse
import logging
import pickle

def coords(s):
    try:
        x, y = map(int, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Center must be row,col")

parser = argparse.ArgumentParser()
parser.add_argument('--NROW', type=int, default=-1)
parser.add_argument('--NCOL', type=int, default=-1)
parser.add_argument('--NSAMP', type=int, default=-1)
parser.add_argument('--CCDNROW', type=int, default=-1)
parser.add_argument('--CCDNCOL', type=int, default=-1)
parser.add_argument('--NROWCLUSTER', type=int, default=-1)
parser.add_argument('--NCOLCLUSTER', type=int, default=-1)
parser.add_argument('--filename', type=str, default=None)
parser.add_argument('--default-file', type=str, default='astroskipper_sequencer_default.xml')

# REQUIRED ARGS #
parser.add_argument('--center', type=coords, dest="center", nargs="+", default=[0,0]) # unit colcluster 
								  # length of centers defines number of ROIs
parser.add_argument('--width', type=int, nargs="*", default=[2]) # unit x_ncols
parser.add_argument('--height', type=int, nargs="*", default=[2]) # unit x_nrows

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

args = parser.parse_args()
vardict = vars(args)

#*********************************************************************************************************************************************#
#********************************************************DEFINE-FUNCTIONS*********************************************************************#
#*********************************************************************************************************************************************#

def search_and_update_vardict(vardict):
    '''Updates vardict with parameter values from default-file if no value input for param in arguments
    '''
    with open(vardict['default_file'], 'r') as file:
        file_data = file.read()
        default_seq_dict = dict()
        for key in vardict:
            if vardict[key] == -1:
                logging.debug("Searching for " + key + "...")
                try:
                    result = re.search(f'name="{key}" val="(.*)"/>', file_data)
                    logging.debug(f'Found {key} = {result.group(1)}')
                    vardict[key] = int(result.group(1))
                except:
                    logging.warning(f"{key} not in {vardict['default_file']}")
            if key == 'filename':
                break
    for key in vardict:
        logging.debug(f'{key} = {vardict[key]}')
    return vardict

def update_x(x, vardict, NSAMP):
    logging.debug(f'x before updating = {x}')
    logging.debug(f'shape(x) = {x.shape}')
    for (j, center) in enumerate(vardict['center']):
        logging.debug(f'Updating center {j}: {center}')
        width_left = int(vardict['width'][j]/2)
        width_right = width_left + 1
        if vardict['width'][j] % 2 == 0:
            width_right -= 1
        height_below = int(vardict['height'][j]/2)
        height_above = height_below + 1
        if vardict['height'][j] % 2 == 0:
            height_above -= 1
        col_start = center[1] - width_left
        col_end = center[1] + width_right
        row_start = center[0] - height_below
        row_end = center[0] + height_above
        logging.debug(f'BEFORE FIX --  row_start:row_end, col_start:col_end = {row_start}:{row_end}, {col_start}:{col_end}')
        col_start = fix_lower_bound(val=col_start)
        col_end = fix_upper_bound(val=col_end, upper_bound=vardict['x_ncols'])
        row_start = fix_lower_bound(val=row_start)
        row_end = fix_upper_bound(val=row_end, upper_bound=vardict['x_nrows'])
        logging.debug(f'AFTER FIX -- row_start:row_end, col_start:col_end = {row_start}:{row_end}, {col_start}:{col_end}')
        x[row_start:row_end, col_start:col_end] = NSAMP
    logging.debug(f'x after update = {x}')
    return x

def fix_lower_bound(val):
    if val < 0:
        return 0
    return val

def fix_upper_bound(val, upper_bound):
    if val > upper_bound:
        return upper_bound
    if val <= 0:
        return 1
    return val

def x_to_string(x, Print=False):
    x = ',\n'.join(','.join(str(element) for element in row) for row in x)
    if Print:
        print(x)
    return x

def search_and_replace(vardict, x): 
    with open(vardict['default_file'], 'r') as file:
        file_data = file.read()
        for key in vardict:
            find = r'name="' + key + '" val=.*/>'
            replace = r'name="' + key + '" val="' + str(vardict[key]) + '"/>'
            file_data = re.sub(find, replace, file_data)
    return re.sub(r'name="x" vals="NSAMP"', r'name="x" vals="' + str(x) + '"', file_data)

#*********************************************************************************************************************************************#
#********************************************************END-FUNCTIONS*************************************************************************************#
#*********************************************************************************************************************************************#

# update vardict
vardict = search_and_update_vardict(vardict)

# define number of centers and widths in input lists
ncenters = len(vardict['center'])
nwidths = len(vardict['width'])
nheights = len(vardict['height'])

# check that nwidths and nheights have same length or length of 1
if ncenters != nwidths and nwidths != 1:
    sys.exit('Error: nwidths must be same length as ncenters or of length 1')
if ncenters != nheights and nheights != 1:
    sys.exit('Error: nheights must be same length as ncenters or of length 1')
# if widths is length 1, make it a list of length ncenters for iteration below
if nwidths == 1:
    vardict['width'] = np.full((ncenters), vardict['width'])
if nheights == 1:
    vardict['height'] = np.full((ncenters), vardict['height'])
logging.debug(f"vardict heights and widths: {vardict['height']},  {vardict['width']}")
# get shape of dynamic variable (nrow/nrowcluster, ncol/ncolcluster)
if (vardict['NROW'] % vardict['NROWCLUSTER'] == 0) and (vardict['NCOL'] % vardict['NCOLCLUSTER'] == 0):
    vardict['x_nrows'] = int(vardict['NROW'] / vardict['NROWCLUSTER'])
    vardict['x_ncols'] = int(vardict['NCOL'] / vardict['NCOLCLUSTER'])
    x = np.ones(shape=(vardict['x_nrows'], vardict['x_ncols']), dtype=np.int32)
else:
    sys.exit('Error: NROW (NCOL) should be a multiple of NROWCLUSTER (NCOLCLUSTER)')

# check if center column clusters are out of bounds
centers_less_than_ncolclusters = all(c[1] < vardict['x_ncols'] for c in vardict['center']) 
centers_less_than_nrowclusters = all(c[0] < vardict['x_nrows'] for c in vardict['center']) 
if not centers_less_than_ncolclusters or not centers_less_than_nrowclusters:
    sys.exit('Error: Center values must be less than number of clusters')

x = update_x(x, vardict, vardict['NSAMP'])
file_data = search_and_replace(vardict, x_to_string(x))

if args.filename == None:
    args.filename = f"sequencer_center_locs_{'__'.join(str(x[0]) + '_' + str(x[1]) for x in vardict['center'])}_widths_{'_'.join(str(x) for x in vardict['width'])}_heights_{'_'.join(str(x) for x in vardict['height'])}_NSAMP_{vardict['NSAMP']}.xml"
vardict['filename'] = args.filename

for key in vardict:
    logging.info(f'{key} : {vardict[key]}')

with open(vardict['filename'], 'w') as file:
    file.write(file_data)
with open(args.filename[10:-4] + '_DYNAMIC_VAR.pkl', 'wb') as fn:
    pickle.dump(x, fn)
