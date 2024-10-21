"""
River Forcing Generation Script for Black Sea Model, version 1.0

This script generates monthly river forcing files for the NEMO-BAMHBI Black Sea model for the period 1950-2024. It processes data from various sources including EFAS, SESAME, and in-situ measurements to
create consistent river input files for both physical and biogeochemical variables.

Author: Mathurin Choblet
Email: mathurin@choblet.com

Created on: 12.10.2024
Last modified: 14.10.2024

Version: 1.0

Dependencies:
- Python 3.7+
- numpy
- xarray
- pandas
- scipy

Input data required:
- EFAS discharge data
- SESAME/PERSEUS nutrient data
- In-situ measurements for Danube, Dnepr, and Dnestr
- Domain configuration files for low resolution (LR) and high resolution (HR) setups

Output:
- NetCDF files containing river forcing data for each year and domaintype 

Usage:
python river_forcing.py

Notes:
- Ensure all required input files are present in the specified directories
- The script uses a no-leap calendar for time calculations (output is only monthly, so no problem)
- River positions are based on visual inspection and matching the model land sea mask
- Nutrient calculations use estimated ratios and should be revised in future versions

For detailed methodology and data sources, refer to the accompanying documentation.

Next releases will include recent Danube nutrients from the TNMN database (https://wq-db.icpdr.org/)

"""

################ LOAD PACKAGES#################
import os
from datetime import datetime
import numpy as np
import xarray as xr
import glob
import pandas as pd
from scipy import stats  # for the extrapolation of riverflux to nutrients

################ BASIC DATA LOCATIONS ################

# Danube
# Daily discharge (since 2020, used from then on)
danube_daily_path = '/gpfs/scratch/acad/bsmfc/river_forcings/data/DANUBE_DAILY/HF*.csv'  # *-wildcard used by glob.glob

# Shelf rivers (Danube, Dnepr, Dnestr)
shelf_rivers_path = '/gpfs/scratch/acad/bsmfc/river_forcings/data/B-S_rivers_runoff_m_s-3_upd.xlsx'

# EFAS data (currently 6 hourly, will create downsampled files at some point)
efas_paths = '/gpfs/scratch/acad/bsmfc/river_forcings/data/cds_download/data_*.nc'  # *-wildcard used by glob.glob


# Temperature data for the Danube
danube_temp = '/gpfs/scratch/acad/bsmfc/river_forcings/data/NIHWM_WaterTemp_DataSet_V1.xlsx'  # covers 1984-2018. Outside that period, use climatology

# Ludwig/SESAME/PERSEUS files locations
ludwig_flux_path = '/gpfs/scratch/acad/bsmfc/river_forcings/data/SESAME_LUDWIG/water_fluxes.xls'
ludwig_nutrient_path = '/gpfs/scratch/acad/bsmfc/river_forcings/data/SESAME_LUDWIG/nutrient_fluxes.xls'
ludwig_silica_path = '/gpfs/scratch/acad/bsmfc/river_forcings/data/SESAME_LUDWIG/silica_fluxes.xls'

#file of in-situ nutrient data (only using oxygen here)
path_tnmn_ukraine='/gpfs/scratch/acad/bsmfc/river_forcings/data/tnmn_ukraine.csv'

savepath = '/gpfs/scratch/acad/bsmfc/river_forcings/forcing_files_v1'
################ DOMAIN AND RIVER LOCATIONS ################
#these are the locations in the domain files (there is also a location check (border of the land sea mask) in the final function.

domains={
    'lr':{
        'path': '/gpfs/scratch/acad/bsmfc/river_forcings/data/domain_lr.nc',  
               'locs': {
                   'Sakarya': [41.41555404663086, 30.51814842224121],
 'Filyos': [41.69333267211914, 31.999629974365234],
 'Kizil Irm': [41.83222198486328, 35.888519287109375],
 'Yesil': [41.41555404663086, 36.6292610168457],
 'Rioni': [42.2488899230957, 41.81444549560547],
  'Inguri': [42.66555404663086, 41.6292610168457],
 'Coroch': [41.554443359375, 41.44407272338867],
 'Kodori': [42.94333267211914, 41.258888244628906],
 'Bzyb': [43.36000061035156, 40.33296203613281],
 'Dnestr': [45.99889,30.518148],
 'Danube': [[[45.72111,29.962593],[45.304443,29.96259],[45.02667,29.962593]],[0.58,0.19,0.23]],
 'Dnepr': [46.554443,31.814444],
 'Bug': [46.554443,31.814444], #Bug will be put at same place as dnepr. Important that it comes **after** the Dnepr because we'll simply add values, we will need to carefully treat the salinity,temperature and runoff depth variables
}},
'hr':{
    'path':'/gpfs/scratch/acad/bsmfc/river_forcings/data/domain_hr.nc',
    'locs':
    {'Sakarya': [41.150001525878906, 30.600000381469727],
 'Filyos': [41.57500076293945, 32.025001525878906],
 'Kizil Irm': [41.75, 35.974998474121094],
 'Yesil': [41.375, 36.67499923706055],
 'Rioni': [42.20000076293945, 41.625],
  'Inguri': [42.400001525878906, 41.54999923706055],
 'Coroch': [41.599998474121094, 41.57500076293945],
 'Kodori': [42.79999923706055, 41.125],
 'Bzyb': [43.20000076293945, 40.25],
  'Dnestr': [46.075,30.5],
 'Danube': [[[45.475,29.825],[45.35,29.775],[45.2,29.775],[45.125,29.7],[44.875,29.65]],[0.58/4,0.58/4 ,0.58/2,0.19,0.23]], #second part are the fractions (as in old script, see e.g.  https://unece.org/DAM/env/eia/documents/inquiry/Annex%2004.pd)
 'Dnepr' : [[[46.6,31.475],[46.575,31.475]],[0.5,0.5]],
 'Bug':  [[[46.6,31.475],[46.575,31.475]],[0.5,0.5]], # same as Dnepr. Important that it comes **after** the Dnepr because we'll simply add values
  }},
}

# Create directory structure for saving data (e.g. HR/rnf_bgc)
for k in domains.keys():
    for subdir in ['', 'rnf_bgc', 'rnf']:
        path = os.path.join(savepath, k.upper(), subdir)
        os.makedirs(path, exist_ok=True)

# River locations in EFAS (needed for all rivers except for Danube)
# EFAS is a gridded file with surface discharge. locations should be select with xarray sel-method nearest. 
# The river positions have been found by visual inspection (using Google Earth and looking for the closest cell with river discharge in EFAS files)
# Near the rivermouth, the different river cells are very similar (it doesn't matter which one you take)
# Note, that when redownloading EFAS data, the grid latitudes and longitudes can be slightly different from older versions and thus cause problems.
efas_pos = {
    'Dnestr': [46.07, 30.48],
    'Dnepr': [46.56, 32.42],
    'Bug': [46.92, 31.96],
    'Sakarya': [41.11, 30.61],
    'Filyos': [41.57487300000069, 32.05833379740995],
    'Kizil Irm': [41.74, 35.96],
    'Yesil': [41 + 22/60, 36 + 40/60],
    'Rioni': [42.19155200000063, 41.64166730804642],
    'Inguri': [42.391556000000605, 41.55833397317132],  # not named in Ludwig dataset
    'Coroch': [41.60820700000069, 41.57500064014634],                
    'Kodori': [42 + 48/60, 41 + 9/60],
    'Bzyb': [43.19157200000052, 40.275000616094786],
}
river_names = list(efas_pos.keys()) + ['Danube']

################ VARIABLES OF INTEREST ################

# Primary BGC variables derived from total NO3, PO4, SIL in SESAME Ludwig dataset
#inner list is ['reference nutrient', fraction]. 
#ratios and multiplicators for bgc tracers come without any guarantee.
#probably the exact distribution is not that important, more the total nitrogen influx
vars_bgc_prim = {
    'NOS': ['no3', 1], 'NHS': ['no3', 0.1], 'DNL': ['no3', 1/6*(1+0.1)], 'DNS': ['no3', 1/6*(1+0.1)],
    'DCL': ['no3', 1/6*(1+0.1)*10], 'DCS': ['no3', 1/6*(1+0.1)*10],
    'PHO': ['po4', 1],
    'SIO': ['SIL', 0.9], 'SID': ['SIL', 0.1]
}

# Secondary BGC tracers with arbitrary multiplicators (multiplied with flux)
#mainly very small values, except for BAC,DIC,TA (SMI and AGG currently disabled in simulations)
#these numbers were more varying and in part larger, but I prefer to put them small and more equally distributed [to be discussed!]
#also what you can do to check if these values are realistic is to compare these concentrations to typical surface concentrations in existing model data
vars_bgc_sec = {
    'POC': 100, 'PON': 10, 'CFL': 0.01, 'CEM': 0.01, 'CDI': 0.01, 'NFL': 0.01, 'NEM': 0.01, 'NDI': 0.01,
    'MIC': 0.001, 'MES': 0.001, 'BAC': 0.1, 'SMI': 25, 'AGG': 82661000, 'GEL': 0, 'NOC': 0, 'DIC': 3000, 'TA': 3083
}
#setting pon to 10, corresponds to about 26kt/year for the danube, about 10% or less of DIN (~300-600kt), which is ok. (10*10**(-3)*12 [mmol to molar weight]*10**(-9)[g to kilotonnes)*24*3600*365 [s -> year]*7000 [m3/s]
#DIC 3000: comes from gems glori value (~36mg/l)

vars_phys = ['sorunoff', 'rosaline', 'rotemper', 'rodepth']
all_variables = vars_phys + ['DOX'] + list(vars_bgc_prim.keys()) + list(vars_bgc_sec.keys())
#note, that the excess negative charge (CHA) is missing here and will be added at the end of the script (computed from other variables)

################ CREATE EMPTY DATASETS ################

times_all = xr.cftime_range(start="1950", end="2024-09", freq="MS", calendar="noleap")
river_data = {
    riv: xr.Dataset(
        {v: xr.DataArray(np.zeros(len(times_all)), dims=['time'], coords={'time': times_all})
         for v in all_variables},
        coords={'time': times_all}
    )
    for riv in river_names
}

################ LOAD EFAS DATA ################
#loading takes about 25seconds on my machine (efas is 6 hourly). at some we'll just store monthly values: less time and disk space
dis_efas = xr.open_mfdataset(glob.glob(efas_paths), preprocess=lambda ds: ds.resample(time='MS').mean()['dis06'],coords='minimal').load()['dis06']
dis_efas = dis_efas.sel(time=slice("1992","2024-09")).drop(('surface','step'))
################ GET WATER DISCHARGE ################

#HELPER FUNCTIONS FOR READING CSV FILES with varying delimiter
def detect_delimiter(file_path):
    """Detect the delimiter used in a CSV file."""
    with open(file_path, 'r') as file:
        first_line = file.readline()
    
    if ',' in first_line:
        return ','
    elif ';' in first_line:
        return ';'
    else:
        raise ValueError('Delimiter not found. The file does not use , or ;')

def read_csv_with_dynamic_delimiter(file_path):
    """Read a CSV file with automatic delimiter detection."""
    delimiter = detect_delimiter(file_path)
    return pd.read_csv(file_path, delimiter=delimiter)

def extract_date_from_filename(file_path):
    """Extract date from filename in format YYYYMMDD."""
    base_name = os.path.basename(file_path)
    date_str = base_name.split('_')[1].split('.')[0][:8]  # Extract YYYYMMDD
    return datetime.strptime(date_str, '%Y%m%d')



# Load data for Danube, Dnepr, and Dnestr
datas = []
for sheet_name in ['Danube', 'Dnieper', 'Dniester']:
    df = pd.read_excel(shelf_rivers_path, sheet_name=sheet_name)
    years = df.iloc[:, 0]
    months = df.iloc[:, 1:13]
    time_index = pd.date_range(start=f'{years.iloc[0]}-01', periods=len(years) * 12, freq='MS')
    river_flow_values = months.values.flatten()
    datas.append(xr.Dataset({"RIV": (["time"], river_flow_values)}, coords={"time": time_index}))

# Process daily Danube data since 2020
date_list, q_list = [], []
for p in sorted(glob.glob(danube_daily_path)):
    file_date = extract_date_from_filename(p)
    df = read_csv_with_dynamic_delimiter(p)
    df['date'] = file_date
    filtered_df = df[df['Section_Name'] == 'Isaccea']
    date_list.extend(filtered_df['date'])
    q_list.extend(filtered_df['Q'])

danube_daily = xr.DataArray(
    data=pd.DataFrame({'time': pd.to_datetime(date_list), 'Q': q_list}).set_index('time')['Q'].values,
    dims=['time'],
    coords={'time': pd.to_datetime(date_list)},
    name='RIV'
)

# Combine Danube data
danube_monthly = danube_daily.resample(time='MS').mean()
danube1 = datas[0]['RIV'].resample(time='MS').mean()
danube = xr.concat([danube1, danube_monthly], dim='time')
danube = danube.sortby('time').drop_duplicates('time', keep='last').convert_calendar('noleap') #for the overlap between daily danube data and the data from shelf_rivers_path, we choose the daily/monthly data (keep='last')
danube = danube.sel(time=times_all, method='nearest')
river_data['Danube']['sorunoff'] = danube

# Process Dnepr and Dnestr data. add efas for latest years where the insitu data ends
#index 1,2 according to how we created 'datas' above
for i, river in enumerate(['Dnepr', 'Dnestr'], start=1):
    lat, lon = efas_pos[river]
    efas_data = dis_efas.sel(latitude=lat, longitude=lon, method='nearest').sel(time=slice("1992", None))
    in_situ_mean = datas[i]['RIV'].sel(time=slice("1992", None)).mean('time')
    recent_data = efas_data - efas_data.mean('time') + in_situ_mean #"debias" efas wrt i_situ
    combined_data = xr.concat([recent_data, datas[i]['RIV']], dim='time').convert_calendar('noleap')
    river_data[river]['sorunoff'] = (combined_data.sortby('time')
                                     .drop_duplicates('time', keep='last')
                                     .drop(('latitude', 'longitude'))
                                     .sel(time=times_all, method='nearest'))




############################ FOR THE OTHER RIVERS: USE SESAME AND EFAS  ############################
#yearly sesame data is distributed across months according to efas (1992-2000) climatology.
#sesame is 'debiased' with respect to efas
#for the 1950s: take climatology from 1960-1969
def process_sesame_efas(river, lat, lon):
    """Process SESAME and EFAS data for a given river."""
    sesame_flux = flux[flux['RiverName'].str.lower() == river.lower()]
    flux_ds = xr.DataArray(data=sesame_flux[np.arange(1960, 2001)].T.values.flatten(), coords={'time': time_sesam})
    flux_ds = flux_ds * 10**9 / (365.25 * 24 * 60 * 60)  # Convert from km3/year to m3/second
    efas = dis_efas.sel(latitude=lat, longitude=lon, method='nearest').drop(('latitude', 'longitude'))
    
    # Debias SESAME data with respect to EFAS for the overlap period
    flux_ds = flux_ds - flux_ds.sel(time=slice("1992", "2000")).mean('time') + efas.sel(time=slice("1992", "2000")).mean('time')
    flux_ds = flux_ds * (365.25 * 24 * 60 * 60)  # Convert flux from m3/second to m3/year
    
    
    # Get the seasonality of EFAS flux
    mean_cycle_90s = efas.sel(time=slice("1992", "2000")).groupby('time.month').mean()
    mean_cycle_normalized_90s = mean_cycle_90s / mean_cycle_90s.sum()
    
    # Resample to monthly, fill with repeating value
    flux_monthly = flux_ds.resample(time='MS').ffill().sel(time=slice(None, "1991"))
    flux_monthly = (flux_monthly.groupby('time.month') * mean_cycle_normalized_90s / (30.4375 * 24 * 3600)).drop('month')
    
    # Compute climatology for 1950-1959
    climatology = flux_monthly.sel(time=slice("1960", "1969")).groupby('time.month').mean()
    flux_50s = xr.DataArray(data=np.tile(climatology.values, 10),
                            coords={'time': xr.cftime_range(start='1950', end='1959-12', freq='MS', calendar='standard')},
                            dims=['time'], name='flux')
    
    return xr.concat([flux_50s, flux_monthly, efas], dim='time').assign_coords(time=times_all)

# Process SESAME and EFAS data for rivers
flux = pd.read_excel(ludwig_flux_path, sheet_name='km3 per yr')
time_sesam = xr.cftime_range(start='1960', end='2000', freq='YS', calendar='standard')

for river, (lat, lon) in efas_pos.items():
    if river not in ['Danube', 'Dnepr', 'Dnestr', 'Inguri']:
        river_data[river]['sorunoff'] = process_sesame_efas(river, lat, lon)

# Special treatment for Inguri (not named in sesame file). just using efas and efas 90s climatology for time before 1992.
lat, lon = efas_pos['Inguri']
efas = dis_efas.sel(latitude=lat, longitude=lon, method='nearest').drop(('latitude', 'longitude'))
climatology = efas.sel(time=slice("1992", "2000")).groupby('time.month').mean()
flux_5091 = xr.DataArray(data=np.tile(climatology.values, 42),
                         coords={'time': xr.cftime_range(start='1950', end='1991-12', freq='MS', calendar='standard')},
                         dims=['time'], name='flux')
river_data['Inguri']['sorunoff'] = xr.concat([flux_5091, efas], dim='time').assign_coords(time=times_all)

############################ WATER TEMPERATURE ############################
def process_danube_temp(df):
    """Process Danube temperature data."""
    branches = {
        'Water Temp': 'DANUBE',
        'Water Temp.1': 'TULCEA ARM',
        'Water Temp.2': 'CHILIA ARM',
        'Water Temp.3': 'SULINA ARM',
        'Water Temp.4': 'SFANTU GHEORGHE ARM'
    }
    ds = xr.Dataset({branch: (('time'), df[col], {'long_name': f'Water Temperature for {branch}', 'units': 'Celsius'})
                     for col, branch in branches.items()}, coords={'time': df.index}).convert_calendar('noleap')
    
    clim_80s = ds['TULCEA ARM'].sel(time=slice("1984", "1993")).groupby('time.month').mean()
    clim_2010s = ds['TULCEA ARM'].sel(time=slice("2010", "2018")).groupby('time.month').mean()
    
    temps = {}
    for branch in ['TULCEA ARM', 'CHILIA ARM', 'SULINA ARM', 'SFANTU GHEORGHE ARM']:
        empty = xr.DataArray(np.zeros(len(times_all)), dims=['time'], coords={'time': times_all})
        empty.loc[{'time': times_all[times_all < ds.time.min()]}] = clim_80s.sel(month=times_all[times_all < ds.time.min()].month).values
        empty.loc[{'time': times_all[times_all > ds.time.max()]}] = clim_2010s.sel(month=times_all[times_all > ds.time.max()].month).values
        empty.loc[{'time': times_all[(times_all >= ds.time.min()) & (times_all <= ds.time.max())]}] = ds[branch].sel(time=times_all[(times_all >= ds.time.min()) & (times_all <= ds.time.max())], method='nearest').values
        temps[branch] = empty
    
    return temps

# Process Danube temperature data
df_temp = pd.read_excel(danube_temp, sheet_name='Sheet1', header=2, parse_dates=['Date (Month/Year)'])
df_temp = df_temp.rename(columns={'Date (Month/Year)': 'time'}).set_index('time')
temps = process_danube_temp(df_temp)

# Assign temperatures to rivers (will use specific arms when creating gridded files in last part of script)
#for non-shelf rivers use temperature -999: nemo will select sea surface temperature [to be discussed, how representative is the danube climatology for rivers in Georgia and Turkey?
temp_sst = xr.DataArray(np.ones(len(times_all)) * (-999), dims=['time'], coords={'time': times_all})
for riv in river_names:
    river_data[riv]['rotemper'] = temps['TULCEA ARM'] if riv in ['Dnestr', 'Dnepr', 'Bug', 'Danube'] else temp_sst

############################ DISSOLVED OXYGEN CLIMATOLOGY FROM DANUBE DATA ############################
#used for all rivers
def create_station_dataset(station_data, variables):
    """Create an xarray Dataset from station data for TNMN data."""
    data_arrays = {}
    for var_long, var_short in variables.items():
        var_data = station_data[station_data['parameter_display'] == var_long]
        da = xr.DataArray(
            data=var_data['value'].values,
            dims=['time'],
            coords={'time': var_data['date_of_analysis']},
            name=var_short,
            attrs={'long_name': var_long, 'units': var_long.split(' ')[-1]}
        )
        data_arrays[var_short] = da.reindex(time=station_data['date_of_analysis'].unique())
    
    return xr.Dataset(data_arrays).sortby('time')
    
def process_dox(df):
    """Process dissolved oxygen data. Computing the average seasonality of two stations."""
    df['date_of_analysis'] = pd.to_datetime(df['date_of_analysis'])
    variables = {"Dissolved oxygen in mg/l": "dox"}
    stations = ['Reni', 'Vylkove']
    station_datasets = {station: create_station_dataset(df[df['location'] == station], variables).resample(time='MS').mean() for station in stations}
    dox_avg = xr.concat([station_datasets[station]['dox'].groupby('time.month').mean() for station in stations], dim='new').mean('new')
    dox_avg = dox_avg/32*1000 #convert from mg/l to mmolO2/m3
    dox = xr.DataArray(np.zeros(len(times_all)), dims=['time'], coords={'time': times_all})
    dox.loc[{'time': times_all}] = dox_avg.sel(month=times_all.month).values
    return dox

# Process and assign dissolved oxygen data (multiplied with riverrunoff!)
df_dox = pd.read_csv(path_tnmn_ukraine, sep=',')
dox = process_dox(df_dox)
for riv in river_names:
    river_data[riv]['DOX'] = dox*river_data[riv]['sorunoff']



############################ NUTRIENTS ############################
def calculate_linear_regression(x, y):
    """Calculate linear regression for nutrient-water flux relationship."""
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return slope, intercept

def get_nutrient_convert(nutrient_df, river, nutrient_name):
    """Convert nutrient data from kt/yr to mmol/year"""
    conversion_factors = {'sil': 28.0855, 'no3': 14.01, 'po4': 30.9738}
    sesame_data = nutrient_df[nutrient_df['RiverName'].str.lower() == river.lower()]
    return xr.DataArray(data=sesame_data[np.arange(1960, 2001)].T.values.flatten(), 
                        coords={'time': time}) * 10**12 / conversion_factors[nutrient_name]

def process_river_nutrients(river, runoff):
    """Process nutrient data for a given river."""
    flux_ds = runoff.sel(time=slice("1960", "2000")).resample(time='YS').mean()
    nutrients = {name: get_nutrient_convert(df, river, name) 
                 for name, df in zip(['sil', 'no3', 'po4'], [sil, no3, po4])}
    #For each month of the years get the relative discharge importance wrt to yearly discharge
    #this will be repetetive for all rivers except danube, dnepr and dnestr. 
    suma=runoff.sel(time=slice("1960", "2000")).groupby('time.year').sum()
    perc=(runoff.sel(time=slice("1960", "2000")).groupby('time.year')/suma)
    #mean_cycle_90s = runoff.sel(time=slice("1992", "2000")).groupby('time.month').mean() 
    #mean_cycle_normalized_90s = mean_cycle_90s / mean_cycle_90s.sum()
    
    #convert from per month to per seconds
    nutrients_monthly = {name: (nutrient.resample(time='MS').ffill()
                                .reindex(time=time_sesam, method='nearest')*perc.values / (30.417 * 24 * 3600))
                         for name, nutrient in nutrients.items()}

    # Calculate climatology for each nutrient for the 50s (no sesame data)
    runoff_50s = runoff.sel(time=slice("1950", "1959"))
    suma50s = runoff_50s.groupby('time.year').sum()
    perc50s = runoff_50s.groupby('time.year') / suma50s
    
    # Create time ranges
    time_range_50s = xr.cftime_range(start='1950', end='1959', freq='YS', calendar='standard')
    time_range_50s_monthly = xr.cftime_range(start='1950', end='1959-12', freq='MS', calendar='standard')
    fifties_climatology = {}
    for name, nutrient in nutrients.items():
        # Calculate mean for 1960s
        mean_60s = nutrient.sel(time=slice("1960", "1969")).mean('time')

        # Create DataArray with replicated mean values
        replicated_mean = xr.DataArray(
            data=np.tile(mean_60s.values.item(), 10),
            coords={'time': time_range_50s}
        )

        # Resample to monthly frequency and adjust
        monthly_values = (
            replicated_mean
            .resample(time='MS')
            .ffill()
            .reindex(time=time_range_50s_monthly, method='nearest')
        )

        # Apply percentages (month wrt yearly discharge) and convert units
        fifties_climatology[name] = (
            monthly_values * perc50s.values / (30.417 * 24 * 3600)
        )
    
    #calculate linear regression between flux and nutrient (ludwig base nutrients) and use this to extrapolate to years after 2000.
    recent_extrapolation = {}
    coefficients = []
    for name, nutrient in nutrients.items():
        slope, intercept = calculate_linear_regression(flux_ds.values, nutrient.values / (365 * 24 * 3600))
        nutrient_flux = slope * runoff.sel(time=slice("2001", None)) + intercept
        recent_extrapolation[name] = nutrient_flux
        coefficients.append([slope, intercept])
    
    combined_nutrients = {name: xr.concat([fifties_climatology[name], 
                                           nutrients_monthly[name], 
                                           recent_extrapolation[name]], 
                                          dim='time').assign_coords(time=times_all)
                          for name in nutrients.keys()}
    
    return combined_nutrients, coefficients

# Load nutrient data
no3 = pd.read_excel(ludwig_nutrient_path, sheet_name='no3_kt_yr')
po4 = pd.read_excel(ludwig_nutrient_path, sheet_name='po4_kt_yr')
sil = pd.read_excel(ludwig_silica_path, sheet_name='silica_kt_yr')

# Define time ranges
time = xr.cftime_range(start='1960', end='2000', freq='YS')
time_sesam = xr.cftime_range(start='1960', end='2000-12', freq='MS')

# Process nutrients for each river
rioni_coefficients = []
for river in river_names:
    if river != 'Inguri':
        nutrients, coeffs = process_river_nutrients(river, river_data[river]['sorunoff'])
        river_data[river].update(nutrients)
        if river.lower() == 'rioni':
            rioni_coefficients = coeffs

# Special treatment for Inguri (not in SESAME data)
#we will use the flux-nutrient relationship from the nearby rioni river
for j, nutrient_name in enumerate(['sil', 'no3', 'po4']):
    slope, intercept = rioni_coefficients[j]
    river_data['Inguri'][nutrient_name] = slope * river_data['Inguri']['sorunoff'] + intercept

# Fill remaining biogeochemical data for each river
for river in river_names:
    for var, (ref_nutrient, factor) in vars_bgc_prim.items():
        river_data[river][var] = river_data[river][ref_nutrient.lower()] * factor
    for var, factor in vars_bgc_sec.items():
        river_data[river][var] = factor * river_data[river]['sorunoff'] #multiplicator times runoff
    river_data[river]['rosaline'] = xr.DataArray(data=np.ones((len(times_all))) * 2, dims=['time'], coords={'time': times_all})
    river_data[river]['rodepth'] = xr.DataArray(data=np.ones((len(times_all))) * 13, dims=['time'], coords={'time': times_all})  # Assume all rivers are 9 meters deep at their mouth (FORTRAN depth level index 13)
    # Better than taking -999 (depth until ground), because especially for Turkish and Georgian rivers,
    # bathymetry is directly very steep and we'd have fresh water at large depths


############################ CREATE FORCING FILES ############################
def check_location(lat, lon, riv, domain):
    """Check if the given location is valid for river discharge according to domain['top_level'] (land-sea mask)"""
    maskval = domain['top_level'].sel(lat=lat, lon=lon, method='nearest')
    if maskval == 0:
        raise ValueError(f"Location ({lat}, {lon}) for river '{riv}' is on land! Aborting: Refine the position!")
    
    # Get indexes of those locations to check that it's really on the coastline
    lat = maskval.lat.values.item()
    lon = maskval.lon.values.item()
    
    lat_i = np.argwhere(domain.lat.values == lat).item()
    lon_i = np.argwhere(domain.lon.values == lon).item()
    
    surrounding_water = (
        domain['top_level'].isel(lat=lat_i-1, lon=lon_i).values.item() &
        domain['top_level'].isel(lat=lat_i+1, lon=lon_i).values.item() &
        domain['top_level'].isel(lat=lat_i, lon=lon_i-1).values.item() &
        domain['top_level'].isel(lat=lat_i, lon=lon_i+1).values.item()
    )
    
    if surrounding_water:
        raise ValueError(f"Location ({lat}, {lon}) for river '{riv}' is surrounded by water, but must have at least one land neighbor! Aborting: Refine the position!")

def convert_xy(ds, y='nav_lat', x='nav_lon'):
    """
    Convert latitudes and longitudes from nested to normal.
    Need that to have the domain file in a nice format.
    
    Args:
        ds (xarray.Dataset): Input dataset (any shape)
        y (str): Name of latitudes in coordinates
        x (str): Name of longitudes in coordinates
    
    Returns:
        xarray.Dataset: Dataset with simple 1D coordinates for latitudes and longitudes.
    """
    try:
        latitudes = ds[y].values
        longitudes = ds[x].values
        # Flatten latitudes and longitudes. get rid of -1 that are sometimes in the model data
        flat_latitudes = np.setdiff1d(np.unique(latitudes.flatten()), [-1])
        flat_longitudes = np.setdiff1d(np.unique(longitudes.flatten()), [-1])
        ds[y] = ('y', flat_latitudes)
        ds[x] = ('x', flat_longitudes)
        ds = ds.rename({y: 'lat', x: 'lon'})
        try:
            ds = ds.rename({'x': 'lon', 'y': 'lat'}).set_index({'lat': 'lat', 'lon': 'lon'})
        except:
            ds = ds.set_index({'lat': 'lat', 'lon': 'lon'})
    except:
        print('Data already converted')
    return ds

#  Main loop to create forcing files for each domain
for key, domdic in domains.items():
    locs = domdic['locs']
    dom = xr.open_dataset(domdic['path'])
    dom = convert_xy(dom).squeeze().drop('time_counter')
    
    surface = dom['e1t'] * dom['e2t']  # Needed for NEMO water flux conversion
    
    # Create empty dataset
    empty_field = xr.Dataset(
        {v: xr.DataArray(np.full(( len(dom['lon']), len(dom['lat']),len(times_all)), 
                                 fill_value=-1 if v == 'rodepth' else 0, dtype=np.float64), 
                         dims=['lon','lat','time'], 
                         coords={'lon': dom['lon'],'lat': dom['lat'],'time': times_all})
         for v in all_variables},
        coords={'lon': dom['lon'],'lat': dom['lat'],'time': times_all  }
    )

    # Fill data for each river
    for k, pos in locs.items():
        if (k.lower() not in ['danube', 'dnepr', 'bug']) or (key == 'lr' and k.lower() in ['dnepr', 'bug']):
            lat, lon = pos
            lat = dom.lat.sel(lat=lat, method='nearest').item()
            lon = dom.lon.sel(lon=lon, method='nearest').item()
            check_location(lat, lon, k.lower(), dom)
            
            if k.lower() != 'bug':
                for v in all_variables:
                    empty_field[v].loc[{'lat': lat, 'lon': lon}] = river_data[k][v].astype(float)
            else:
                for v in [v for v in all_variables if v not in ['rosaline', 'rotemper', 'rodepth']]:
                    empty_field[v].loc[{'lat': lat, 'lon': lon}] += river_data[k][v].astype(np.float64)
        else:
            # Handle rivers with multiple mouths (Danube, Dnepr, Bug)
            for i, (lat, lon) in enumerate(pos[0]):
                lat = dom.lat.sel(lat=lat, method='nearest').item()
                lon = dom.lon.sel(lon=lon, method='nearest').item()
                check_location(lat, lon, k.lower(), dom)
                
                loc = {'lat': lat, 'lon': lon}
               
                if k.lower() != 'bug':
                    for v in all_variables:
                        if v not in ['rosaline','rotemper','rodepth']:
                            empty_field[v].loc[loc] = river_data[k][v] * pos[1][i]
                        else:
                            empty_field[v].loc[loc] = river_data[k][v]
                    
                    # Handle water temperature for Danube
                    if key == 'hr':
                        temp_indices = [0, 0, 0, 1, 2]
                    else:
                        temp_indices = [0, 1, 2]
                    
                    if i < len(temp_indices):
                        temp_names = ['CHILIA ARM', 'SULINA ARM', 'SFANTU GHEORGHE ARM']
                        empty_field['rotemper'].loc[loc] = temps[temp_names[temp_indices[i]]]
                #the bug will be added on top of the dnepr cells, so we should not apply the rosaline,rotemper and rodepth variables here because we already get them from the dnepr 
                else:
                    for v in [v for v in all_variables if v not in ['rosaline', 'rotemper', 'rodepth']]:
                        empty_field[v].loc[loc] += river_data[k][v].astype(np.float64) * pos[1][i]
    
    # Convert water flux to kg/m2/s as required by nemo
    empty_field['sorunoff'] = empty_field['sorunoff'] * 1000 / surface 
    
    # CHA (excess negative charge) is computed from TA, NOS, PHO, NHS (as in original script, I don't know why it is done this way, to be discussed).
    # the CHA data in the old forcing field also don't look as if they were actually computed this way (CHA is constant)
    empty_field['CHA'] = empty_field['TA'] - empty_field['NHS'] - empty_field['NOS'] - empty_field['PHO']
    
    # Separate into physics and bgc
    phys = empty_field[['sorunoff', 'rosaline', 'rotemper', 'rodepth']]
    #phys['rodepth'] = phys['rodepth'].isel(time=0).drop('time') #make rodepth only 2dimensional (no change in time)
    #not needed, we can also have rodepth as a 3d field
    
    bgc = empty_field[[v for v in (all_variables + ['CHA']) if v not in ['sorunoff', 'rosaline', 'rotemper', 'rodepth']]]
    bgc['unit']=empty_field['sorunoff'] /1000*surface #also simply include the runoff (in m3/s). 
    #some namelist configurations just use the river runoff and some multiplicator in the namelist (variable 'unit')
    # Save forcing files for each year
    for year in np.unique(phys.time.dt.year):
        year_str = str(year.item())
        for data, data_type in [(phys, 'rnf'), (bgc, 'rnf_bgc')]:
            year_data = data.sel(time=year_str)
            # Set the dimension order explicitly (might be an error source in fortran)
            year_data = year_data.transpose('time','lat','lon')
            year_data=year_data.drop(['time', 'lat', 'lon'])
            year_data.time.encoding["unlimited"] = True
            filename = f'rnf_y{year_str}.nc'
            path = os.path.join(savepath, key.upper(), data_type, filename)
            year_data.to_netcdf(path, unlimited_dims=["time"],encoding={v:{'_FillValue': None} for v in year_data.data_vars},format="NETCDF4_CLASSIC")
            #year_data.to_netcdf(path, unlimited_dims=["time"],format="NETCDF4_CLASSIC")
print("Forcing file creation completed.")
