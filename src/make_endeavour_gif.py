import scipy.io as sio
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed, cpu_count
import geojson
import glob
import pygmt
from pygmt.datasets import load_earth_relief

def get_sta_info(sta_file,start,end):
	sta_df = pd.read_csv(sta_file)
	sta_df['starttimes'] = pd.to_datetime(sta_df.start.values-719529,unit='D')
	sta_df.loc[np.isinf(sta_df.end),'end'] = 800000
	sta_df['endtimes'] = pd.to_datetime(sta_df.end-719529,unit='D')
	sta_df.channel = sta_df.channel.str[0:2]+'*'
	sta_df.loc[sta_df.station == 'KEMF_W3','station'] = 'KEMF'
	sta_df = sta_df.drop_duplicates(subset=['station','channel','starttimes'])
	sta_df.loc[sta_df.location.isnull(),'location'] = ''
	sta_df = sta_df[(sta_df.starttimes <= start) & (sta_df.endtimes >= end)]
	sta_df.reset_index(drop=True,inplace=True)
	
	return sta_df
	
def get_segments(endeavour_file_dir):
	endeavour_files = glob.glob(endeavour_file_dir+'/*boundary.dat')
	en_df = pd.DataFrame()
	en_dfs = [pd.read_csv(file,sep='\s+',header=None,
		names=['lon','lat','-','dist','w','0']) for file in endeavour_files]
		
	return en_dfs
		
def plot_eqs(i,date_range,date,df,circle_size,max_transparency,freq,dgrid,sta_file,en_dfs):
	mag_circles = np.array([[0.5, 4.5, circle_size * 1.5 ** 1],
					[0.5225, 3.5, circle_size * 1.5 ** 2],
					[0.53375, 2.5, circle_size * 1.5 ** 3],
					[0.55063, 1.5, circle_size * 1.5 ** 4],
					[0.57594, 0.5, circle_size * 1.5 ** 5]])
	text_array = np.array([[2, 4.5, '1'],
					[2, 3.5, '2'],
					[2, 2.5, '3'],
					[2, 1.5, '4'],
					[2, 0.5, '5']])

	print('Generating map for '+str(date))
	start = date
	if i == len(date_range)-1:
		df_sub = df
		end = df_sub.iloc[-1]['datetime']
	else:
		end = date_range[i+1]
		df_sub = df[(df['datetime'] < end)].reset_index(drop=True)
	sta_df = get_sta_info(sta_file,start,end)
	df_sub['transparency'] = 0
	day_before = date - pd.Timedelta('1D')
	epochtimes = df_sub[(df_sub['datetime'] > day_before) & (df_sub['datetime'] < start)]
	df_sub.loc[epochtimes.index.values,'transparency'] = ((start - epochtimes['datetime']).dt.total_seconds().values  / pd.Timedelta('1D').total_seconds() * max_transparency) # FIGURE THIS OUT, transparency number should be low for recent values
	df_sub.loc[df_sub['datetime'] < day_before,'transparency'] = max_transparency
	
	fig = pygmt.Figure()
	fig.basemap(region=region, projection="M15c", frame=['a', '+t'+str(date)])
	fig.grdimage(grid=grid,cmap='gray',shading='+a-45+nt0.5+m0')
	fig.coast(shorelines='0.5p,black')
	pygmt.makecpt(cmap="viridis",
		reverse=True,
		series=[df.z.min(),df.z.max()])
	for en_df in en_dfs:
		fig.plot(
			x=en_df.lon.values,
			y=en_df.lat.values,
			pen='1p,orange'
			)
	for feature in onc_cable['features']:
		for i,coordinate in enumerate(feature['geometry']['coordinates']):
			coordinate = np.array(coordinate)
			fig.plot(
				x=coordinate[:,0],
				y=coordinate[:,1],
				pen='1p,red'
				)
	fig.plot(
		x=sta_df.lon.values,
		y=sta_df.lat.values,
		style='i0.5c',
		fill='pink',
		pen='black'
		)
	if len(df_sub) > 0:
		fig.plot(
			x=df_sub.lon.values,
			y=df_sub.lat.values,
			size=0.1 * 2**df_sub.mag.values,
			fill=df_sub.z.values,
			cmap=True,
			style="cc",
			pen="black",
			transparency=df_sub.transparency.values
			)
	with fig.inset(position="jTL+w3c/5c+o0.2c", margin=0, box='+gwhite+p1p'):
		fig.basemap(region=[0, 3, 0, 6], projection="X3c/6c", frame=True)
		fig.plot(data=mag_circles,style='cc',fill='white',pen='black')
		fig.text(text='Magnitude',x=0.4,y=5.2,justify='LB',font='14p')
		for vals in text_array:
			fig.text(text=str(vals[2]),x=float(vals[0]),y=float(vals[1]),font='14p')
	fig.colorbar(frame='af+l"Depth (km)"')
# 	fig.show(method="external")
	
	if not os.path.exists('../results/images'):
		os.mkdirs('../results/images')
	fig.savefig('../results/images/'+date.strftime('%Y-%m-%d-%H-%M-%S')+'_endeavour.png',dpi='150')

# File locations
mat_file = 'location_739317.mat' # Daily location file
geojson_file = '../data/input/RingSpur.geoJson'
endeavour_file_dir = '../data/input' # Location of Endeavour segment .dat files
location_file = '../data/raw/'+mat_file
sta_file = '../data/input/endeavour_metadata.csv'

# Input parameters
freq = '5min' # Frequency of EQ sampling, can be changed to minutes (min) or hours (H)
circle_size = 0.1 # Controls the size of earthquakes on the map, which depend on magnitude
max_transparency = 100 # How transparent do you want day old eqs? 0 (opaque) - 100 (invisible)
region_padding = 0.1 # Padding for the map region (degrees)

# Load location data
loc_dat = sio.loadmat(location_file)['location']

# Parse location data
loc_dat['nlloc_hypo']
complete = np.array([(i,loc[0][0]['complete'][0][0]) for i,loc in enumerate(loc_dat['nlloc_hypo'][0]) if loc.shape[1] > 0])
complete_ids = complete[:,0]
complete = complete[:,1]

origin = [[x[0][0] for x in loc[['ot','lat','lon','z','erh','erz']][0][0]] for loc in loc_dat[0][complete_ids]['nlloc_hypo']]
p_mag = [np.nanmedian(mag[0]) for mag in loc_dat[0][complete_ids]['PMomMagnitude']]
s_mag = [np.nanmedian(mag[0]) for mag in loc_dat[0][complete_ids]['SMomMagnitude']]
p_mags = [mag[0] for mag in loc_dat[0][complete_ids]['PMomMagnitude']]
s_mags = [mag[0] for mag in loc_dat[0][complete_ids]['SMomMagnitude']]
mags = [np.nanmedian(np.concatenate([p_mags[i],s_mags[i]])) for i in range(len(p_mags))]
data = np.array([[x[0][0] for x in loc[['rms','nwp','nws','nwr']]] for loc in loc_dat[0][complete_ids]])

# Create Pandas DataFrame for location data
df = pd.DataFrame(origin,columns=['ot','lat','lon','z','erh','erz'])
df['p_mag'] = p_mag
df['s_mag'] = s_mag
df['mag'] = mags
df[['rms','nwp','nws','nwr']] = data

# Convert MATLAB time to datetime
df['datetime'] = pd.to_datetime(df.ot - 719529,unit='D')

# Load ONC cable geojson file
with open(geojson_file) as f:
	onc_cable = geojson.load(f)

# Load Endeavour segment files
en_dfs = get_segments(endeavour_file_dir)
# data = df[['lon','lat','rel_ve','rel_vn']]

# Create bounding region for map
region = [
    df.lon.min() - region_padding,
    df.lon.max() + region_padding,
    df['lat'].min() - region_padding,
    df['lat'].max() + region_padding,
]

# Get bathymetry and create a high-resolution grid
grid = load_earth_relief(
    '03s', 
    region=region, 
    registration='gridline'
    )
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 70])

# Get range of times by hour.
date_range = pd.date_range(start=df['datetime'].min().round('H'), end=df['datetime'].max().round('H'), freq=freq, inclusive='both')

# Parallel process the data, you may need to change the n_jobs value depending on how many
# cores you have. I have 20...
Parallel(n_jobs=int(cpu_count()/2))(delayed(plot_eqs)(i,date_range,date,df,circle_size,max_transparency,
	freq,dgrid,sta_file,en_dfs) for i,date in enumerate(date_range))

print('Combining maps into a GIF')
# This requires ImageMagick which can be installed with Homebrew (brew install ImageMagick)
# Use the -delay command to increase the animation time
# os.system("convert -delay 60 images/*.png images/eqs.gif")
os.system("convert ../results/images/*.png ../results/images/eqs_"+date_range[0].strftime('%Y%m%d')+'_'+freq+".gif")
# Remove PNG files
[os.remove(file) for file in glob.glob('../results/images/*.png')]

### Code for loading 'event' .mat files
# dat = sio.loadmat('/Volumes/SeaJade 2 Backup/ONC/Programs/EndeavourAutoLocate/output/event_739317.mat')['event']
# ev_type = dat['type']
# ev_type = np.array([ev[0][0] for ev in ev_type])
# counts = np.unique(ev_type,return_counts=True)
# print(counts)
