import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
import gsw
import pandas as pd
import numpy as np
import scipy
import matplotlib
from Haversine import haversine

#-----------------------------------------------#
filepath = "C:/Users/Ben/Desktop/SODA/nc_files/soda3.3.1_mn_ocean_reg_1980.nc"

#reading in a file
ncf = nc.Dataset(filepath,'r')
f = nc.Dataset(filepath)

f.variables.keys()
#-----------------VARIABLES---------------------#
#LATITUDE (Equator)
lats = f.variables['latitude'][:]
lat = lats[150]
lati = np.where(lats == 0.25)[0]

#LONGITUDE (Some random transect)
lons = f.variables['longitude'][:]
lon = lons[98:101]
loni = np.where((lons >= 49) & (lons <=51))[0]

#DEPTH
depths = f.variables['depth'][:]
depth = -1*depths[:]
depthi = np.where(depths == 25.219351)[0]

#TEMPERATURE
temps = f.variables['temp'][:]
temp = temps[:,:,lati,loni]

#REFERENCE TEMPERATURE C
Tr = -1.9

#VELOCITY OF WATER m/s
vels = f.variables['v'][:]
vel = vels[:,:,lati,loni]

#SALT
salts = f.variables['salt'][:]
salt = salts[:,:,lati,loni]

#SPECIFIC HEAT CAP OF OECAN WATER
Cp = 3.985 #J kg−1 K−1

# Making dz (height of each grid cell in meters):
## Width of each gridcell
aLon = np.abs(haversine(lat,lat, lon[1], lon[0], units="meters"))

d = [5] + list(depth) + [depth[49] + depth[49] - depth[48]]
    
dzlist = []
for i in range(0,50):
    dzlist.append(((d[i]+d[i+1])/2)-((d[i+1]+d[i+2])/2))
dzlist = np.asarray(dzlist)

A = dzlist * aLon    
#Reshaping A to be the same size as vel
A2 = (np.swapaxes(np.swapaxes(np.repeat(np.repeat(A,4),12).reshape(50,4,12),0,1),0,2))

# Volume Flux (m^3/s)
vF = vel * A2

#----v------GSW-CALCULATION-OF-RHO-----------v----#
# dbars
p1 = gsw.p_from_z(depth,lat)

# To get pressure from a 1-D array to a 3-D array that matches the salt and temp,
## first repeat the depths 120 times (10 times, then 12 times),
## next, reshape the array to be 3-D (50 by 10 by 12)
## last, swap the axes around so they're in the same order as salt and temp (12 by 50 by 10)
p3 = np.swapaxes(np.swapaxes(np.repeat(np.repeat(p1,4),12).reshape(50,4,12),0,1),0,2)

# conservative temp (psu, C > C)
cT = gsw.conversions.CT_from_pt(salt,temp)

# density (in-situ) (psu, C, dbars > kg/meters)
rho = gsw.rho(salt,cT,p3)

# in-situ temp (psu, C, dbars > C)
iT = gsw.t_from_CT(salt , cT, p3)

#----v----------Flux-Calculation------------v----#
h = (rho * vel * ((iT+273.15)-(Tr+273.15)) * Cp * A2)
#--(kg/m)-(m/s)----------(K-K)---------(J kg−1 K−1)-

#Sum all the heat flux values
hsum = np.sum(h)
    