#constants.py
#
#Reference for constants
#
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:01:47 2020

@author: minab
"""

'''General constants'''
k_m=0.4            #W/mC       Thermal conductivity of beef
cp_w= 4170         #J/kgC      Specific heat of water
rho_w= 988         #kg/m^3     Density of water 
h=33.4             #W/m^2C     Heat transfer ceofficient 
T_oven=175         #C          Oven temperature
y_p=0.21           #-          Protein composition  
y_c=0              #-          Carbohydrate composition 
y_f=0.36           #-          Fat composition 
y_w=0.74           #-          Water composition 
rho_f=920          #kg/m^3     Density of fat 
rho_p=1320         #kg/m^3     Densitu of protein 
rho_c=1600         #kg/m^3     Density of carbohydrate 
H_evap=2.3e6       #J/kg       Latent heat of vaporization of water
T_0=13             #C          Initial temperature
C_0=0.75           #kg/kg      Initial moisture concentration 
D=4e-10            #m^2/s      Diffusion coefficient
K = 1e-17          #m^2        Permeability - in range 1e-17 to 1e-19
f = 0.9            #-          Fraction of energy used for evaporation
'''Composite constants'''
cp_m = (1.6*y_c+2*y_p+2*y_f+4.2*y_w)*1e3
rho_m = 1 / (y_c/rho_c + y_p/rho_p + y_f/rho_f + y_w/rho_w)
'''Constants for Elasticity model'''
E0 = 12e3          # Pa
Em = 83e3          # Pa
En = 0.3
ED = 60
'''Constants for equilibrium water holding capacity'''
a1 = 0.745
a2 = 0.345
a3 = 30
a4 = 0.25
T_sig = 52         # ⁰C
'''Discretization constants'''
dx = 0.01          #m
dt = 0.1           #s
Lx = 0.10          #m
Ly = 0.20          #m
Lz = 0.15          #m
