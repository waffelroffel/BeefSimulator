# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:01:47 2020

@author: minab
"""

k_b=0.4            #W/mC       Thermal conductivity of beef 
Cp_w= 4170         #J/kgC      Specific heat of water 
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
H_evap=2.3*10**6   #J/kg       Latent heat of vaporization of water 
T_0=13             #C          Initial temperature
C_0=0.75           #kg/kg      Initial moisture concentration 
D=4*10**-10        #m^2/s      Diffusion coefficient 
viscosity_w=1      #           Viscosity of water 