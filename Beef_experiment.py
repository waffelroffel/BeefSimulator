from matplotlib import pyplot as pp
import numpy as np
#The beefs were cooked in oven with temperature 175*C until the temperature in the center reached 65*C.
#The temperature was recorded every 10s.

#Beef A
#Dimensions: 9.3 * 5.0 * 2.6 [cm * cm * cm]
#Weight before cooking: 143 +- 1 g
#Weight after cooking (13 min): 128 +- 1 g

timelist1=np.linspace(0, 13*60, 79)
T1=[14, 15, 18, 20, 22, 23, 25, 26, 26, 28, 28, 29, 30, 31, 31, 31, 32, 32, 33, 33, 34, 34, 34, 35, 35, 37, 37, 37,
   38, 39, 39, 40, 40, 40, 41, 42, 42, 42, 43, 44, 45, 45, 46, 47, 47, 48, 48, 48, 49, 49, 50, 51, 51, 51, 53,
   53, 54, 54, 54, 55, 56, 56, 56, 57, 57, 58, 59, 59, 59, 60, 61, 62, 62, 62, 62, 63, 63, 64, 65]
pp.figure()
pp.title("Beef A")
pp.scatter(timelist1,T1)
pp.xlabel("Time (s)")
pp.ylabel("Temperature in center (*C)")
pp.show()

#Beef B
#Dimensions: 10.0 * 4.6 * 2.7 [cm * cm * cm]
#Weigh before cooking: 140 +- 1 g
#Weight after cooking (14 min): 118 +- 1 g

timelist2=np.linspace(0, 14*60, 85)
T2=[16, 18, 20, 21, 22, 24, 24, 25, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 31, 31, 32, 32, 32, 33,
   34, 34, 35, 35, 36, 36, 37, 38, 38, 39, 40, 41, 41, 41, 42, 43, 43, 43, 44, 45, 45, 46, 46, 47, 48, 49, 49, 49,
   50, 51, 51, 52, 52, 52, 53, 54, 54, 55, 55, 56, 57, 57, 57, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63, 63, 64, 65]
pp.figure()
pp.title("Beef B")
pp.scatter(timelist2,T2)
pp.xlabel("Time (s)")
pp.ylabel("Temperature in center (*C)")
pp.show()

pp.figure()
pp.title("Beef A and Beef B")
pp.scatter(timelist1, T1, label="Beef A")
pp.scatter(timelist2, T2, label="Beef B")
pp.xlabel("Time (s)")
pp.ylabel("Temperature in center (*C)")
pp.legend()
pp.show()