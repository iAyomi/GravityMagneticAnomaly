#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Oct 11 19:04:54 2023

@author: Ayomitunde Isijola
"""

import math
import csv
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


req_Field_Type = ["G", "M"]
req_Model_Type = ["T", "C"]
csv_extract = []
anomalies = []
distances = []


def findSmoothedF(arg1):
    Fs_list = []
    for i in range(len(arg1)):
        if i - 3 < 0:
            A = 0
        else:
            A = arg1[i - 3]
            
        if i + 3 < len(arg1):
            B = arg1[i + 3]
        else:
            B = 0
            
        if i - 2 < 0:
            C = 0
        else:
            C = arg1[i - 2]
        
        if i + 2 < len(arg1):
            D = arg1[i + 2]
        else:
            D = 0
        
        if i - 1 < 0:
            E = 0
        else:
            E = arg1[i - 1]
        
        if i + 1 < len(arg1):
            F = arg1[i + 1]
        else:
            F = 0
        
        G = arg1[i]
            
        Fs = (1/231) * ( 5 * (A + B) - 30 * (C + D) + 75 * (E + F) + 131 * (G) )
        Fs_list.append(Fs)
            
    return Fs_list


def findMaxValues(arg1, arg2):
    F_list = []
    x_Dist_List = []
    F_max = max(arg1)
    F_min = min(arg1)
    X_max = max(arg2)
    X_min = min(arg2)
    
    F_list.extend((F_min, F_max))           
    x_Dist_List.extend((X_min, X_max))
    
    return F_list, x_Dist_List


def findInitialSolution(arg1, arg2):
    A_list = []
    b_list = []
    
    for i in range(len(arg1)):
        A_list.append( [ arg1[i] * arg2[i], arg1[i], arg2[i]**3, arg2[i]**2, arg2[i], 1 ] )
        b_list.append( arg1[i] * (arg2[i] ** 2) )
        
    if len(A_list) > 0 and len(b_list) > 0:
        
        # A = np.array(A_list).reshape((len(A_list), 6))
        A = np.array(A_list)
        b = np.array(b_list).reshape((len(b_list), 1))
        A_pinv = linalg.pinv(A)
      
        # x = linalg.solve(A, b)
        x = A_pinv @ b
    
        C1 = x[0][0]
        C2 = x[1][0]
        C3 = x[2][0]
        C4 = x[3][0]
        C5 = x[4][0]
        C6 = x[5][0]
        
        D = C1/2
        
        number1 = -1 * (C2 + (D ** 2))
        if number1 >= 0:
            H = math.sqrt( number1 )
        else:
            H = str(math.sqrt( -1 * number1 )) + 'z'
            
        M = C3
        
        C = C4 + (C1 * C3)
        
        P5 = C5 + (C * C1) + (C2 * C3)
        P6square = -1 * (((C6 + (D * P5) + (C * C2)) ** 2) / (C2 + (D ** 2)))
        number2 = (P5 ** 2) + P6square
        if number2 >= 0:
            P = math.sqrt( number2 )
        else:
            P = str(math.sqrt( -1 * number2 )) + 'z'
        
        if P6square >= 0:
            Q = math.atan( P5 / (math.sqrt(P6square)) )
        else:
            Q = "arctan(" + str(P5 / (math.sqrt(-1 * P6square))) + "z)"
        
        return D, H, M, C, P, Q
    else:
        return "Something went wrong!"
    
    
    
def newFunction(array_of_Initial_Values, array_of_Obs_Anomaly):
    H = array_of_Initial_Values[0]
    M = array_of_Initial_Values[1]
    C = array_of_Initial_Values[2]
    P = array_of_Initial_Values[4]
    Q = array_of_Initial_Values[5]
    rms_Error_List = []
    array_of_Theoretical_Anomaly = []
    array_of_i = []
    
    for j in range(1, 10):
        if j <= len(array_of_Obs_Anomaly):
            theoretical_Anomaly = (P * (((j * math.sin(Q)) + (H * math.cos(Q))) / ((j ** 2) + (H ** 2)))) + (M * j) + C
            array_of_Theoretical_Anomaly.append(theoretical_Anomaly)
            
    
    for i in range(len(array_of_Obs_Anomaly) - 1, -1):
        summation = 0
        while i > -1:
            summation += (array_of_Obs_Anomaly[i] - array_of_Theoretical_Anomaly[i]) ** 2
            i - 1
        rms_Error = math.sqrt( summation / (i + 1) )
        rms_Error_List.append(rms_Error)
        array_of_i.append(i)
        
    minimum_RMS_Error = min(rms_Error_List)
    
    return minimum_RMS_Error, array_of_i[rms_Error_List.index(minimum_RMS_Error)]



fileInput = str(input("Please input the file name: \n"))
if not ".csv" in fileInput:
    fileInput += ".csv"
with open(fileInput, 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        csv_extract.append(row)
        
        
if len(csv_extract) > 0:
    no_Of_Obs = len(csv_extract)
    for i in range(len(csv_extract)):
        distances.append(float(csv_extract[i][0]))
        anomalies.append(float(csv_extract[i][1]))
        


if no_Of_Obs < 6:
    print("Make sure your number of observations is >= 6")
else:
    while True:
        field_Type = input("What is the type of field you are investigating? Enter 'G' for gravity field or 'M' for magnetic field: \n")
        
        if field_Type not in req_Field_Type:
            print("Kindly make sure your values are either 'G' for gravity field or 'M' for magnetic field")
            continue
        elif field_Type == "G":
            print("Great! We are investigating Gravity Fields!")
            g_Const = float(input("Enter the Universal Gravitational Constant in Nm^2/kg^2: \n"))
            break
        elif field_Type == "M":
           print("Great! We are investigating Magnetic Fields!")
           mG_Int = int(input("Intensity of Earth's Magnetic Field in nT: \n"))
           mG_Inc = float(input("Inclination of Earth's Magnetic Field in degrees: \n"))
           str_Azm = float(input("Strike Azimuth of the body measured clockwise from Magnetic North in degrees: \n"))
           break  
    while True:
        model_Type = input("What is the model assumed? Enter 'T' for Thin Sheet or 'C' for Contact: \n")
        
        if model_Type not in req_Model_Type:
            print("Kindly make sure your values are either 'T' for Thin Sheet or 'C' for Contact")
            continue
        elif field_Type == "G" and model_Type == "C":
            
            # second horizontal derivative of grav_Ano list
            anomalies = np.array(anomalies)
            distances = np.array(distances)
            
            first_change_in_Anomaly = np.gradient(anomalies, distances)
            change_in_Anomaly = np.gradient(first_change_in_Anomaly, distances)
           
            break
        elif field_Type == "G" and model_Type == "T":

            # first horizontal derivative of grav_Ano list
            anomalies = np.array(anomalies)
            distances = np.array(distances)

            change_in_Anomaly = np.gradient(anomalies, distances)
            
            break
        elif field_Type == "M" and model_Type == "C":

            # first horizontal derivative of mG_Ano list
            anomalies = np.array(anomalies)
            distances = np.array(distances)

            change_in_Anomaly = np.gradient(anomalies, distances)
            
            break
        elif field_Type == "M" and model_Type == "T":
            
            # magnetic anomaly in total, vertical or horizontal field
            change_in_Anomaly = np.array(anomalies)
            
            break 
    

# =============================================================================
#     plt.plot(xpoints, ypoints)
#     plt.plot(xpoints, der, 'r')
#     plt.show()
# =============================================================================
    

    # print(change_in_Anomaly)
    # print(findSmoothedF(change_in_Anomaly))
    # print(findMaxValues(change_in_Anomaly, distances))     
    
    print(findInitialSolution(findSmoothedF(change_in_Anomaly), distances))
















# =============================================================================
#             change_in_Anomaly = []
# 
#             for i in range(1, len(anomalies)):
#                 derivative = (anomalies[i] - anomalies[i-1]) / (distances[i] - distances[i-1])
#                 change_in_Anomaly.append(derivative)
# =============================================================================