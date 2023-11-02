"""
Created on Thu Nov 11 09:04:54 2023

@author: Ayomitunde Isijola
"""

import streamlit as st
import numpy as np
import pandas as pd
import math
from scipy import linalg


def findChangeInAnomaly(arg1, arg2, arg3, arg4):
    if arg1 == "Gravity Field" and arg2 == "Thin Sheet Model":
        # first horizontal derivative of grav_Ano list
        anomalies = np.array(arg3)
        distances = np.array(arg4)
        change_in_Anomaly = np.gradient(anomalies, distances)
    elif arg1 == "Gravity Field" and arg2 == "Contact Model":
        # second horizontal derivative of grav_Ano list
        anomalies = np.array(arg3)
        distances = np.array(arg4)  
        first_change_in_Anomaly = np.gradient(anomalies, distances)
        change_in_Anomaly = np.gradient(first_change_in_Anomaly, distances)
    elif arg1 == "Magnetic Field" and arg2 == "Thin Sheet Model":
        # magnetic anomaly in total, vertical or horizontal field
        change_in_Anomaly = np.array(arg3)
    elif arg1 == "Magnetic Field" and arg2 == "Contact Model":
        # first horizontal derivative of mG_Ano list
        anomalies = np.array(arg3)
        distances = np.array(arg4)
        change_in_Anomaly = np.gradient(anomalies, distances)
    return change_in_Anomaly


def SmoothChangeInAnomaly(arg1):
    smoothed_Values_List = []
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
        smoothed_Val = (1/231) * ( 5 * (A + B) - 30 * (C + D) + 75 * (E + F) + 131 * (G) )
        smoothed_Values_List.append(smoothed_Val)
    return smoothed_Values_List



# def findMaxValues(arg1, arg2):
#     F_list = []
#     x_Dist_List = []
#     F_max = max(arg1)
#     F_min = min(arg1)
#     X_max = max(arg2)
#     X_min = min(arg2)
    
#     F_list.extend((F_min, F_max))           
#     x_Dist_List.extend((X_min, X_max))
    
#     return F_list, x_Dist_List



def calculateCoefficients(arg1, arg2):
    A_list = []
    b_list = []

    for a, b in zip(arg1, arg2):
        A_list.append( [ a * b, a, b**3, b**2, b, 1 ] )
        b_list.append( a * (b ** 2) )

    if len(A_list) > 0 and len(b_list) > 0:
        
        A = np.array(A_list)
        b = np.array(b_list).reshape((len(b_list), 1))
        A_pinv = linalg.pinv(A)
      
        x = A_pinv @ b

        C1, C2, C3, C4, C5, C6 = x.flatten()

        D = C1/2
        
        number1 = -1 * (C2 + (D ** 2))
        H = math.sqrt(number1) if number1 >= 0 else str(math.sqrt(-1 * number1)) + 'z'

        M = C3
        
        C = C4 + (C1 * C3)
        
        P5 = C5 + (C * C1) + (C2 * C3)
        P6square = -1 * (((C6 + (D * P5) + (C * C2)) ** 2) / (C2 + (D ** 2)))
        number2 = (P5 ** 2) + P6square
        P = math.sqrt(number2) if number2 >= 0 else str(math.sqrt(-1 * number2)) + 'z'
        Q = math.atan(P5 / (math.sqrt(P6square))) if P6square >= 0 else f"arctan({P5 / (math.sqrt(-1 * P6square))}z)"
        
        return [D, H, M, C, P, Q]
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
    
    


st.title('Inversion of Gravity and Magnetic Anomalies over some bodies of Simple Geometric Shape')

st.divider()
st.write("**Instructions**")
st.write('Make sure you select "Gravity Field" to resolve a Gravity Anomaly Problem or "Magnetic Filed" for Magnetic Anomaly respectively')
st.write('Prepare a .csv file with your data beforehand')
st.write('This is important! Use headers "Distance" and "GravityAnomaly" for Gravity Anomaly and "MagneticAnomaly" for Magnetic Anomaly')
st.divider()


req_Field_Type = st.selectbox(
   "What is the type of field you are investigating?",
   ("Gravity Field", "Magnetic Field"),
   index=None,
   placeholder="Select type of field...",
)
if req_Field_Type is not None:
    st.write("Great! We are investigating", req_Field_Type)


st.divider()


anomalies = None
distances = None


fileInput = st.file_uploader("Please upload the .csv file with your data", type=["csv"])
if fileInput is not None:

    dataframe = pd.read_csv(fileInput)
    st.write(dataframe)

    if len(dataframe) < 6:
        st.write("Make sure your number of observations is greater than or equal to 6")
    else:
        distances = dataframe['Distance'].tolist()

        if req_Field_Type == "Gravity Field":
            anomalies = dataframe['GravityAnomaly'].tolist()
        elif req_Field_Type == "Magnetic Field":
            anomalies = dataframe['MagneticAnomaly'].tolist()


st.divider()

if anomalies is not None and distances is not None:
    chart_data = pd.DataFrame(
    {
        "Distances, x": distances,
        "Gravity Anomaly, ∆g": anomalies,
    }
    )

    st.line_chart(chart_data, x="Distances, x", y="Gravity Anomaly, ∆g")

st.divider()


if req_Field_Type == "Gravity Field":

    univ_Gravitational_Const = st.number_input("Enter the Universal Gravitational Constant in Nm^2/kg^2", value=None, placeholder="Type a number...")
    if univ_Gravitational_Const is not None: st.write('Your Universal Gravitational Constant is ', univ_Gravitational_Const)
    st.divider()

elif req_Field_Type == "Magnetic Field":

    intensity_Magnetic_Field = st.number_input("Enter the Intensity of Earth's Magnetic Field in nT", value=None, placeholder="Type a number...")
    if intensity_Magnetic_Field is not None: st.write("Your Intensity of Earth's Magnetic Field is ", intensity_Magnetic_Field)

    inclination_Magnetic_Field = st.number_input("Enter the Inclination of Earth's Magnetic Field in degrees", value=None, placeholder="Type a number...")
    if inclination_Magnetic_Field is not None: st.write("Your Inclination of Earth's Magnetic Field is ", inclination_Magnetic_Field)

    strike_Azimuth = st.number_input("Enter the Strike Azimuth of the body measured clockwise from Magnetic North in degrees", value=None, placeholder="Type a number...")
    if strike_Azimuth is not None: st.write('Your Strike Azimuth is ', strike_Azimuth)

    st.divider()


req_Model_Type = st.selectbox(
   "What is the model assumed?",
   ("Thin Sheet Model", "Contact Model"),
   index=None,
   placeholder="Select the model assumed...",
)


if req_Field_Type is not None and req_Model_Type is not None:
    result = findChangeInAnomaly(req_Field_Type, req_Model_Type, anomalies, distances)

    smooth_Values = SmoothChangeInAnomaly(result)

    chart_data = pd.DataFrame(
    {
        "Distances, x": distances,
        "Change in Anomaly, ∆gx": smooth_Values,
    }
    )

    st.line_chart(chart_data, x="Distances, x", y="Change in Anomaly, ∆gx")


    coefficients = calculateCoefficients(smooth_Values, distances)
    for i in range(len(coefficients)):
        st.write(coefficients[i])