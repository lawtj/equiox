# version 0.1 2023-11-21

import config
from redcap import Project
import math
import numpy as np
import pandas as pd

#load redcap project from config.py file
def load_project(key):
    api_key = config.logins[key]
    api_url = 'https://redcap.ucsf.edu/api/'
    project = Project(api_url, api_key)
    df = project.export_records(format_type='df', record_type='flat', raw_or_label_headers='raw',raw_or_label='label', export_checkbox_labels=True, export_survey_fields=True)
    return project, df

#apply ITA
def ita(row):
    return (np.arctan((row['lab_l']-50)/row['lab_b'])) * (180/math.pi)

#reshape manual entered data into long format
def reshape_manual(df):

    reshaped=pd.DataFrame()

    for index, row in df.iterrows(): #iterate through every patient
        for i in range(1,11): #iterate through each device in every patient
            # create temp df from the row containing only device information
            t2 = row.filter(regex=f'{i}$') 
            t2 = pd.DataFrame(t2)

            #label the sample number from the index
            t2['sample_num'] = t2.index
            t2['sample_num'] = t2['sample_num'].str.extract(r'sat(\d+)') 

            #within each row, label the device
            t2['device'] = row[f'dev{i}'] 

            #within each row, label the location
            t2['probe_location'] = row[f'loc{i}'] 
            
            #etc
            t2['date'] = row['date']
            t2['session_num'] = row['session']
            t2['patient_id'] = row['patientid']

            #drop the columns not relating to saturation, device and location
            t2 = t2.drop([f'dev{i}', f'loc{i}']) 
            
            #label first column as saturation
            t2.columns.values[0] = 'saturation'

            #concatenate
            reshaped = pd.concat([reshaped, t2], axis=0)

    reshaped=reshaped[reshaped['saturation'].notnull()]
    return reshaped

#fitzpatrick scale color definitions
fpcolors = {'I - Pale white skin': '#f4d0b0',
        'II - White skin':'#e8b48f',
        'III - Light brown skin':'#d39e7c',
        'IV - Moderate brown skin':'#bb7750',
        'V - Dark brown skin':'#a55d2b',
        'VI - Deeply pigmented dark brown to black skin': '#3c201d'}
    
#monk scale color definitions
mscolors= {'A': '#f6ede4',
            'B': '#f3e7db',
            'C': '#f7ead0',
            'D': '#eadaba',
            'E': '#d7bd96',
            'F': '#a07e56',
            'G': '#825c43',
            'H': '#604134',
            'I': '#3a312a',
            'J': '#292420'}

#von lutzau scale color definitions
vlcolors={
        'Light (1-15)': 'rgb(241,231,195)',
        'Light Medium (16-21)': 'rgb(235,214,159)',
        'Dark Medium (22-28)' : 'rgb(188,151,98)',
        'Dark (29-36)': 'rgb(87,50,41)'
    }