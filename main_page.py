import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import math
import streamlit as st
import requests
from datetime import datetime as dt
from io import StringIO

def check_password():
    """Returns `True` if the user had one of the correct passwords."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            st.session_state['internal_team'] = True
            del st.session_state["password"]  # don't store password
        elif st.session_state["password"] ==  st.secrets['fdapassword']:
            st.session_state['password_correct'] = True
            st.session_state['internal_team'] =  False
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
    data = {
    'token': st.secrets['token'],
    'content': 'record',
    'format': 'csv',
    'type': 'flat',
    'csvDelimiter': '',
    'rawOrLabel': 'label',
    'rawOrLabelHeaders': 'raw',
    'exportCheckboxLabel': 'false',
    'exportSurveyFields': 'true',
    'exportDataAccessGroups': 'false',
    'returnFormat': 'json'
}
    r = requests.post('https://redcap.ucsf.edu/api/',data=data)
    df = pd.read_csv(StringIO(r.text))
    
    st.set_page_config(layout="wide")
    pd.options.plotting.backend = "plotly"
    # remove below when updated to streamlit 16
    pio.templates.default = 'simple_white'

    legendict = dict(orientation='h', 
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1)

    def hist1(j): #for quantitative variables
        fig = px.histogram(df, x=j, text_auto=True)
        fig.update_layout(bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)

    def hist2(j): #for qualitative variables
        fig = px.histogram(df, x=j, text_auto=True)
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)

    def npct(i):
        taba=df[i].value_counts()
        taba = taba.rename('n')
        tabb=df[i].value_counts(normalize=True)
        tabb = tabb.rename('%')
        tab2 = pd.concat([taba, tabb], axis=1)
        st.caption('Table of ' + i + ' descriptives')
        st.table(tab2)
        
    spo2list = ['spo2','spo2_v2','spo2_v3','spo2_v4','spo2_v5','spo2_v6', 'spo2_v7','spo2_v8','spo2_v9','spo2_v10']
    so2list = ['so2','so2_v2','so2_v3','so2_v4','so2_v5','so2_v6', 'so2_v7','so2_v8','so2_v9','so2_v10']
    so2list_research = ['research_gem_so2','research_gem_so2_v2','research_gem_so2_v3','research_gem_so2_v4','research_gem_so2_v5','research_gem_so2_v6','research_gem_so2_v7','research_gem_so2_v8','research_gem_so2_v9','research_gem_so2_v10']
    collectionreasonlist = ['abg_collection_reason',
    'abg_collection_reason_v2',
    'abg_collection_reason_v3',
    'abg_collection_reason_v4',
    'abg_collection_reason_v5',
    'abg_collection_reason_v6',
    'abg_collection_reason_v7',
    'abg_collection_reason_v8',
    'abg_collection_reason_v9',
    'abg_collection_reason_v10']
    artmaplist = ['art_map', 'art_map_v2', 'art_map_v3','art_map_v4','art_map_v5', 'art_map_v6', 'art_map_v7', 'art_map_v8','art_map_v9','art_map_v10']
    stabilitylist = ['so2_period_of_stability', 'spo2_period_of_stability_v2', 'spo2_period_of_stability_v3', 'spo2_period_of_stability_v4', 'spo2_period_of_stability_v5', 'spo2_period_of_stability_v6', 'spo2_period_of_stability_v7', 'spo2_period_of_stability_v8', 'spo2_period_of_stability_v9', 'spo2_period_of_stability_v10']
    pilist = ['perfusion', 'perfusion_v2', 'perfusion_v3', 'perfusion_v4', 'perfusion_v5',
    'perfusion_v6', 'perfusion_v7', 'perfusion_v8', 'perfusion_v9', 'perfusion_v10']
    massimo_pi_list = ['masimo_perfusion', 'masimo_perfusion_v2', 'masimo_perfusion_v3', 'masimo_perfusion_v4', 'masimo_perfusion_v5',
    'masimo_perfusion_v6', 'masimo_perfusion_v7', 'masimo_perfusion_v8', 'masimo_perfusion_v9', 'masimo_perfusion_v10']
    capture_time_list = ['capture_time','capture_time_v2','capture_time_v3','capture_time_v4','capture_time_v5',
    'capture_time_v6','capture_time_v7','capture_time_v8','capture_time_v9','capture_time_v10']
    sampleanalysis_time_list = ['time_of_so2_sample_analysi','so2_time_v2','so2_time_v3','so2_time_v4','so2_time_v5',
    'so2_time_v6','so2_time_v7','so2_time_v8','so2_time_v9','so2_time_v10']
    probelocation_list = ['probe_location', 'probe_location_v2', 'probe_location_v3', 'probe_location_v4', 'probe_location_v5', 
                          'probe_location_v6', 'probe_location_v7', 'probe_location_v8', 'probe_location_v9', 'probe_location_v10']

    msoldlist = ['ms_inner_arm','ms_fingernail','ms_surface_b','ms_surface_c','ms_forehead']
    msnewlist = ['ms_new_inner_arm','ms_new_fingernail','ms_new_dorsal','ms_new_ventral','ms_new_forehead']

    mslocations = ['Inner Arm','Fingernail','Dorsal','Ventral','Forehead']
    monkvalues = ['A','B','C','D','E','F','G','H','I','J']
    vlvalues = ["Light (1-15)", "Light Medium (16-21)","Dark Medium (22-28)","Dark (29-36)"]

    vllist = ['vl_inner_arm','vl_fingernail','vl_surface_b','vl_surface_c']

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

    vlcolors={
            'Light (1-15)': 'rgb(241,231,195)',
            'Light Medium (16-21)': 'rgb(235,214,159)',
            'Dark Medium (22-28)' : 'rgb(188,151,98)',
            'Dark (29-36)': 'rgb(87,50,41)'
        }
    
    def vlbins(row, var):
        try: 
            float(row[var])
            if row[var] >= 1 and row[var] <=15:
                return "Light (1-15)"
            elif row[var] >= 16 and row[var] <=21:
                return "Light Medium (16-21)"
            elif row[var] >= 22 and row[var] <=28:
                return "Dark Medium (22-28)"
            elif row[var] >= 29 and row[var] <= 36:
                return "Dark (29-36)"
            else:
                return np.nan
        except ValueError:
            return row[var]


    ########################################
    ############## data cleaning
    ########################################
    df['age'] = df.apply(lambda x: 89 if x['age']>=90 else x['age'], axis=1) #censor ages above 90 -> 89
    # dfa = df #retain unaltered dataframe

    #convert sample drawn timestamp to datetime
    for i in capture_time_list:
        df[i] = pd.to_datetime(df[i])

    #convert enrollment date to datetime, and calculate days since enrollment
    df['enrollment_date'] = pd.to_datetime(df['enrollment_date'])
    df['days_since_enrollment'] = dt.now() - df['enrollment_date']

    #########################
    ######### convert sample run time to datetime
    #sample run time is in redcap as hh:mm:ss, NOT a timestamp.
    #assume sample run time is the same calendar day as the sample draw time and convert to datetime
    for i in sampleanalysis_time_list:
        df[i] = pd.to_datetime(df[i], format='%H:%M:%S', errors='coerce') #convert to datetime, defaults to 1900-1-1Thh:mm:ss

    # now copy the DAY from capture_time and concat with the time from sample_analysis time
    for i, j in zip(capture_time_list,sampleanalysis_time_list):
        df[j] = df[i].dt.strftime("%Y/%m/%d")+'T'+df[j].dt.strftime('%H:%M:%S')
    
    #convert from string to datetime
    for i in sampleanalysis_time_list:
        df[i] = pd.to_datetime(df[i])
    
    #subtract start time from end time
    # for i, j in zip(capture_time_list,sampleanalysis_time_list):
    #     df['sample_analysis_timedelta'] = df[j] - df[i]
    
    #print(df['sample_analysis_timedelta'])

    def create_timedelta(row):
        for i, j in zip(capture_time_list,sampleanalysis_time_list):
            if pd.notnull(row[j]):
                return row[j] - row[i]
            else:
                continue

    df['sample_analysis_timedelta'] = df.apply(create_timedelta, axis=1)
    
    ######### END section on convert sample_run to datetime
    ##########################    

    #######################################
    ##### filter based on consent status ##
    #######################################
    st.title('EquiOx study dashboard')

    #all patients with data
    keeplist0 = df[((df['blood_sample_1_complete'] == 'Complete') & (df['skin_pigment_characterization_complete']=='Complete'))]['study_id'] 
    #patients who completed study 
    keeplist = df[((df['blood_sample_1_complete'] == 'Complete') & (df['consent_complete'] == 'Complete') & (df['skin_pigment_characterization_complete']=='Complete'))]['study_id'] 
    # incompletely consented patients with data
    keeplist2 = df[((df['blood_sample_1_complete'] == 'Complete') & (df['consent_complete'] != 'Incomplete') & (df['skin_pigment_characterization_complete']=='Complete'))]['study_id']

    if st.session_state['internal_team']:
        un, deux, trois = st.columns(3)

        with un:
            option = st.selectbox(
                'Filter dashboard by:',
                ('All patients with data','Incompletely consented patients with data', 'Patients completed study'))
        
        if option == 'All patients with data':
            df = df[df['study_id'].isin(keeplist0)]
        elif option == 'Patients completed study':
            df = df[df['study_id'].isin(keeplist)]
        elif option == 'Incompletely consented patients with data':
            df = df[df['study_id'].isin(keeplist2)]

        with deux:
            st.write('- **All patients with data:** All patients with skin pigmentation and a blood sample, regardless of consent status. Currently: ' + str(len(keeplist0)))
            st.write('- **Incompletely consented patients with data:** patients with skin pigmentation and a blood sample, whose consent is Complete or Unverified. Currently: '+ str(len(keeplist2)))
            st.write('- **Patients completed study:** All patients with data and consent status is Complete (gave personal consent, 28 days since enrollment, or died). Currently: ' + str(len(keeplist)))
    else:
        df = df[df['study_id'].isin(keeplist)]
        st.write('This dashboard is currently showing only fully consented patients.')

    ###########################calculations and dfs

    # create long spo2 df
    t1 = df[['study_id']+spo2list]
    t1[spo2list] = t1[spo2list].apply(pd.to_numeric, errors='coerce') #surprise! someone has allowed non-int data into this field
    spo2long = t1.melt(id_vars='study_id', value_name='value')

    # create long so2 df
    t1 = df[['study_id']+so2list]
    t1[so2list] = t1[so2list].apply(pd.to_numeric, errors='coerce') #surprise! someone has allowed non-int data into this field
    so2long = t1.melt(id_vars='study_id', value_name='value')

    # create long period of stability df
    t1 = df[['study_id']+stabilitylist]
    stabilitylong = t1.melt(id_vars='study_id', value_name='value')

    # create long collection reason df
    t1 = df[['study_id']+collectionreasonlist]
    collectionreasonlong = t1.melt(id_vars='study_id', value_name='collection_reason')

    # long perfusion index
    t1 = df[['study_id']+pilist]
    pilong = t1.melt(id_vars='study_id', value_name='pi')
    pilong = pilong.apply(pd.to_numeric, errors='coerce')

    t1 = df[['study_id']+massimo_pi_list]
    m_pilong = t1.melt(id_vars='study_id', value_name='mpi')
    m_pilong = m_pilong.apply(pd.to_numeric, errors='coerce')

    # long probe location
    probelong = df[['study_id']+probelocation_list].melt(id_vars='study_id', value_name='probe_loc')


    # combined df for scatterplot
    spo2so2long = spo2long.join(so2long, lsuffix='_spo2', rsuffix='_so2')
    spo2so2long = spo2so2long.join(stabilitylong['value'])
    spo2so2long = spo2so2long.join(collectionreasonlong['collection_reason'])
    spo2so2long = spo2so2long.join(pilong['pi'])
    spo2so2long = spo2so2long.join(m_pilong['mpi'])
    spo2so2long['value'].fillna('None', inplace=True) #need to remove <NA> from stability list
    spo2so2long['collection_reason'].fillna('None',inplace=True) #need to remove NA from collection reasons

    # add fitzpatrick
    t1 = df[['study_id','fitzpatrick']]
    fitzpatricklong = t1.melt(id_vars='study_id', value_name='fitzpatrick')
    t1 = fitzpatricklong[(fitzpatricklong['fitzpatrick'].notna())]
    spo2so2long = spo2so2long.merge(t1, left_on='study_id_spo2', right_on='study_id')
    spo2so2long.drop(['variable', 'study_id'], axis=1)

    #code records so that 1 = spo2 only, 2= so2 only, 3 = both, 0 = neither
    t1 = df[['study_id']+spo2list+so2list]
    t1 = t1.apply(pd.to_numeric, errors='coerce') #surprise! someone has allowed non-int data into this field
    t2=t1.copy()

    # recode values for spo2 = 1, so2 = 2
    # sum column will = 1 if they only have an spo2, =2 if only has an spo2, =3 if has both
    for i in spo2list:
        t1[i] = t1[i].apply(lambda x: 1 if x>=1 else x)
    for i in so2list:
        t1[i] = t1[i].apply(lambda x: 2 if x>=1 else x)
    # both values = 3
    t1['sum']= t1[spo2list+so2list].sum(axis=1)

    # put sum back into original dataframe, t3 so one can see the spo2/so2 values 
    t3 = t1['sum']
    t2 = t2.join(t3)
    
    #print records with ONLY so2 or ONLY spo2
    #st.write(t2[((t2['sum']==1) | (t2['sum']==2))])
    
    #summarize number of records with each kind of response
    #st.write(t1['sum'].value_counts())

    ########################################
    ############## scorecards
    ########################################
    enrolled = len(df['study_id'].value_counts())
    averageage = round(df['age'].mean(),1)
    totalabgs = len(t1[((t1['sum']==2) | (t1['sum']==3))])

    st.markdown("""---""")
    one, two, three= st.columns(3)
    one.metric(label='Total enrolled patients', value=enrolled)
    two.metric(label='Enrolled patients in last 7 days', value=len(df[df['days_since_enrollment'] <= pd.Timedelta(7, unit='d')]))
    with three:
        with st.expander('Enrollment histogram'):
            fig = px.histogram(x=df['enrollment_date'], labels={'x':'Enrollment date'})
            st.plotly_chart(fig, use_container_width=True)

    one, two, three = st.columns (3)
    #one.metric(label=''' ***:exclamation: :red[SpO2 samples <=90%] :exclamation:***''', value=len(spo2long[spo2long['value']<=90]))
    # remove below once updated to streamlit 16
    one.metric(label=''' ***:exclamation: SpO2 samples <=90% :exclamation:***''', value=len(spo2long[spo2long['value']<=90]))
    two.metric(label='Total ABGs', value=totalabgs)
    three.metric(label='Avg ABGs per patient', value=round(totalabgs/enrolled, 1))

    st.markdown("""---""")


# st.subheader('Number of enrollments over time')
#         fig = px.histogram(x=df['enrollment_date'], labels={'x':'Enrollment date'})
#         st.plotly_chart(fig, use_container_width=True)
    ########################################
    ########################################
    ############## above the fold header
    ########################################
    one, two, three = st.columns(3)

    with one:
        st.subheader('% of samples collected by time of day')
        st.caption(' ')
        fig = px.histogram(x=df[capture_time_list].melt()['value'].dt.hour, labels={'x':'Hour'}, histnorm='percent')
        st.plotly_chart(fig, use_container_width=True)

    with three:
        st.subheader('Frequency of Spo2 readings')
        st.caption(' ')
        fig = px.histogram(spo2long, x='value',  text_auto=True, labels={'value':'SpO2', 'count':'# of patients'})
        fig.update_traces(xbins=dict( # bins used for histogram
        size=1
        ))
        fig.update_layout(
            yaxis_title="# of measurements", xaxis_title="SpO2"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with two:
        st.subheader('Time until sample run')
        st.caption('Only research samples have a value for when sample is run')
        fig=px.histogram(df['sample_analysis_timedelta'].dt.seconds/60, labels={'value':'Minutes'}, histnorm='percent')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    ########################################

    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Demographics', 'Skin Pigment', 'ABG values','Clinical status', 'Analysis'])

    with tab1:
        st.subheader('Demographics')
        descriptives = df.describe()
        descriptives = descriptives[['age','bmi']].replace()

        first, left, right = st.columns([2,5,5])

        with first:
            st.subheader('')
        with left:
            st.subheader('BMI')
        with right:
            st.subheader('Age')

        first, left, right = st.columns([2,5,5])
        with first:
            st.write('**Descriptive characteristics:**')
            st.table(descriptives)

        with left:
            hist1('bmi')
            with st.expander('What do we mean?'):
                st.info('BMI is a measure of the weight to height ratio. Though not without limitations, it is a commonly used measure.')
        with right:
            hist1('age')

        st.subheader('Race')
        with st.expander('What do we mean?'):
                st.info('Race used here is a term pulled from the medical chart. It is a combination of patient reported, or sometimes staff-identified patient race')
        
        first, left = st.columns([8,4])
        with first:
            hist2('race')
        with left:
            npct('race')

        st.subheader('Ethnicity')
        one, two = st.columns([8,4])
        with one:
            hist2('ethnicity')
        with two:
            npct('ethnicity')

        one, two = st.columns (2)
        with one:
            st.subheader('Gender')
        with two:
            st.subheader('Primary Team')

        one, two, three, four = st.columns([4,2,4,2])
        with one:
            hist2('gender')

        with two:
            npct('gender')

        with three:
            hist2('care_service')
        with four:
            npct('care_service')

    ########################################
    ########### skin color
    ########################################
    with tab2:
        ## monk scale
        ## need monk category order

        st.subheader('Monk Scale')
                
        # The goal is to get all new monk measurements, and discard the old monk measurements except in those that ONLY have old monk measurements.

        monkdf = df.copy()

        for i in msoldlist+msnewlist:
            #make everything uppercase
            monkdf[i] = monkdf[i].str.upper()
        
        #drop non monk columns
        monkdf = monkdf.drop(columns=monkdf.columns[~monkdf.columns.isin(['study_id']+msnewlist+msoldlist+vllist)])

        #create boolean filter according to whether old, new, or both monk scales are present
        isold = monkdf[msoldlist].notnull().any(axis=1) & monkdf[msnewlist].isnull().all(axis=1) #must have old obs, and NOT have any new obs
        isnew = monkdf[msnewlist].notnull().any(axis=1) #must have new obs. may or may not have old obs.
        isboth = monkdf[msoldlist].notnull().any(axis=1) & monkdf[msnewlist].notnull().any(axis=1) #must have old and new obs.
        isany = monkdf[msoldlist].notnull().any(axis=1) | monkdf[msnewlist].notnull().any(axis=1) #has either old OR new obs

        #how many patients with any monk observations?
        pts_w_monk_obs = len(monkdf.loc[isany,['study_id']].value_counts())
        pts_w_old_only_obs = len(monkdf.loc[isold,['study_id']].value_counts())
        pts_w_new_obs = len(monkdf.loc[isnew,['study_id']].value_counts())

        # set msoldlist columns to np.nan, for rows where there are both old and new, effectively only keeping old monk values where there is no new monk values. 
        monkdf.loc[isboth,msoldlist] = np.nan

        # fill new monk values with old monk values where new monk values are nan
        # this ensures the new monk columns have all the data, old and new
        for old, new in zip(msoldlist,msnewlist):
            monkdf[new] = monkdf[new].fillna(monkdf[old])

        # where the new monk values are not A-I, fill with nan (gets rid of 'data not recorded for example')
        for new in msnewlist:
            monkdf[new] = np.where(monkdf[new].isin(monkvalues), monkdf[new],np.nan)

        #to make histogram facet plots, need to take monkdf and transform into long
        #melt monkdf
        skinsite = monkdf.drop(['study_id']+msoldlist, axis=1).melt()

        def renamesurface(row,col):
            if row[col] == 'vl_surface_b':
                return 'vl_dorsal'
            elif row[col] == 'vl_surface_c':
                return 'vl_ventral'
            elif row[col] == 'ms_surface_b':
                return 'ms_dorsal'
            elif row[col] == 'ms_surface_c':
                return 'ms_ventral'
            else:
                return row[col]
        
        def makefloatsfloat(row,col):
            try:
                float(row[col])
                return float(row[col])
            except ValueError:
                return row[col]

        #get rid of surface_x
        skinsite['variable'] = skinsite.apply(lambda x: renamesurface(x, 'variable'), axis=1)

        #make floats in value actually float not str
        skinsite['value'] = skinsite.apply(lambda x: makefloatsfloat(x, 'value'), axis=1)

        #create Scale column from e.g. vl_inner_arm or ms_inner_arm, returning vl or ms
        skinsite['Scale'] = skinsite['variable'].str.extract(r'(\w\w)_')

        #create measurement Site column from the part of the string after vl_ or ms_
        skinsite['Site'] = skinsite['variable'].str.extract(r'\D\D_(\w*)')

        #some sites are ms_new_forehead, or ms_forehead
        #select rows with 'new' in them, and update them to read only whatever is after 'new'
        mask = skinsite['Site'].str.contains('new')
        skinsite.loc[mask, 'Site'] = skinsite.loc[mask, 'Site'].str.extract(r'new_(\w*)', expand=False)
        skinsite['value']=skinsite.apply(lambda x: vlbins(x, 'value'), axis=1)

        ##### monk layout
        with st.expander('Monk Scale by Site, per patient'):
            st.plotly_chart(
                px.histogram(skinsite[(skinsite['Scale']=='ms') & (skinsite['value'].isin(monkvalues))].sort_values(by='value', ascending=True), 
                x='value', 
                facet_col='Site', 
                facet_col_wrap=2,
                text_auto=True,
                color='value', 
                color_discrete_map=mscolors,
                title='Histogram of Monk Scale measurements by site', height=1000).update_xaxes(title='', showticklabels=True), use_container_width=True
            )

        with st.expander("Monk scale by site, per Sample"):
            # go across the given set of columns and join them together. 
            # since each row should only have one observation, the rest of the visits should be nan
            # the strip the _nan
            def joincol(col, varlist):
                df[col] = df[varlist].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
                df[col] = df[col].str.strip('_nan')

            # the new column names, and the column lists to join
            newcombinedcols = ['pl','so2com','spo2com']
            colstocombine = [probelocation_list,so2list,spo2list]

            for i, j in zip(newcombinedcols, colstocombine):
                joincol(i,j)

            #this is the mini df with just probe location, so2 and spo2
            tdf = df[['study_id','pl','so2com','spo2com']]

            ##################
            ##############
            ########### NB THIS IS NOT A COMPLETE LIST OF FINGERS 
            ######## MORE COULD BE ADDED!! ITS NOT A DROPDOWN
            ########### NEEED TO CHECK VALUE_COUNTS() LATER

            fingerlist = ['Right Index',  'Right Ring',
            'Right Middle',
            'Left Index',
            'Left Ring',
            'Left Middle',
            'Right Thumb',
            'Left Thumb',
            'Left Pinky'
            'Right Pinky']

            earlist = [ 'Right earlobe',
            'Left earlobe']

            #consolidate the probe locations to a smaller list
            def consolidate_locations(row,col):
                if row[col] in fingerlist:
                    return "Finger"
                elif row[col] in earlist:
                    return "Ear"
                elif row[col] == 'Nare':
                    return "Nare"
                else:
                    return 'other'

            tdf['consolidated_pl'] = tdf.apply(lambda row: consolidate_locations(row, 'pl'), axis=1)

            #want to get just the skin color measurements of the patients and join them to the probe/spo2/so2 list
            #however each patient appears n times depending on how many samples they had taken
            #group monkdf by first to effectively drop NaN cells, and retain only non nans, including for the color measurments
            tdf2 = monkdf.groupby(by='study_id').first()

            #join so that tdf now has the skin color measurements from tdf2
            tdf3 = tdf.join(tdf2, on='study_id', how='left')
            xt = pd.concat([tdf3[tdf3['consolidated_pl']=="Finger"][['ms_new_dorsal']], tdf3[tdf3['consolidated_pl']=='Ear'][['ms_new_forehead']]], axis=1).melt()

            
            one, two = st.columns(2)
            with one:
                fig = px.histogram(xt.sort_values(by='value', ascending=True), 
                x='value', 
                color='value', 
                color_discrete_map=mscolors, 
                text_auto=True,
                title='Dorsal/Forehead measurements for samples taken with finger/ear probes').update_xaxes(title='Monk Scale').update_traces(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                fig = px.histogram(tdf3[tdf3['consolidated_pl']=='Finger'].sort_values(by='ms_new_dorsal', ascending=True), 
                x='ms_new_dorsal', 
                color='ms_new_dorsal', 
                color_discrete_map=mscolors, 
                text_auto=True,
                title='Dorsal monk measurement for samples taken with a finger probe').update_xaxes(title='Dorsal Monk').update_traces(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with two:
                fig = px.histogram(tdf3[tdf3['consolidated_pl']=='Ear'].sort_values(by='ms_new_forehead', ascending=True), 
                x='ms_new_forehead', 
                color='ms_new_forehead', 
                color_discrete_map=mscolors, 
                text_auto=True,
                title='Forehead Monk measurement for samples taken with a ear probe').update_xaxes(title='Forehead Monk').update_traces(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        ##### VL Layout
        st.subheader('Von Luschan')
        with st.expander('VL by site, per patient'):
            st.plotly_chart(px.histogram(skinsite[(skinsite['Scale']=='vl') & skinsite['value'].isin(vlvalues)].sort_values(by='value'), 
                x='value',
                facet_col='Site',
                facet_col_wrap=2,
                facet_row_spacing=.15,
                text_auto=True, 
                title = 'Histogram of VL measurements by site',
                color='value', color_discrete_map=vlcolors, height=750).update_traces(
                    xbins=dict(size=1)).update_xaxes(
                        categoryorder='array', 
                        categoryarray=['Light (1-15)', 'Light Medium (16-21)','Dark Medium (22-28)','Dark (29-36)'],
                        title='',
                        showticklabels=True).update_layout(
                            showlegend=False), use_container_width=True)
        
        one, two = st.columns(2)

        with one:
            monkdf['vl_dorsal_bins'] = monkdf.apply(lambda x: vlbins(x, 'vl_surface_b'), axis=1)
            t1 = pd.crosstab(monkdf['vl_dorsal_bins'], columns=monkdf['ms_new_dorsal'])
            t1 = t1.reindex(index=['Light (1-15)', 'Light Medium (16-21)', 'Dark Medium (22-28)', 'Dark (29-36)'])
            t2 = px.imshow(t1, text_auto=True, color_continuous_scale='YlOrBr').update_layout(xaxis_title='Monk Value', yaxis_title='Von Luscan Bin', title='Number of patients in each VL/Monk pair').update_coloraxes(showscale=False)
            st.plotly_chart(t2, use_container_width=True)
        
        with two:
            st.write("")

        # st.write('These figures represent one anatomic site measurement per patient. They show only the "new monk" values, except where only "old monk" values exist. For reference, there are', 
        #          pts_w_old_only_obs, 
        #          'patients with ONLY "old monk" observations (this number should not go up anymore). There are also',
        #          pts_w_new_obs, 'patients with "new monk" observations.',
        #          'In total, there are',pts_w_monk_obs,'**patients** with any monk observation.')

        # def monkhist(col, title):
        #     fig = px.histogram(monkdf.sort_values(by=col, ascending=True), x=col, color=col, color_discrete_map=mscolors, text_auto=True, title=title).update_layout(showlegend=False, xaxis_title=title)
        #     st.plotly_chart(fig)
        
        # one, two = st.columns(2)

        # with one:
        #     monkhist('ms_new_forehead', 'Monk: Forehead')
        #     # fig = px.histogram(monkdf.loc[monkdf['Site']=='Forehead'], x='Monk', color='Monk',color_discrete_map=mscolors,text_auto=True, title='Monk: Forehead').update_layout(showlegend=False)
        #     # st.plotly_chart(fig)
        #     st.write('The number of forehead samples is:',
        #         monkdf['ms_new_forehead'].value_counts().sum()
        #     )
        
        # with two:
        #     monkhist('ms_new_inner_arm', 'Monk: Inner Arm')
        #     st.write('The number of inner arm samples is:',
        #         monkdf['ms_new_inner_arm'].value_counts().sum())

        # one, two = st.columns(2)
        # with one:
        #     monkhist('ms_new_dorsal', 'Monk: Dorsal')
        #     st.write('The number of dorsal samples is:',
        #             monkdf['ms_new_dorsal'].value_counts().sum())

        # with two:
        #     
            

        # st.subheader('Von Luschan')
        
        # one, two = st.columns(2)

        # with one:
        #     df['vl_inner_arm'] = pd.to_numeric(df['vl_inner_arm'], errors='coerce')
        #     df['vl_inner_arm_bins'] = df.apply(lambda x: vlbins(x, 'vl_inner_arm'), axis=1)
            
        #     hist_vl= px.histogram(df, x='vl_inner_arm_bins', title='Von Luschan Scale: Inner Arm', text_auto=True, color='vl_inner_arm_bins', color_discrete_map=vlcolors)
        #     hist_vl.update_traces(xbins=dict(size=1)).update_xaxes(categoryorder='array', categoryarray=['Light (1-15)', 'Light Medium (16-21)','Dark Medium (22-28)','Dark (29-36)'], title='VL Inner Arm').update_layout(showlegend=False)
        #     st.plotly_chart(hist_vl, use_container_width=True)

        # with two:
        #     df['vl_fingernail'] = pd.to_numeric(df['vl_fingernail'], errors='coerce')
        #     df['vl_fingernail_bins'] = df.apply(lambda x: vlbins(x, 'vl_fingernail'), axis=1)
        #     hist_vl= px.histogram(df, x='vl_fingernail_bins', title='Von Luschan Scale: Fingernail', text_auto=True, color='vl_fingernail_bins', color_discrete_map=vlcolors)
        #     hist_vl.update_traces(xbins=dict(size=1)).update_xaxes(categoryorder='array', categoryarray=['Light (1-15)', 'Light Medium (16-21)','Dark Medium (22-28)','Dark (29-36)'], title='VL Fingernail').update_layout(showlegend=False)
        #     st.plotly_chart(hist_vl, use_container_width=True)
        
        # one, two = st.columns(2)

        # with one:
        #     df['vl_surface_b'] = pd.to_numeric(df['vl_surface_b'], errors='coerce')
        #     df['vl_dorsal_bins'] = df.apply(lambda x: vlbins(x, 'vl_surface_b'), axis=1)

        #     hist_vl = px.histogram(df, x='vl_dorsal_bins', title='Von Luschan Scale: Dorsal', text_auto=True, color='vl_dorsal_bins',color_discrete_map=vlcolors)
        #     hist_vl.update_traces(xbins=dict(size=1)).update_xaxes(categoryorder='array', categoryarray=['Light (1-15)', 'Light Medium (16-21)','Dark Medium (22-28)','Dark (29-36)'], title='VL Dorsal').update_layout(showlegend=False)
        #     st.plotly_chart(hist_vl, use_container_width=True)
        
        # with two:
        #     st.write("")

        st.subheader('Fitzpatrick scores')

        first, left = st.columns([8,4])
        with first:
            fig = px.histogram(df, x='fitzpatrick', color_discrete_map=fpcolors, color='fitzpatrick')
            fig.update_layout(xaxis={'categoryorder':'category ascending'})
            fig.update_layout(legend=legendict, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with left:
            npct('fitzpatrick')    

        st.write('### How do fitzpatrick scores correlate with race?')
        racebytone = pd.crosstab(df['race'], df['fitzpatrick'])
        st.table(racebytone)
        # fig = px.imshow(racebytone, color_continuous_scale='aggrnyl', text_auto=True, aspect='auto')
        # st.plotly_chart(fig, use_container_width=True)
    ########################################
    ########### ABG values
    ########################################
    with tab3:
        st.header('Probe location')
        t1 = px.histogram(probelong, x='probe_loc', 
             title='Probe location', 
             text_auto=True).update_layout(xaxis_categoryorder='total descending', 
                                           xaxis_title='Location', 
                                           yaxis_title='Count')
        
        st.plotly_chart(t1)

        st.markdown('---')

        ####################################spo2 and so2 layout
        st.header('SPO2 analysis')

        one, two = st.columns(2)
        with one:
            st.subheader('Number of ABGs per participant')
            t1 = df['study_id'].value_counts()
            fig = px.histogram(t1, x='study_id', text_auto=True)
            fig.update_layout(xaxis_title="# ABGs", yaxis_title="# subjects")
            st.plotly_chart(fig) 
            
        with two:
            ######## now compare spo2, and so2
            st.subheader('Paired Spo2/SO2 measurements')
            fig = px.scatter(spo2so2long, x='value_spo2', y='value_so2', labels={'value_spo2':'SpO2', 'value_so2':'SaO2'}, color='value')
            fig.add_shape(type="line",
                x0=0, y0=0, x1=100, y1=100,
                line=dict(
                    color="black",
                    width=4,
                    dash="dot",
                )
            )
            fig.update_xaxes(range=[55, 105])
            fig.update_yaxes(range=[55, 105])
            #fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
            st.plotly_chart(fig, use_container_width=False)
        
        ########## bland altman plot
        st.subheader('Bland-altman plot')
        if st.button('Show bias plots:'):

            figlist =  []
            fignames = []

            ### I have made changes to the dataframe here! so lets copy the original and set it back when done
            spo2so2long_orig = spo2so2long.copy()

            #calculate bias
            
            spo2so2long['bias'] = spo2so2long['value_spo2']-spo2so2long['value_so2']
            spo2so2long=spo2so2long[spo2so2long['bias'].notnull()]
            #mean bias
            mb = spo2so2long['bias'].mean()
            #calculate ARMS
            spo2so2long['biassq'] = spo2so2long['bias']**2

            arms = math.sqrt(spo2so2long['biassq'].sum()/len(spo2so2long['biassq']))

            def scatterops(fig, name):
                fig.update_layout(legend=legendict, template='plotly_white', xaxis_title='Hemoximeter (SaO<sub>2</sub>, %)', yaxis_title='Bias (SpO<sub>2</sub> - SaO<sub>2</sub>,%)')
                #fig.update_traces(marker=dict(opacity=0.5))
                fig.update_traces(marker=dict(symbol='circle-open'))
                fig.add_hline(y=mb, annotation_text='Mean Bias: ' + str(round(mb,1)), annotation_position='bottom left', line_dash='dash', line_color='black', line_width=2, opacity=1)
                fig.add_hline(y=arms, annotation_text='ARMS: ' + str(round(arms,1)), annotation_position='top left', line_dash='dash',  line_width=2, opacity=1)
                figlist.append(fig)
                fignames.append(name)

            one, two = st.columns(2)
            with one:
                spo2so2long = spo2so2long.sort_values(by='fitzpatrick', ascending=True)
                fig_fitz = px.scatter(spo2so2long, 
                                    x='value_so2', 
                                    y='bias', 
                                    color='fitzpatrick', 
                                    title='Pulse oximeter bias, color = Fitzpatrick scale', 
                                    color_discrete_map=fpcolors, 
                                    trendline='ols',
                                    trendline_scope='overall')
                scatterops(fig_fitz, 'fig fitz')
                fig_fitz.update_layout(legend=dict(
                    y=-0.5
                ))
                st.plotly_chart(fig_fitz)
            
            with two:
                fig_biasplot = px.scatter(spo2so2long, x='value_so2', y='bias', title='Bias plot', trendline='ols')
                scatterops(fig_biasplot, 'fig_biasplot')
                st.plotly_chart(fig_biasplot, use_container_width=False)

            one, two = st.columns(2)
            with one:
                fig_pi = px.scatter(spo2so2long, x='value_so2', y='bias', color='pi', range_color=[0,4.5], title='Bias plot, color = perfusion index')
                scatterops(fig_pi, 'fig_pi')
                st.plotly_chart(fig_pi)

            with two:
                fig_mpi = px.scatter(spo2so2long, x='value_so2', y='bias', color='mpi', range_color=[0,12], title='Bias plot, color = massimo perfusion index')
                scatterops(fig_mpi, 'fig_mpi')        
                st.plotly_chart(fig_mpi)

            figlist = [fig_fitz, fig_biasplot, fig_pi, fig_mpi]
            fignames = ['fig_fitz', 'fig_biasplot', 'fig_pi', 'fig_mpi']
            
            #import kaleido
            # for i,j in zip(figlist, fignames):
            #     i.write_image(j+'.png',width=700, height=500, engine='kaleido', scale=2)  

            fig = px.violin(spo2so2long, y='pi', x='fitzpatrick', points='all')
            fig.update_xaxes(categoryorder='category ascending')
            st.plotly_chart(fig)

            #revert back to original
            spo2so2long = spo2so2long_orig


        ######## examine only clinical samples
        st.subheader('Research vs clinical samples')
        st.write('Research only samples may sit longer at the lab, and may thus have a higher SO2. Can we compare clinical samples to research only samples?')
        with st.expander('Show me'):
            spo2so2long = spo2so2long[spo2so2long['collection_reason'] != 'None']
            fig = px.scatter(spo2so2long, x='value_spo2', y='value_so2', labels={'value_spo2':'SpO2', 'value_so2':'SaO2'},
            facet_col='collection_reason', 
            trendline='ols'
            )

            fig.add_shape(type="line",
            col = 'all',row='all',
                x0=0, y0=0, x1=100, y1=100,
                line=dict(
                    color="black",
                    width=4,
                    dash="dot",
                )
            )
            fig.update_xaxes(range=[55, 105])
            fig.update_yaxes(range=[55, 105])
            #fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
            st.plotly_chart(fig, use_container_width=True)

            results = px.get_trendline_results(fig)
            one, two = st.columns(2)
            with one:
                st.write(results.query("collection_reason == 'Clinically indicated'").px_fit_results.iloc[0].summary())
            with two:
                st.write(results.query("collection_reason == 'Research purposes only'").px_fit_results.iloc[0].summary())
        
        st.write('The coefficient is being distorted by a few outliers in the Research only group. Let us remove Spo2 < 85.')

        with st.expander('Show me'):
            spo2so2long = spo2so2long[(
                (spo2so2long['collection_reason'] != 'None') &
                (spo2so2long['value_spo2'] >=85)
                )]
            fig = px.scatter(spo2so2long, x='value_spo2', y='value_so2', labels={'value_spo2':'SpO2', 'value_so2':'SaO2'},
            facet_col='collection_reason', 
            trendline='ols'
            )

            fig.add_shape(type="line",
            col = 'all',row='all',
                x0=0, y0=0, x1=100, y1=100,
                line=dict(
                    color="black",
                    width=4,
                    dash="dot",
                )
            )
            fig.update_xaxes(range=[55, 105])
            fig.update_yaxes(range=[55, 105])
            #fig.update_yaxes(scaleanchor = "x",scaleratio = 1,)
            st.plotly_chart(fig, use_container_width=True)

            results = px.get_trendline_results(fig)
            one, two = st.columns(2)
            with one:
                st.write(results.query("collection_reason == 'Clinically indicated'").px_fit_results.iloc[0].summary())
            with two:
                st.write(results.query("collection_reason == 'Research purposes only'").px_fit_results.iloc[0].summary())

    ########################################
    ########### clinical status
    ########################################

    with tab4:
        one, two = st.columns(2)
        with one:
            st.subheader('On Vasopressors?')
        with two:
            st.subheader('Supplemental oxygen use')

        one, two, three, four = st.columns([4,2,4,2])
        with one:    
            fig = px.histogram(df, x='on_vasopressors')
            fig.update_layout(legend=legendict)
            st.plotly_chart(fig, use_container_width=True)
        with two:
            npct('on_vasopressors')
        with three:
            fig = px.histogram(df, x='supplemental_oxygen_type')
            fig.update_layout(legend=legendict)
            st.plotly_chart(fig, use_container_width=True)
        with four:
            npct('supplemental_oxygen_type')

    ########################################
    ########### validation
    ########################################

    with tab5:

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Perfusion index')
            pidf=pd.DataFrame()
            for i in pilist:
                pidf = pd.concat([pidf,df[i]], axis=0)
            
            pidf = pidf.apply(pd.to_numeric, errors='coerce')

            fig = px.histogram(pidf, marginal='box')
            #fig.update_traces(xbins=dict(size=0.2))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            ######################## massimo perfusion
            st.subheader('Massimo Perfusion Index')
            madf=pd.DataFrame()
            for i in massimo_pi_list:
                madf = pd.concat([madf,df[i]], axis=0)
            
            madf = madf.apply(pd.to_numeric, errors='coerce')

            fig = px.histogram(madf[madf<20], marginal='box')
            st.plotly_chart(fig, use_container_width=True)
        
        #################### looking more at sample analysis time
        st.write('---')
        st.subheader('Sample run time by date')
        st.write('How long does it take a sample to be run by clin labs?')


        def single_capture_time(row):
            for i in capture_time_list:
                if row[i] is not pd.NaT:
                    t1=row[i]
            return t1
        
        if st.button('Show me:'):
            df['single_capture_time'] = df.apply(single_capture_time, axis=1)
            
            df['sample_analysis_timedelta_minutes'] = df['sample_analysis_timedelta'].dt.seconds/60
            fig = px.scatter(df, x='single_capture_time', y='sample_analysis_timedelta_minutes', trendline='ewm', trendline_options=dict(halflife=5))
            fig.update_layout(title='time until sample run in minutes, by sample collection date', xaxis_title='Sample collection date', yaxis_title='Time in minutes')
            st.plotly_chart(fig, use_container_width=True)

        ################### how does clinical lab compare to research gem so2?
        st.write('---')
        st.subheader('Clinical vs Research Values')
        st.write('How well do research GEM values correlate with Clin Labs samples?')
        t1 = df[so2list_research].melt().join(df[so2list].melt(), lsuffix='_research')
        t1['value']= t1['value'].apply(pd.to_numeric, errors='coerce')
        t1['value_research']= t1['value_research'].apply(pd.to_numeric, errors='coerce')

        fig = px.scatter(t1, x='value', y='value_research', trendline='ols')
        fig.add_shape(type="line",
                x0=0, y0=0, x1=100, y1=100,
                line=dict(
                    color="black",
                    width=4,
                    dash="dot",
                )
            )
        fig.update_xaxes(range=[74, 101])
        fig.update_yaxes(range=[74, 101])
        st.plotly_chart(fig, use_container_width=True)
        results = px.get_trendline_results(fig)
        st.write(results.px_fit_results.iloc[0].summary())
        one, two = st.columns(2)
        # with one:
        #     st.write(results.query("collection_reason == 'Clinically indicated'").px_fit_results.iloc[0].summary())
        # with two:
        #     st.write(results.query("collection_reason == 'Research purposes only'").px_fit_results.iloc[0].summary())

        ####################this part finding those with/without spo2/so2 readings
        # st.write(spo2so2long[spo2so2long['value_spo2']<90])

        # lowlist = [1,8,9,14,7,5]
        # t1=['study_id']
        # st.write(df[df['study_id'].isin(lowlist)][t1+spo2list+so2list])

        # st.write('''
        # - 0 = neither spo2 nor so2. 
        # - 1 = SpO2 only. 
        # - 2 = sO2 only, 
        # - 3 = both SpO2 and sO2
        # ''')
        # st.write(t2['sum'].value_counts())

        # st.write(df)

        ######################this part for enrollment over time
        # #t1 = df[df['fitzpatrick'].fillna('None')]
        # t1=df[((df['enrollment_date'].notnull()) & (df['fitzpatrick'].notnull()))]
        # st.write(t1)

        # fig = px.ecdf(t1, x='enrollment_date', ecdfnorm=None, color='fitzpatrick')
        # st.plotly_chart(fig)
