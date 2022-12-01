import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import streamlit as st
import requests
from io import StringIO

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
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
    'token': '887B6131ACB0E6DE715DCD8A35E839B9',
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

    ########################################
    ############## data cleaning
    ########################################
    df['age'] = df.apply(lambda x: 89 if x['age']>=90 else x['age'], axis=1) #censor ages above 90 -> 89
    df = df[df['consent_complete'] != 'Incomplete'] #exclude those with incomplete consents
    
    keeplist = df[df['blood_sample_1_complete'] == 'Complete']['study_id'] # create list of study_ids who have a valid value for blood_sample_1
    #df = df[df['study_id'].isin(keeplist)]

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

    # create long fitzpatrick
    ############## NOTE THIS IS NOT WORKING 
    ############### right now its only matching fitzpatrick to v1 visits, because thats when its recorded 
    ############### need to get a list of fitzpatrick and study id, and propagate it to all visits
    #t1 = df[['study_id','fitzpatrick']]
    #fitzpatricklong = t1.melt(id_vars='study_id', value_name='value')

    # combined df for scatterplot
    spo2so2long = spo2long.join(so2long, lsuffix='_spo2', rsuffix='_so2')
    spo2so2long = spo2so2long.join(stabilitylong['value'])
    spo2so2long = spo2so2long.join(collectionreasonlong['collection_reason'])
    #spo2so2long = spo2so2long.join(fitzpatricklong['value'], rsuffix='_fitzpatrick')
    spo2so2long['value'].fillna('None', inplace=True) #need to remove <NA> from stability list
    spo2so2long['collection_reason'].fillna('None',inplace=True) #need to remove NA from collection reasons
    #spo2so2long['value_fitzpatrick'].fillna('None', inplace=True) # need to remove NA from fitzpatrick


    #code records so that 1 = spo2 only, 2= so2 only, 3 = both, 0 = neither
    t1 = df[['study_id']+spo2list+so2list]
    t1 = t1.apply(pd.to_numeric, errors='coerce') #surprise! someone has allowed non-int data into this field
    t2=t1.copy()

    # recode values for spo2 = 1, so2 = 2
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
    st.title('EquiOx study dashboard')

    one, two, three, four = st.columns (4)
    one.metric(label='Enrolled patients', value=enrolled)
    two.metric(label='Average Age', value=averageage)
    three.metric(label='Total ABGs', value=totalabgs)
    four.metric(label='Avg ABGs per patient', value=round(totalabgs/enrolled, 1))

    ########################################
    ########################################
    ############## above the fold header
    ########################################
    one, two, three = st.columns ([4,6,2])
    with two:
        st.subheader('Frequency of Spo2 readings')
        fig = px.histogram(spo2long, x='value',  text_auto=True, labels={'value':'SpO2', 'count':'# of patients'})
        fig.update_traces(xbins=dict( # bins used for histogram
        size=1
        ))
        fig.update_layout(
            yaxis_title="# of measurements", xaxis_title="SpO2"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with one:
        st.subheader('''***Welcome to EquiOx***''')
        st.write('The EquiOx study is an FDA sponsored clinical trial.')
        st.info('How is the study going? Use the tabs below to explore our data.', icon="ℹ️")
    
    with three:
        st.subheader('')
        with st.expander('Descriptive Statistics'):
            st.table(spo2long['value'].apply(pd.to_numeric, errors='coerce').describe())

    ########################################

    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Demographics', 'Fitzpatrick', 'ABG values','Clinical status', 'Analysis'])

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
    ########### fitzpatrick
    ########################################
    with tab2:
        st.subheader('Fitzpatrick scores')

        first, left = st.columns([8,4])
        with first:
            fig = px.histogram(df, x='fitzpatrick')
            fig.update_layout(xaxis={'categoryorder':'category ascending'})
            fig.update_layout(legend=legendict)
            st.plotly_chart(fig, use_container_width=True)

        with left:
            npct('fitzpatrick')    

        st.write('### How do fitzpatrick scores correlate with race?')
        racebytone = pd.crosstab(df['race'], df['fitzpatrick'])
        st.table(racebytone)
        fig = px.imshow(racebytone, color_continuous_scale='aggrnyl', text_auto=True, aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
    ########################################
    ########### ABG values
    ########################################
    with tab3:   
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
        
        ### examine only clinical samples
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
        st.write(spo2so2long[spo2so2long['value_spo2']<90])

        lowlist = [1,8,9,14,7,5]
        t1=['study_id']
        st.write(df[df['study_id'].isin(lowlist)][t1+spo2list+so2list])

        st.write('''
        - 0 = neither spo2 nor so2. 
        - 1 = SpO2 only. 
        - 2 = sO2 only, 
        - 3 = both SpO2 and sO2
        ''')
        st.write(t2['sum'].value_counts())

        st.write(df)

        #t1 = df[df['fitzpatrick'].fillna('None')]
        t1=df[((df['enrollment_date'].notnull()) & (df['fitzpatrick'].notnull()))]
        st.write(t1)

        fig = px.ecdf(t1, x='enrollment_date', ecdfnorm=None, color='fitzpatrick')
        st.plotly_chart(fig)
