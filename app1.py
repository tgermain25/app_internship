import pandas as pd 
import numpy as np 
import streamlit as st
import plotly.graph_objects as go
import base64
from app_be_prot_1 import FeatureExtractor


### USEFUL FUNCTION ###
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

@st.cache
def get_data(filename, start =0, end =0): 
    if end==0: 
        end = None
    data = np.loadtxt(filename)
    data = data[start:-end,0]
    data = pd.DataFrame([[0,data]], columns=['sequence_id', 'flow'])
    return data
@st.cache
def transform(data,sampfreq, t_volume,t_insp,t_exp,t_pep,t_pip):
    fe = FeatureExtractor(t_volume,t_insp,t_exp,t_pep,t_pip)
    fe.transform(data, sampfreq)
    return fe

### APP Sidebar ### 
st.sidebar.title('ResApp')
task = st.sidebar.selectbox('select a task', ['Individual Signal Analysis','Application Documentation'])
st.sidebar.header('Feature Extraction Parameters:')
sampfreq = st.sidebar.number_input('Sampling Frequency:', min_value =0)
r_start = st.sidebar.number_input('Remove First Seconds', min_value=0)
r_end = st.sidebar.number_input('Remove Last Seconds', min_value=0)
t_volume = st.sidebar.number_input('Volume Prominence', min_value=0.00)
t_insp = st.sidebar.number_input('Valid Inspiration Peak Height', min_value=0.00)
t_pep = st.sidebar.number_input('Post Expiration Pause Threshold', min_value=0.00)
t_exp = st.sidebar.number_input('Valid Expiration Peak Height', min_value=0.00)
t_pip = st.sidebar.number_input('Post Inspiration Pause Threshold', min_value=0.00)

if task == 'Individual Signal Analysis': 
    st.title('Individual Signal Analysis')
    st.header('Upload Data')

    filename = st.file_uploader('')
    
    set_extraction_param = st.checkbox('Have you select a file and set feature extraction parameters ?')
    if not set_extraction_param:
        st.stop() 

    with st.spinner('Loadind Data...'): 
        data = get_data(filename, int(r_start*sampfreq), int(r_end*sampfreq))
        fe = transform(data,sampfreq,t_volume,t_insp,t_exp,t_pep,t_pip)
        st.success('Done')

    ### DATA ANALYSIS ### 
    st.header('Individual Analysis')

    sequence_id = 0

    df = fe.get_basic_features()
    df = df[df['sequence_id'] == sequence_id]

    with st.beta_expander('Signal'):
        start_time = st.number_input('Start time (sec)',value =0)
        window = st.number_input('Window size (sec)', value = 1)
        st.plotly_chart(fe.signal_display(sequence_id,start_time,window))

    with st.beta_expander('Cycle Frequency & Period'): 
        st.plotly_chart(fe.get_cycle_frequency_graph(sequence_id))
        min_cycle = df[df['Fcycle'] == df['Fcycle'].min()]['Tstart']
        max_cycle = df[df['Fcycle'] == df['Fcycle'].max()]['Tstart']

    with st.beta_expander('Other Features'): 
        attribute = st.selectbox('Feature', ('Pif', 'Pef', 'Finsp', 'Fexp', 'PIP', 'PEP'))
        st.plotly_chart(fe.get_attribute_graph(sequence_id,attribute))
        min_att = df[df[attribute] == df[attribute].min()]['Tstart']
        max_att = df[df[attribute] == df[attribute].max()]['Tstart']

    with st.beta_expander('Feature Distribution Analysis'): 
        attribute = st.selectbox('Feature', ('Fcycle','Pif', 'Pef', 'Finsp', 'Fexp', 'PIP', 'PEP'))
        x = df[attribute].values
        quantile = np.quantile(x, [0.25,0.75])
        iqr = quantile[1]-quantile[0]
        lf = quantile[0]-1.5*iqr
        uf = quantile[1]+1.5*iqr
        lowfence,upfence = st.slider("Label", float(np.min(df[attribute].values)), float(np.max(df[attribute].values)), (float(lf), float(uf)))
        mask = np.where((x <lf)|(x > uf))[0]
        poutliers = x[mask]
        outmask = np.where((poutliers < lowfence)+(poutliers > upfence))[0]
        outliers = poutliers[outmask]
        fig = fe.display_attribute_distribution(attribute)
        fig.add_vline(x = lowfence, line_dash = 'dash')
        fig.add_vline(x = upfence,line_dash = 'dash')
        fig.add_trace(go.Scatter(x = outliers, y = np.zeros_like(outliers), mode = 'markers', name = 'selected outliers'))
        
        st.plotly_chart(fig)

        selected_outlier = st.selectbox('select outlier', options=np.sort(outliers))
        outlier_start_time = df[(df['sequence_id'] == sequence_id) & (df[attribute] == selected_outlier)]['Tstart'].values[0]
        st.write(f"Outlier Cycle Start Time: {outlier_start_time}")

        st.plotly_chart(fe.signal_display(sequence_id,int(outlier_start_time)/1e9-1,3))

    with st.beta_expander('Download Dataframe'): 
        with_features = st.checkbox('Include features')
        with_time = st.checkbox('Include times')
        remove_outliers = st.checkbox('Remove outliers')
        if st.button('Download'):
            if (with_features or with_time): 
                expdf = fe.exportable_dataframe(with_features,with_time,remove_outliers)
                st.markdown(get_table_download_link(expdf), unsafe_allow_html=True)

            else: 
                st.warning('you must select an option')

        
elif task == 'Cohort Signal Transform': 
    st.title('Cohort Signal Transform')
    st.write('Under development')

elif task == 'Application Documentation': 
    with st.beta_expander('Features and Time Description'):
        st.subheader('Respiratory Cycle Description')
        st.image("code\\app_feature_plot.png")
        st.subheader('Features')
        st.write('**PEP**: Post Expiration Pause')
        st.write('**PIP**: Post Inspiration Pause')
        st.write('**Active Inspiration**: Inspiration from the end of PEP until the start of PIP')
        st.write('**Active Expiration**: Expiration from the end of PIP until the start of PEP')
        st.write('**Full Inspiration**: PEP + Active Inspiration')
        st.write('**Full Expiration**: PIP + Active Expiration')
        st.write('**Pif**: Maximum inspiration flow')
        st.write('**Pef**: Maximum expiration flow')
        st.write('**Valid Inspiration Peak**: Valid  local  inspiration  peaks  which  are  defined using height')
        st.write('**Valid Expiration Peak**: Valid  local  expiration  peaks  which  are  defined using height')
        st.subheader('Time')
        st.write('**Tstart**: Full inspiration start time, correspond to the minimum volume')
        st.write('**TEPEP**: Time End Post Expiration Pause, last time for which PEP Threshold is reached between Tstart and time of the first Valid Inspiration Peak')
        st.write('**Tpif**: Inspiration peak time')
        st.write('**Texp**: Full expiration starting time, correspond to the local maximum volume')
        st.write('**TEPIP**: Time End Post Inspiration Pause, last time for which PIP Threshold is reached between Texp and time of the first Valid Expiration Peak')
        st.write('**Tpef**: Expiration peak time')
        st.write('**Tend**: Next full inspiration start time, correspond to the next minimum volume')

    with st.beta_expander('Individual Signal Analysis'): 
        st.subheader('Setting Workflow Description')
        st.write('2. Select task: Individual Signal Analysis')
        st.write('3. drop or select your data file')
        st.write('4. set feature extraction parameters. This is just for initialization, you will be able to modify then later on.')
        st.write('5. check the box \"Have yuo select a file and set feature extraction parameters ?\"')
        
        st.subheader('Analysis')
        st.write('Signal, cycle frequency and period, others features and feature distribution analysis are all tab to visualize data.')

        st.subheader('Dowmload dataframe') 
        st.write('this module aims to dowmload basic features and times about the file uploaded file')
        st.write('1. Set your preferences times, features and outliers.(Outliers are defined using Tukey\'s fences method with constant parameter set to 1.5.)')
        st.write('2. Click on download')
        st.write('3. Click on download csv file. (It is currently a text file with \",\" as separator.)')







    















