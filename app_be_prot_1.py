import ruptures as rpt
from scipy import signal
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from functools import partial
"""
### PRESENTATION ### 

Here its is a custom back end incoporates robust features. 
It has been program to speed up feature extraction. 
"""
# Methods
def get_volume(flow,sampfreq):
    return signal.detrend(np.cumsum(flow*1/sampfreq))


class FeatureExtractor(object): 
    """
    Args: 
    - t_volume: float, volume local peak detetection prominence
    - t_insp: float, flow local inspiration peak height
    - t_exp: float, flow local expiration peak height (must be positive)
    - t_pep: float, threshold post expiration pause 
    - t_pip: float, threshold post inspiration pause
    """

    def __init__(self,t_volume,t_insp,t_exp,t_pep,t_pip):
        super().__init__()
        self.t_volume = t_volume
        self.t_insp = t_insp
        self.t_exp = t_exp
        self.t_pep = t_pep
        self.t_pip = t_pip

    def transform(self,data,sampfreq): 
        """
        Args:
        - data: pd.DataFrame, columns: {index: sequence_id (integer), flow: np.array, shape (nsamples,)}
        - sampfreq: integer, sampling frequency
        Return: 
        - df: pd.DatarFrame, columns {sequence_id, cycle_id, [Features]}
        """
        self.sampfreq = sampfreq
        self.data = self.preprocessing(data,sampfreq)
        df = self.data.copy()
        df['local_min'] = df['volume'].apply(lambda x: np.insert(find_peaks(-x, prominence=self.t_volume)[0],0,0)) # add indice 0 for the first inspiration
        df['tdf'] = df.apply(self.get_cycle_dataframe, axis=1)
        df = pd.concat(df['tdf'].values).reset_index()
        df = df.rename(columns={'index' : 'cycle_id'})
        df['Texp'] = df.apply(lambda row: np.argmax(self.get_subsequence('volume', row['sequence_id'],row['Tstart'],row['Tend']))+row['Tstart'],axis=1)
        df[['Tpif', 'Pif', 'TEPEP']] = df.apply(lambda row: self.get_phase_feature(self.get_subsequence('flow', row['sequence_id'],row['Tstart'].astype(int),row['Texp'].astype(int)),row['Tstart'].astype(int), self.t_insp, self.t_pep),axis=1,result_type = 'expand')
        df[['Tpef', 'Pef', 'TEPIP']] = df.apply(lambda row: self.get_phase_feature(-self.get_subsequence('flow', row['sequence_id'],row['Texp'].astype(int),row['Tend'].astype(int)),row['Texp'].astype(int), self.t_exp, self.t_pip),axis=1,result_type = 'expand')
        
        self.transformed_data = df.copy()
        print('Data Transformed')

    def get_basic_features(self): 

        # Getting basic features
        df = self.transformed_data.copy()
        df['Finsp'] = (df['Texp']-df['Tstart'])/self.sampfreq
        df['Fexp'] = (df['Tend']-df['Texp'])/self.sampfreq
        df['Fcycle'] = df['Finsp']+df['Fexp']
        df['PIP'] = (df['TEPIP']-df['Texp'])/self.sampfreq
        df['PEP'] = (df['TEPEP']-df['Tstart'])/self.sampfreq
        # change time to datetime
        df.loc[:,~df.columns.isin(['sequence_id', 'cycle_id', 'Pif', 'Pef','Finsp','Fexp','Fcycle','PIP','PEP'])] *= 1e9/self.sampfreq
        df.loc[:,~df.columns.isin(['sequence_id', 'cycle_id', 'Pif', 'Pef','Finsp','Fexp','Fcycle','PIP','PEP'])] = df.loc[:,~df.columns.isin(['sequence_id', 'cycle_id', 'Pif', 'Pef','Finsp','Fexp','Fcycle','PIP','PEP'])].apply(pd.to_datetime)
        
        return df

    def preprocessing(self,data,sampfreq): 
        """
        Aims: get signal to start at the first inspiration 

        Args:
        Data: pd.DataFrame, columns: {sequence_id: integer, flow: np.array, shape (nsamples,)}
        
        Return: 
        - df: pd.DataFrame, columns: {sequence_id: integer, flow: np.array, shape (nsamples,), volume: np.array, shape (nsamples,)}
        """
        df = data.copy()
        df['volume'] = df['flow'].apply(get_volume,args=(sampfreq,))
        df['min_peaks'] = df['volume'].apply(lambda x: signal.argrelmin(x)[0])
        df['flow'] = df.apply(lambda row: row['flow'][row['min_peaks'][0]:],axis=1)
        df['volume'] = df.apply(lambda row: row['volume'][row['min_peaks'][0]:]-row['volume'][row['min_peaks'][0]],axis=1)
        df = df[['sequence_id', 'flow', 'volume']].reset_index(drop=True)
        return df
    
    def get_cycle_dataframe(self,row):
        """
        Args: 
        - row: dataframe row, must include columns local_min, sequence_id

        Return: 
        df: pd.DataFrame, temporary dataframe with {index: cycle_id, sequence_id, Tstart: start cycle time, Tend: end cycle time }
        """
        arr = np.array(list(rpt.utils.pairwise(row['local_min'])))
        arr = np.insert(arr,0,row['sequence_id'], axis=1)
        df = pd.DataFrame(arr, columns=['sequence_id', 'Tstart', 'Tend'])
        return df

    def get_subsequence(self,type,sequence_id,start,end): 
        """
        Args: 
        type: string, volume or flow
        sequence_id: integer, sequence id
        start: integer, cylcle start time 
        end: integer, cycle end time

        Return: 
        subsequence: np.array, subsequence
        """
        if start != end: 
            subsequence = (self.data[self.data['sequence_id'] == sequence_id][type].values[0])[start:end]
        else: 
            subsequence = self.get_point(type,sequence_id,start)
        return subsequence

    def get_point(self,type,sequence_id,idx): 
        """
        Args: 
        type: string, volume or flow
        sequence_id: integer, sequence id
        idx: integer, cylcle start time 

        Return: 
        point: scalar
        """
        point = (self.data[self.data['sequence_id'] == sequence_id][type].values[0])[idx]
        return point

    def get_phase_feature(self,sequence,Tstart,t_phase,t_pause): 
        """
        Args: 
        - sequence: np.array, 
        - Tstart: integer, sequence start time
        - t_phase: float, phase local peak height
        - t_pause: float, pause threshold 

        Return: 
        - features: list
        """
        #Getting local maximun with height
        phase = find_peaks(sequence, height = t_phase)[0]

        #Getting sequence maximum 
        if phase.any(): 
            tmax_phase = phase[np.argmax(sequence[phase])]
        else: 
            tmax_phase = np.argmax(sequence)
        max_phase = sequence[tmax_phase]
        tmax_phase = tmax_phase

        #Getting pause endtime 
        try: 
            subsequence = sequence[:phase[0]]
            end_idx = np.where(subsequence < t_pause)[0][-1]
            tend_pause = end_idx
        except: 
            tend_pause = sequence.shape[0]

        return [tmax_phase+Tstart, max_phase, tend_pause+Tstart]

    def exportable_dataframe(self, features = True, times = False, remove_outliers = False): 
        """Dataframe for exportation

        Args:
            features (bool, optional): Include features. Defaults to True.
            times (bool, optional): Include times. Defaults to False.
            remove_outliers (bool, optional): Remove Outliers. Defaults to False.

        Returns:
            pd.DataFrame: Dataframe with our without ourliers.
        """
        df = self.get_basic_features()
        if remove_outliers: 
            columns = (df.dtypes != np.dtype('datetime64[ns]')).values 
            columns[:2] = False
            x = df.loc[:,columns]
            quantile = np.quantile(x, [0.25,0.75], axis=0)
            iqr = quantile[1]-quantile[0]
            lf = quantile[0]-1.5*iqr
            uf = quantile[1]+1.5*iqr
            idx = np.unique(np.where((x<lf)|(x>uf))[0])
            df = df.loc[~df.index.isin(idx)]
        
        if features*times: 
            pass
        else:
            if features: 
                columns = (df.dtypes != np.dtype('datetime64[ns]')).values 
                df = df.loc[:,columns]
            elif times: 
                columns = (df.dtypes == np.dtype('datetime64[ns]')).values 
                columns[:2] = True
                df = df.loc[:,columns]
        return df


    def get_cycle_frequency_graph(self,sequence_id = 0,agg_period = 'T'): 
        """
        Args: 
        - sequence_id: integer, sequence to display
        - agg_period: string, resample period for pandas, use pandas typo

        Return: 
        fig: plotly figure
        """
        # getting the proper dataframe
        df = self.get_basic_features()
        df = df[df['sequence_id'] == sequence_id]
        df = df.set_index('Tstart')
        df = df.resample(agg_period).agg({'Fcycle': [np.mean,np.std,np.min,np.max,'count']})
        df.columns = df.columns.levels[1]
        df = df.reset_index()

        # Figure
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x = df['Tstart'], y = df['count'],name = 'frequency'))
        fig.add_trace(go.Scatter(x = df['Tstart'], y = df['amin'], name = 'min',mode = 'lines'),secondary_y = True)
        fig.add_trace(go.Scatter(x = df['Tstart'], y = df['amax'], name = 'max',mode = 'lines'),secondary_y = True)
        fig.add_trace(go.Scatter(x = df['Tstart'], y = df['mean'], name = 'mean',mode = 'lines',line = dict(color='rgba(255,191,90,1)')),secondary_y = True)
        fig.add_trace(go.Scatter(x = df['Tstart'], y = df['mean']+df['std'], name = 'upper',mode = 'lines',line=dict(width=0), showlegend = False),secondary_y = True)
        fig.add_trace(go.Scatter(x = df['Tstart'], y = df['mean']-df['std'], name = 'lower',mode = 'lines', fill = 'tonexty',line=dict(width=0), fillcolor = 'rgba(255,191,90,0.3)', showlegend = False),secondary_y = True)
        
        fig.update_layout(
            yaxis_title='frequency',
            yaxis2 = dict(
                title = 'period (s)',
                anchor="x",
                overlaying="y",
                side="right",
            ),
            xaxis_title='time',
            title='Cycle Frequency & Cycle Period',
        )
        
        return fig

    def get_attribute_graph(self,sequence_id = 0,attribute = 'Fcycle',agg_period = 'T', title = None, ylabel = None):
        """
        Args: 
        - sequence_id: integer, sequence to display
        - attribute: string, dataframe column to use for analysis
        - agg_period: string, resample period for pandas, use pandas typo
        - title: string, graph title
        - ylabel: string, graph y label

        Return: 
        fig: plotly figure
        """ 
        # getting the proper dataframe
        df = self.get_basic_features()
        df = df[df['sequence_id'] == sequence_id]
        df = df.set_index('Tstart')
        df = df.resample(agg_period).agg({attribute: [np.mean,np.std,np.min,np.max]})
        df.columns = df.columns.levels[1]
        df = df.reset_index()
        # Figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = df['Tstart'], y = df['amin'], name = 'min',mode = 'lines'))
        fig.add_trace(go.Scatter(x = df['Tstart'], y = df['amax'], name = 'max',mode = 'lines'))
        fig.add_trace(go.Scatter(x = df['Tstart'], y = df['mean'], name = 'mean',mode = 'lines',line = dict(color='rgba(255,191,90,1)')))
        fig.add_trace(go.Scatter(x = df['Tstart'], y = df['mean']+df['std'], name = 'upper',mode = 'lines',line=dict(width=0), showlegend = False))
        fig.add_trace(go.Scatter(x = df['Tstart'], y = df['mean']-df['std'], name = 'lower',mode = 'lines', fill = 'tonexty',line=dict(width=0), fillcolor = 'rgba(255,191,90,0.3)', showlegend = False))
        
        if not title: 
            title = attribute
        if not ylabel: 
            ylabel = 'value'
        fig.update_layout(
            yaxis_title= ylabel,
            xaxis_title='time',
            title= title,
        )
        
        return fig

    def signal_display(self,sequence_id =0,start_time =0, window = 1): 
        """
        Args: 
        - sequence_id: integer, sequence to display
        - start_time: integer, starting observation time in seconds
        - window: integer, observation range in seconds

        Return: 
        - fig: plolty figure, signal representation with basic features
        """
        #selection mask
        start_time = start_time*self.sampfreq
        end_time = start_time+window*self.sampfreq
        mask = np.arange(start_time, end_time).astype(int)

        #basic signal
        from plotly.subplots import make_subplots
        df = self.data.copy()
        df = df[df['sequence_id'] == sequence_id]
        flow = df['flow'].values[0]
        volume = df['volume'].values[0]
        time = pd.to_datetime((np.arange(mask.shape[0]) + start_time)/self.sampfreq*1e9)
        fig = make_subplots(rows=2, cols=1,shared_xaxes=True,vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x = time,y = flow[mask], name = 'flow'), row = 1, col = 1)
        fig.add_trace(go.Scatter(x = time, y = volume[mask], name = 'volume'), row = 2, col = 1)

        #Pause
        fdata = self.transformed_data.copy()
        fdata = fdata[fdata['sequence_id'] == sequence_id]
        fdata = fdata[(fdata['Tstart'] >= start_time) & (fdata['Tend'] <= end_time)]
        #IEP
        x_iep = []
        y_iep = []
        for s,e in fdata[['Texp', 'TEPIP']].values: 
            idx =range(int(s),int(e))
            x_iep += idx
            x_iep.append(np.NaN)
            y_iep += flow[np.array(idx).astype(int)].tolist()
            y_iep.append(np.NaN)
        fig.add_trace(go.Scatter(x = pd.to_datetime(np.array(x_iep)/self.sampfreq*1e9),y =y_iep, mode = 'lines', name = 'PIP', line = dict(width = 10,),opacity =0.5), row =1, col = 1)
        
        #EIP
        x_eip = []
        y_eip = []
        for s,e in fdata[['Tstart', 'TEPEP']].values: 
            idx =range(int(s),int(e))
            x_eip += idx
            x_eip.append(np.NaN)
            y_eip += flow[np.array(idx).astype(int)].tolist()
            y_eip.append(np.NaN)
        fig.add_trace(go.Scatter(x = pd.to_datetime(np.array(x_eip)/self.sampfreq*1e9),y =y_eip, mode = 'lines', name = 'PEP', line = dict(width = 10,),opacity =0.5), row =1, col = 1)

        #Features Volume
        fig.add_trace(go.Scatter(x = fdata['Tstart'].apply(lambda t: pd.to_datetime(t/self.sampfreq*1e9)), y = volume[fdata['Tstart'].astype(int)],marker_color = '#FFA15A', mode = 'markers', name = 'insp start', showlegend = False, legendgroup = 'group1'), row =2, col = 1)
        fig.add_trace(go.Scatter(x = fdata['Texp'].apply(lambda t: pd.to_datetime(t/self.sampfreq*1e9)), y = volume[fdata['Texp'].astype(int)],marker_color = '#19D3F3', mode = 'markers', name = 'exp start', showlegend = False, legendgroup = 'group2'), row =2, col = 1) 

        #Features flow 
        fig.add_trace(go.Scatter(x = fdata['Tstart'].apply(lambda t: pd.to_datetime(t/self.sampfreq*1e9)), y = flow[fdata['Tstart'].astype(int)],marker_color = '#FFA15A', mode = 'markers', name = 'insp start',legendgroup = 'group1'), row =1, col = 1)
        fig.add_trace(go.Scatter(x = fdata['Texp'].apply(lambda t: pd.to_datetime(t/self.sampfreq*1e9)), y = flow[fdata['Texp'].astype(int)], marker_color = '#19D3F3',mode = 'markers', name = 'exp start',legendgroup = 'group2'), row =1, col = 1)
        fig.add_trace(go.Scatter(x = fdata['Tpif'].apply(lambda t: pd.to_datetime(t/self.sampfreq*1e9)), y = flow[fdata['Tpif'].astype(int)], mode = 'markers', name = 'Pif'), row =1, col = 1)
        fig.add_trace(go.Scatter(x = fdata['Tpef'].apply(lambda t: pd.to_datetime(t/self.sampfreq*1e9)), y = flow[fdata['Tpef'].astype(int)], mode = 'markers', name = 'Pef'), row =1, col = 1)

        fig.update_layout(
            title = 'Signal Analysis',
            yaxis = dict(title = 'flow (ml/sec)'),
            yaxis2 = dict(title = 'volume (ml)'),
            xaxis2 = dict(title = 'time')
        )
        return fig

    def display_attribute_distribution(self,attribute):
        """
        Args: 
        - attribute: sting, attribute to analyse
        - upfence: float, up value for outlier selection
        - lowfence float, low value for outlier selection

        Return: 
        - fig: violin plotly figure
        """
        df = self.get_basic_features()
        x = df[attribute].values
        fig = go.Figure(go.Violin(x = x, box_visible = True, meanline_visible = True, showlegend = False, y0 = 0))
        fig.update_layout(yaxis=dict(visible = False))
        fig.update_layout(title = f'{attribute} Distribution')

        return fig
