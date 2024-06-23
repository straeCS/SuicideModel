import pandas as pd
import numpy as np

"""
The magic numbers 4 and 14 are based on the given dataset. Please do not change them, otherwise the loaded data may not be valid.
"""
MACRO_LIST = ['Personal Income', 'GDP per Capita', 'Population', 'Temperature', 'Disaster']


# get the max value of a given feature for normalization
def get_range(data, feature):
    df_cur = data[data['Description']==feature]
    df_cur.reset_index(drop=True, inplace=True)
    values = df_cur.values[:,3:]
    return np.max(values)


# ---------- Not used by default ---------- simple 2D interpolation
def interpolation(data, df_depr):
    states = list(df_depr['GeoName'].values)
    for y_idx in range(14, len(data.columns), 2):
        year = data.columns[y_idx]
        for s_idx in range(len(states)):
            cur_y = df_depr.loc[s_idx, year]
            if np.isnan(cur_y):
                has_next = False
                for next_year_idx in range(y_idx+2, len(data.columns), 2):
                    if not np.isnan(df_depr.loc[s_idx, data.columns[next_year_idx]]):
                        has_next = True
                        break
                if not has_next:
                    cur_y = df_depr.loc[s_idx, data.columns[y_idx-2]]
                else:
                    if y_idx == 14:
                        cur_y = df_depr.loc[s_idx, data.columns[next_year_idx]]
                    else:
                        cur_y = df_depr.loc[s_idx, data.columns[y_idx-2]] + (df_depr.loc[s_idx, data.columns[next_year_idx]] - df_depr.loc[s_idx, data.columns[y_idx-2]]) / (next_year_idx - y_idx + 2) * 2
                df_depr.loc[s_idx, year] = cur_y
    return df_depr


# load data for a given Target Y
def load_data(time_step=5, target='Seriously considered attempting suicide', apply_interpolation=False):
    data = pd.read_csv('US_data.csv', index_col=None)


    # get normalization factor for each feature
    list_norm = []
    for i in range(len(MACRO_LIST)):
        list_norm.append(get_range(data, MACRO_LIST[i]))
    list_norm = np.array(list_norm)[:, np.newaxis]

    # get a dataframe for Target Y
    df_depr = data[data['Description']==target]
    df_depr.reset_index(drop=True, inplace=True)
    if apply_interpolation:
        df_depr = interpolation(data, df_depr)

    # get states that have Target depression data
    states = list(df_depr['GeoName'].values)

    # the data is every-other-year. Get the data from years with values
    for s_idx in range(len(states)):
        for y_idx in range(4, 14, 2):
            year = data.columns[y_idx]
            df_depr.loc[s_idx, year] = df_depr.loc[s_idx, data.columns[14]]
    gap_years = np.arange(4, len(data.columns), 2)
    gap_years = [data.columns[y_idx] for y_idx in gap_years]
    depr_data = df_depr[gap_years].values

    # save valid data into a dictionary (not in a matrix because there are many invalid data points due to the missing data)
    dict_save = {}
    for y_idx in range(14, len(data.columns), 2):
        year = data.columns[y_idx]
        list_state, list_depr, list_feat = [], [], []
        for s_idx in range(len(states)):

            # get Target Y for current state
            cur_y = df_depr.loc[s_idx, year]
            if np.isnan(cur_y):
                continue

            # get features for current state
            cur_df = data[data['GeoName']==states[s_idx]]
            cur_df = cur_df[gap_years]
            cur_feat = cur_df.values[:-4,int(y_idx/2)-time_step-2:int(y_idx/2)-2] / list_norm

            # ---------- Not used by default ---------- use previous Target Y as a feature if adopting interpolation. Otherwise do not use it because there are many missing data.
            if apply_interpolation:
                cur_pre_y = depr_data[s_idx:s_idx+1,int(y_idx/2)-time_step-2:int(y_idx/2)-2] / 100.0
                cur_feat = np.concatenate([cur_feat, cur_pre_y], axis=0)

            list_state.append(states[s_idx])
            list_depr.append(cur_y/100.0)
            list_feat.append(cur_feat)
        dict_save[year + '_state'] = list_state
        dict_save[year + '_depr'] = list_depr
        dict_save[year + '_feat'] = list_feat

    return dict_save



# ---------- Not used ----------get depression data, and do not load features
def load_data_suicide_only(time_step=5, target='Seriously considered attempting suicide'):
    data = pd.read_csv('US_data.csv', index_col=None)
    df_depr = data[data['Description']==target]
    df_depr.reset_index(drop=True, inplace=True)
    df_depr = interpolation(data, df_depr)

    list_states = list(df_depr['GeoName'].values)
    list_year = list(np.arange(1991, 2022, 2))
    data_depr = df_depr[[str(y) for y in list_year]].values / 100.0
    return data_depr, list_states, list_year





# load macro data only
def load_data_macro(time_step=5, target='Seriously considered attempting suicide', state='Alabama'):
    data = pd.read_csv('US_data.csv', index_col=None)


    # get normalization factor for each feature
    list_norm = []
    for i in range(len(MACRO_LIST)):
        list_norm.append(get_range(data, MACRO_LIST[i]))
    list_norm = np.array(list_norm)[:, np.newaxis]


    # the data is every-other-year. Get the data from years with values
    gap_years = np.arange(4, len(data.columns), 2)
    gap_years = [data.columns[y_idx] for y_idx in gap_years]


    # save valid data into a dictionary (not in a matrix because there are many invalid data points due to the missing data)
    dict_save = {}
    for y_idx in range(14, len(data.columns), 2):
        year = data.columns[y_idx]

        # get features for current state
        cur_df = data[data['GeoName']==state]
        cur_df = cur_df[gap_years]
        cur_feat = cur_df.values[:-4,int(y_idx/2)-time_step-2:int(y_idx/2)-2] / list_norm

        dict_save[year + '_feat'] = np.array([cur_feat], dtype=np.float32)
    return dict_save