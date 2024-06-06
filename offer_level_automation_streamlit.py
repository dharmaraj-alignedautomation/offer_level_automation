# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:10:00 2024

@author: dharmaraj.dhanapal
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error, r2_score,mean_squared_error
from tqdm import tqdm
import warnings
import joblib
import streamlit as st

warnings.filterwarnings('ignore')

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


uploaded_file = st.file_uploader("Choose a CSV file",type=['csv','xlsx'],accept_multiple_files=False)
if uploaded_file is not None:
    try:
        new_duration = st.number_input('Enter the New Duration Value')
        new_svc_discount = st.number_input('Enter the New SVC Discount Value')
        new_pru = st.number_input('Enter the New PRU Value')
        new_ar = st.number_input('Enter the New AR Value')
        new_vsp= st.number_input('Enter the New VSP Value')
        if st.button('SUBMIT'):
        
            np.random.seed(42)
            
            column_names = 'DELL_INDUSTRY_TAXONOMY_L1'
            segment_col = 'Offer'
            try:
                df=pd.read_csv(uploaded_file)
            except:
                df=pd.read_excel(uploaded_file)
            column_name_unique = df[column_names].unique()
            value_df = pd.read_excel('Imputed_Education_Results.xlsx')
            
            equation_column_name = ['Revn','New_Revn','Delta','Actual DURATION','New DURATION','Actual AR','New AR','Actual VSP','New VSP','Actual Svc_discount','New Svc_discount','Actual PRU','New PRU','Equation']
            
            equation_df = pd.DataFrame(columns = equation_column_name)
            df_list = []
            
            
            for idx,column_name in enumerate(column_name_unique):
                filtered_df = df[df[column_names] == column_name]
                df_list.append((column_name,filtered_df))
                account_seg_uni = filtered_df[segment_col].unique().tolist()
                rev = 0
                new_rev = 0
                del_ = 0
                #Removing the ProSupport Flex/One
                try:
                    account_seg_uni.remove('ProSupport Flex/One')
                except:
                    continue
                #End of Removing the ProSupport Flex/One
                selected_columns=['AR','DURATION','PRU','Svc_discount','VSP','SRU']
                target = 'SRU'
                for account_seg in account_seg_uni:
                    temp_df = filtered_df[filtered_df[segment_col]==account_seg]
                    temp_df__ = temp_df[selected_columns].drop(target,axis=1)
                    temp_df_ = temp_df.copy()
                    filtered_df_ = filtered_df[filtered_df[segment_col]==account_seg]
                    final_df = filtered_df_[selected_columns]
                    target='SRU'
                    value_df_ = value_df[value_df[column_names] == column_name]
                    values__ = value_df_[value_df_[segment_col] == account_seg]
                    values = values__[selected_columns]
                    values_list = values.drop('SRU',axis=1).to_numpy()
                    model = LinearRegression()
                    np.random.seed(42)
                    X, y = final_df.drop(columns=[target]), final_df[target]
                    acs=account_seg.replace('/',' ')
                    with open(f'Model/{column_name} & {acs} model.pkl','rb') as f:   
                        model = joblib.load(f)
                        f.close()
                    
                    with open(f'Model/{column_name} & {acs} poly.pkl','rb') as f:   
                        poly = joblib.load(f)
                        f.close()
                    
                    ft_names_out = poly.get_feature_names_out()
                    
                    coefficients = model.coef_
                    intercept = np.mean(model.intercept_)
                    
                    X = poly.transform(X)
                    X = pd.DataFrame(X, columns=ft_names_out)
                    
                    equation = f"SRU = {intercept}\n"
                    # for i, col in enumerate(model.feature_names_in_):
                    SRU_Eqn = float(intercept)
                    for i, col in enumerate(ft_names_out):
                        cols = col.split(' ')
                        if len(cols)==1:
                            SRU_Eqn += round(coefficients[i],4) * X[cols[0]]
                        else:
                            SRU_Eqn += round(coefficients[i],4) * X[cols[0]] * X[cols[1]]
                    SRU_Eqn = SRU_Eqn.to_numpy()
                    temp_df['SRU_By_Eqn'] = SRU_Eqn
                #Revn_by_eqn
                    temp_df['Revn_By_Eqn'] = SRU_Eqn * temp_df['TOT_SYS_QTY_EXTRNL'].to_numpy()
                    
                    #new_AR
                    new_ar_temp = new_ar/100
                    AR = temp_df['AR'].to_numpy()
                    new_ar_temp = AR + new_ar_temp
                    ar_new = []
                    for i in new_ar_temp:
                        if i>1:
                            ar_new.append(1)
                        else:
                            ar_new.append(i)
                    temp_df['New_AR'] = ar_new
                    
                    #new_PRU
                    new_pru_temp = new_pru/100
                    PRU = temp_df['PRU'].to_numpy()
                    pru_new = PRU + new_pru_temp
                    temp_df['New_PRU'] = pru_new
                    
                    #new Svc_discount
                    new_svc_discount_temp = new_svc_discount/100
                    Svc_discount = temp_df['Svc_discount'].to_numpy()
                    svc_discount_new = Svc_discount + new_svc_discount_temp
                    temp_df['New_Svc_discount'] = svc_discount_new
                    
                    #new Svc_discount
                    new_vsp_temp = new_vsp
                    VSP = temp_df['VSP'].to_numpy()
                    new_vsp_temp = new_vsp_temp+VSP
                    vsp_new = []
                    for i in new_vsp_temp:
                        if i>100:
                            vsp_new.append(100)
                        else:
                            vsp_new.append(i)
                    temp_df['New_VSP'] = vsp_new
                    
                    #new DURATION
                    new_duration_temp = new_duration
                    DURATION = temp_df['DURATION'].to_numpy()
                    duration_new = DURATION + new_duration_temp
                    temp_df['New_DURATION'] = duration_new
                    
                    #New_SRU_By_Eqn
                    temp_df_['AR'] = ar_new
                    temp_df_['PRU'] = pru_new
                    temp_df_['Svc_discount'] = svc_discount_new
                    temp_df_['DURATION'] = duration_new
                    temp_df_['VSP'] = vsp_new
                    
                    final_df_temp = temp_df_[selected_columns]
                    X = final_df_temp.drop(columns=[target])
                    X = poly.transform(X)
                    X = pd.DataFrame(X, columns=ft_names_out)
                    
                    SRU_Eqn_ = float(intercept)
                    for i, col in enumerate(ft_names_out):
                        cols = col.split(' ')
                        print(cols)
                        if len(cols)==1:
                            SRU_Eqn_ += round(coefficients[i],4) * X[cols[0]]
                        else:
                            SRU_Eqn_ += round(coefficients[i],4) * X[cols[0]] * X[cols[1]]
                    SRU_Eqn_ = SRU_Eqn_.to_numpy()
                    
                    #Revn_by_eqn
                    temp_df['New_Revn_By_Eqn'] = SRU_Eqn_ * temp_df['TOT_SYS_QTY_EXTRNL'].to_numpy()
                    df_list.append((column_name + ' & ' + account_seg,temp_df))
                    
                    #Equation
                    equation_list = []
                    #equation_list.append(column_name+' & '+account_seg)
                    Revn = sum(temp_df['Revn_By_Eqn'].to_numpy())
                    equation_list.append(Revn)
                    rev+=Revn
                    New_Revn = sum(temp_df['New_Revn_By_Eqn'].to_numpy())
                    equation_list.append(New_Revn)
                    new_rev += New_Revn
                    Delta = New_Revn - Revn
                    del_ += Delta
                    equation_list.append(Delta)
                    equation_list.append(np.mean(temp_df['DURATION'].to_numpy()))
                    equation_list.append(np.mean(temp_df['New_DURATION'].to_numpy()))
                    equation_list.append(np.mean(temp_df['AR'].to_numpy()))
                    equation_list.append(np.mean(temp_df['New_AR'].to_numpy()))
                    equation_list.append(np.mean(temp_df['VSP'].to_numpy()))
                    equation_list.append(np.mean(temp_df['New_VSP'].to_numpy()))
                    equation_list.append(np.mean(temp_df['Svc_discount'].to_numpy()))
                    equation_list.append(np.mean(temp_df['New_Svc_discount'].to_numpy()))
                    equation_list.append(np.mean(temp_df['PRU'].to_numpy()))
                    equation_list.append(np.mean(temp_df['New_PRU'].to_numpy()))
                    equation_list.append("Complete Linear Equation:" + equation)
                    equation_df.loc[column_name + ' & ' + account_seg] = equation_list
                
                    
            
                equation_df.loc[f'Potential for {column_name}'] = [rev,new_rev,del_] + [' ']*11
                equation_df.loc[" "*idx] = [' ']*14
            #equation_df.loc[2] = [None]*14writer = pd.ExcelWriter('Edu_all_Eqns_Automation.xlsx', engine='xlsxwriter')
            st.dataframe(equation_df)
            csv = convert_df(equation_df)
            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="Offer Level Automation.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.write(e)
        st.warning('Please upload a valid file', icon="⚠️")
