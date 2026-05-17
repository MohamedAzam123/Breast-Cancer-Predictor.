import streamlit as st
import pickle as pk 
import pandas as pd 
import numpy as np

pipeline = pk.load(open('breast_cancer_model.sav', 'rb'))

st.title('توقع الشخص مريض بالكانسر ام لا')

race_map = {'White':2, 'Other':1, 'Black':0}
Marital_Status_map = {'Married':1, 'Single':3, 'Divorced':0, 'Widowed':4, 'Separated':2}
t_stage_map = {'T1':0, 'T2':1, 'T3':2, 'T4':3}
n_stage_map = {'N1':0, 'N2':1, 'N3':2}
sixth_stage_map = {'IIA':0, 'IIB':1, 'IIIA':2, 'IIIB':3, 'IIIC':4}
differentiate_map = {'Moderately differentiated':0, 'Poorly differentiated':1, 'Well differentiated':3, 'Undifferentiated':2}
a_stage_map = {'Regional':1, 'Distant':0}
estrogen_status_map = {'Positive':1, 'Negative':0}
progesterone_status_map = {'Positive':1, 'Negative':0}

Age = st.number_input('Age', value=50)
Race = st.selectbox('Race', options=list(race_map.keys()))
Marital_Status = st.selectbox('Marital Status', options=list(Marital_Status_map.keys()))
t_stage = st.selectbox('T Stage', options=list(t_stage_map.keys()))
n_stage = st.selectbox('N Stage', options=list(n_stage_map.keys()))
sixth_stage = st.selectbox('6th Stage', options=list(sixth_stage_map.keys()))
differentiate = st.selectbox('Differentiate', options=list(differentiate_map.keys()))
grade = st.number_input('Grade', value=2)
a_stage = st.selectbox('A Stage', options=list(a_stage_map.keys()))
tumor_size = st.number_input('Tumor Size', value=25)
estrogen_status = st.selectbox('Estrogen Status', options=list(estrogen_status_map.keys()))
progesterone_status = st.selectbox('Progesterone Status', options=list(progesterone_status_map.keys()))
regional_node_examined = st.number_input('Regional Node Examined', value=14)
reginol_node_positive = st.number_input('Reginol Node Positive', value=2)
survival_months = st.number_input('Survival Months', value=70)

con = st.button('عرض النتيجة')

if con:
    input_data = [Age, Race, Marital_Status, t_stage, n_stage, sixth_stage, differentiate, grade, a_stage, tumor_size, estrogen_status, progesterone_status, regional_node_examined, reginol_node_positive, survival_months]

    cols = ['Age', 'Race', 'Marital Status', 'T Stage ', 'N Stage', '6th Stage', 'differentiate', 'Grade', 'A Stage', 'Tumor Size', 'Estrogen Status', 'Progesterone Status', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']

    df = pd.DataFrame([input_data], columns=cols)

    df['Race'] = df['Race'].map(race_map)
    df['Marital Status'] = df['Marital Status'].map(Marital_Status_map)
    df['T Stage '] = df['T Stage '].map(t_stage_map)
    df['N Stage'] = df['N Stage'].map(n_stage_map)
    df['6th Stage'] = df['6th Stage'].map(sixth_stage_map)
    df['differentiate'] = df['differentiate'].map(differentiate_map)
    df['A Stage'] = df['A Stage'].map(a_stage_map)
    df['Estrogen Status'] = df['Estrogen Status'].map(estrogen_status_map)
    df['Progesterone Status'] = df['Progesterone Status'].map(progesterone_status_map)

    pred = pipeline.predict(df)

    if pred[0] == 1:
        st.error('النتيجة المتوقعة ايجابي والخطورة مرتفعة')
    else:
        st.success('النتيجة المتوقعة سلبي والحالة في امان')