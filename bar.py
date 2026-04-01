import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import os


st.set_page_config(page_title="توقع سرطان الثدي", layout="wide")

@st.cache_resource
def train_and_get_mappings():
    
    
    file_path = os.path.join(os.getcwd(), 'Breast_Cancer.csv')
    df = pd.read_csv(file_path)
    
 
    mappings = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Status':
            df[col] = df[col].astype('category')
            mappings[col] = dict(enumerate(df[col].cat.categories))
            df[col] = df[col].cat.codes
            
    df['Status'] = df['Status'].map({'Alive': 0, 'Dead': 1})
    X = df.drop('Status', axis=1)
    y = df['Status']
    
    scaler = StandardScaler()
    num_cols = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']
    scaler.fit(X[num_cols])
    
    model = GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=200)
    model.fit(X, y)
    
    return model, scaler, mappings, X.columns.tolist()

model, scaler, mappings, col_order = train_and_get_mappings()

if model:
    st.title("🏥 نظام توقع مآل سرطان الثدي")
    st.write("أدخل البيانات بالكلمات الواضحة وسيقوم النظام بالتحليل")
    st.markdown("---")
    
    inputs = {}
    col1, col2 = st.columns(2)
    
    num_cols = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']
    
    for i, col in enumerate(col_order):
        with col1 if i % 2 == 0 else col2:
            if col in num_cols:
               
                inputs[col] = st.number_input(f"{col} (رقمي)", value=0)
            else:
                
                options_dict = mappings[col]
                
                display_label = st.selectbox(f"{col} (اختر الحالة)", list(options_dict.values()))
                
                inputs[col] = [k for k, v in options_dict.items() if v == display_label][0]

    if st.button("🚀 بدء التحليل الطبي"):
        input_df = pd.DataFrame([inputs])
        
        input_df[num_cols] = scaler.transform(input_df[num_cols])
        
        
        prediction = model.predict(input_df)
        prob = model.predict_proba(input_df)
        
        st.markdown("---")
        if prediction[0] == 1:
            st.error(f"⚠️ النتيجة المتوقعة: إيجابي (خطورة مرتفعة) - نسبة التأكد: {prob[0][1]*100:.2f}%")
        else:
            st.success(f"✅ النتيجة المتوقعة: سلبي (حالة مستقرة) - نسبة التأكد: {prob[0][0]*100:.2f}%")
