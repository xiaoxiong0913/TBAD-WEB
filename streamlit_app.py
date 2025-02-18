import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略 sklearn 的版本警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 加载模型和标准化器
model_path = r"svm_model.pkl"
scaler_path = r"scaler.pkl"

try:
    with open(model_path, 'rb') as model_file, open(scaler_path, 'rb') as scaler_file:
        model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)
except FileNotFoundError as e:
    st.error(f"Model or scaler file not found: {e}")
    st.stop()

# 特征名称和范围
feature_ranges = {
    "age": list(range(20, 101)),  # 年龄范围
    "CREA(μmol/L)": [round(x, 1) for x in range(30, 301)],  # 肌酐范围
    "HR": list(range(30, 181)),  # 心率范围
    "hospitalization （d）": list(range(1, 101)),  # 住院天数
    "HGB (g/L)": [round(x, 1) for x in range(50, 181)]  # 血红蛋白范围
}

normal_ranges = {
    "CREA(mg/dL)": (60.0, 110.0),  # 肌酐正常值
    "HR": (60, 100),  # 心率正常值
    "HGB (g/dL)": (120.0, 160.0)  # 血红蛋白正常值
}

# 页面布局：两列
col1, col2 = st.columns([1, 2])  # 左列1份宽度，右列2份宽度

with col1:
    # 左侧：选择面板
    st.markdown("<h4 style='margin-bottom:10px;'>Selection Panel</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:14px;'>Picking up parameters</p>", unsafe_allow_html=True)

    # 使用表单
    with st.form("selection_form"):
        age = st.selectbox('Age (years)', options=feature_ranges["age"], index=30)
        crea = st.selectbox('CREA (mg/dL)', options=feature_ranges["CREA(μmol/L)"], index=70)
        hr = st.selectbox('Heart Rate (HR, bpm)', options=feature_ranges["HR"], index=40)
        hospitalization_days = st.selectbox('Hospitalization Days', options=feature_ranges["hospitalization （d）"], index=9)
        hgb = st.selectbox('Hemoglobin (HGB, g/dL)', options=feature_ranges["HGB (g/L)"], index=70)

        # 提交按钮
        submit_button = st.form_submit_button("Predict")

with col2:
    # 右侧：调整标题大小
    st.markdown("<h3 style='margin-bottom:10px;'>3-Year Mortality Prediction for B-Type Aortic Dissection</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-bottom:10px;'>Introduction</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:14px;'>This web-based calculator was developed based on an SVM model with an AUC of 0.82 for predicting 3-year mortality in B-type aortic dissection patients.</p>", unsafe_allow_html=True)

    # 预测结果
    if submit_button:
        # 收集输入数据
        data = {
            "age": age,
            "CREA(μmol/L)": crea,
            "HR": hr,
            "hospitalization （d）": hospitalization_days,
            "HGB (g/L)": hgb
        }

        try:
            # 转换为 DataFrame
            data_df = pd.DataFrame([data], columns=feature_ranges.keys())

            # 数据标准化
            data_scaled = scaler.transform(data_df)

            # 预测
            prediction = model.predict_proba(data_scaled)[:, 1][0]

            # 显示预测结果，添加颜色区分
            if prediction >= 0.23910744828243835:  # 高风险
                st.markdown(
                    f"<span style='color:red; font-size:16px;'>Predicted Mortality Risk: **{prediction * 100:.2f}%** (High risk)</span>",
                    unsafe_allow_html=True)
                st.markdown(
                    "<span style='color:red; font-size:16px;'>High risk of mortality within 3 years.</span>",
                    unsafe_allow_html=True)
                st.write("**Recommendations:**")
            else:  # 低风险
                st.markdown(
                    f"<span style='color:green; font-size:16px;'>Predicted Mortality Risk: **{prediction * 100:.2f}%** (Low risk)</span>",
                    unsafe_allow_html=True)
                st.markdown(
                    "<span style='color:green; font-size:16px;'>Low risk of mortality within 3 years.</span>",
                    unsafe_allow_html=True)

            # 个性化建议
            for feature, value in data.items():
                if feature in normal_ranges:
                    normal_min, normal_max = normal_ranges[feature]
                    if value < normal_min:
                        st.markdown(
                            f"<span style='color:red;'>{feature}: Your value is {value}. It is below the normal range. Consider increasing it.</span>",
                            unsafe_allow_html=True)
                    elif value > normal_max:
                        st.markdown(
                            f"<span style='color:red;'>{feature}: Your value is {value}. It is above the normal range. Consider decreasing it.</span>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f"<span style='color:green;'>{feature}: Normal.</span>",
                            unsafe_allow_html=True)
            if prediction >= 0.23910744828243835:
                st.write("**Further Recommendations:**")
                st.markdown(
                    "- Regular follow-up with your healthcare provider.\n"
                    "- Consider additional imaging studies or lab tests as needed.\n"
                    "- Optimize treatment through medication or surgical intervention."
                )
        except Exception as e:
            st.error(f"Error during prediction: {e}")
