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
    "age": (20, 100),  # 年龄范围
    "CREA(μmol/L)": (30.0, 300.0),  # 肌酐范围
    "HR": (30, 180),  # 心率范围
    "hospitalization （d）": (1, 100),  # 住院天数
    "HGB (g/L)": (50.0, 180.0)  # 血红蛋白范围
}

normal_ranges = {
    "CREA(μmol/L)": (60.0, 110.0),  # 肌酐正常值
    "HR": (60, 100),  # 心率正常值
    "HGB (g/L)": (120.0, 160.0)  # 血红蛋白正常值
}

# 创建页面布局：左侧为介绍，右侧为仪表盘
col1, col2 = st.columns([1, 2])  # 左侧占1份宽度，右侧占2份宽度

with col1:
    # 左侧：工具介绍
    st.title('3-Year Mortality Prediction for B-Type Aortic Dissection Patients')

    st.markdown("""
    ## Introduction
    This web-based calculator was developed based on an SVM model with an AUC of 0.82 (95% CI: 0.78 to 0.86) for predicting 3-year mortality in B-type aortic dissection patients. 
    The SVM model provides robust predictions by utilizing clinical and biochemical parameters, enabling clinicians to stratify risk and guide personalized treatment plans.

    **How to use this tool:**
    - Enter the patient details in the input panel on the right.
    - The model will provide a predicted 3-year mortality risk as a percentage.
    - High-risk predictions will be accompanied by personalized recommendations to optimize patient care.

    **Disclaimer:** 
    This tool is designed for educational purposes and is not a substitute for professional medical advice. Always consult a qualified healthcare provider for patient care.
    """)

with col2:
    # 右侧：预测仪表盘
    st.markdown("## Prediction Panel")

    with st.form("prediction_form"):
        age = st.slider('Age (years)', min_value=feature_ranges["age"][0], max_value=feature_ranges["age"][1], value=50)
        crea = st.slider('CREA (μmol/L)', min_value=feature_ranges["CREA(μmol/L)"][0], max_value=feature_ranges["CREA(μmol/L)"][1], value=100.0)
        hr = st.slider('Heart Rate (HR, bpm)', min_value=feature_ranges["HR"][0], max_value=feature_ranges["HR"][1], value=70)
        hospitalization_days = st.slider(
            'Hospitalization Days',
            min_value=feature_ranges["hospitalization （d）"][0],
            max_value=feature_ranges["hospitalization （d）"][1],
            value=10
        )
        hgb = st.slider('Hemoglobin (HGB, g/L)', min_value=feature_ranges["HGB (g/L)"][0], max_value=feature_ranges["HGB (g/L)"][1], value=120.0)

        # 提交按钮
        submit_button = st.form_submit_button("Predict")

    # 风险预测阈值
    risk_threshold = 0.23910744828243835  # 模型判定高风险的阈值

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
            # 将数据转换为 DataFrame
            data_df = pd.DataFrame([data], columns=feature_ranges.keys())

            # 数据标准化
            data_scaled = scaler.transform(data_df)

            # 进行预测
            prediction = model.predict_proba(data_scaled)[:, 1][0]

            # 显示结果
            st.markdown(f"### Predicted Mortality Risk: **{prediction * 100:.2f}%**")

            if prediction >= risk_threshold:
                # 高风险提示
                st.markdown(
                    "<span style='color:red; font-size:18px;'>High risk of mortality within 3 years.</span>",
                    unsafe_allow_html=True)
                st.write("**Recommendations:**")

                # 提供个性化建议
                for feature, value in data.items():
                    if feature in normal_ranges:
                        normal_min, normal_max = normal_ranges[feature]
                        if value < normal_min:
                            st.markdown(
                                f"<span style='color:red;'>{feature}: Your value is {value}. It is below the normal range ({normal_min} - {normal_max}). Consider increasing it through appropriate measures (e.g., diet, medication).</span>",
                                unsafe_allow_html=True)
                        elif value > normal_max:
                            st.markdown(
                                f"<span style='color:red;'>{feature}: Your value is {value}. It is above the normal range ({normal_min} - {normal_max}). Consider decreasing it through lifestyle adjustments or medications.</span>",
                                unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f"<span style='color:green;'>{feature}: Your value is within the normal range ({normal_min} - {normal_max}). Maintain this level for optimal health.</span>",
                                unsafe_allow_html=True)
                st.write("**Further Recommendations:**")
                st.markdown(
                    "- Ensure regular follow-up with your healthcare provider.\n"
                    "- Consider advanced imaging studies or lab tests for further evaluation.\n"
                    "- Optimize treatment with medications or surgical interventions as needed."
                )
            else:
                # 低风险提示
                st.markdown(
                    "<span style='color:green; font-size:18px;'>Low risk of mortality within 3 years.</span>",
                    unsafe_allow_html=True)
                st.write("Your parameters are within acceptable ranges. Continue to monitor and maintain a healthy lifestyle.")

                # 显示特征状态
                for feature, value in data.items():
                    if feature in normal_ranges:
                        normal_min, normal_max = normal_ranges[feature]
                        if value < normal_min:
                            st.markdown(
                                f"<span style='color:red;'>{feature}: Your value is {value}. It is slightly below the normal range ({normal_min} - {normal_max}). Consider monitoring and consulting your doctor if needed.</span>",
                                unsafe_allow_html=True)
                        elif value > normal_max:
                            st.markdown(
                                f"<span style='color:red;'>{feature}: Your value is {value}. It is slightly above the normal range ({normal_min} - {normal_max}). Ensure follow-up to avoid complications.</span>",
                                unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f"<span style='color:green;'>{feature}: Your value is within the normal range ({normal_min} - {normal_max}). No immediate action is needed.</span>",
                                unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
