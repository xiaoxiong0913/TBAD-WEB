import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
import subprocess
import sys

# 处理版本警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 安装所需包函数
def install(package):
    result = subprocess.run([sys.executable, "-m", "pip", "install", package, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"], check=True)
    if result.returncode == 0:
        print(f"{package} successfully installed.")
    else:
        print(f"Failed to install {package}.")

# 加载模型和标准化器
model_path = r"E:\中心实验室\XWY\B型主动脉夹层\mimic数据4模型验证最终结果\WEB\svm_model.pkl"
scaler_path = r"E:\中心实验室\XWY\B型主动脉夹层\mimic数据4模型验证最终结果\WEB\scaler.pkl"

with open(model_path, 'rb') as model_file, open(scaler_path, 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

# 特征名称和范围直接在代码中定义
feature_ranges = {
    "age": (20, 100),
    "CREA(μmol/L)": (30.0, 300.0),
    "HR": (30, 180),
    "hospitalization （d）": (1, 100),
    "HGB (g/L)": (50.0, 180.0)
}

normal_ranges = {
    "CREA(μmol/L)": (60.0, 110.0),  # 国际标准参考范围
    "HR": (60, 100),
    "HGB (g/L)": (120.0, 160.0)
}

# 创建Web应用的标题
st.title('Machine Learning-Based Prediction for 3-Year Mortality in B-Type Aortic Dissection Patients')

# 添加介绍部分
st.markdown("""
## Introduction
This web-based calculator uses an SVM model trained to predict 3-year mortality in B-type aortic dissection patients. Provide the required input parameters to calculate the risk.
""")

# 创建输入表单
st.markdown("## Input Panel")

with st.form("prediction_form"):
    age = st.slider('Age (years)', min_value=feature_ranges["age"][0], max_value=feature_ranges["age"][1], value=50)
    crea = st.slider('CREA (μmol/L)', min_value=feature_ranges["CREA(μmol/L)"][0], max_value=feature_ranges["CREA(μmol/L)"][1], value=100.0)
    hr = st.slider('Heart Rate (HR, bpm)', min_value=feature_ranges["HR"][0], max_value=feature_ranges["HR"][1], value=70)
    hospitalization_days = st.slider('Hospitalization Days', min_value=feature_ranges["hospitalization （d）"][0], max_value=feature_ranges["hospitalization （d）"][1], value=10)
    hgb = st.slider('Hemoglobin (HGB, g/L)', min_value=feature_ranges["HGB (g/L)"][0], max_value=feature_ranges["HGB (g/L)"][1], value=120.0)

    # 提交按钮
    submit_button = st.form_submit_button("Predict")

# 风险截断值
risk_threshold = 0.23910744828243835

# 当用户提交表单时
if submit_button:
    # 构建输入数据
    data = {
        "age": age,
        "CREA(μmol/L)": crea,
        "HR": hr,
        "hospitalization （d）": hospitalization_days,
        "HGB (g/L)": hgb
    }

    try:
        # 将数据转换为 DataFrame，并按特征顺序排列列
        data_df = pd.DataFrame([data], columns=feature_ranges.keys())

        # 应用标准化
        data_scaled = scaler.transform(data_df)

        # 进行预测
        prediction = model.predict_proba(data_scaled)[:, 1][0]  # 获取类别为1的预测概率

        # 显示预测结果
        st.write(f'Prediction: {prediction * 100:.2f}%')  # 将概率转换为百分比

        if prediction >= risk_threshold:
            st.markdown(
                "<span style='color:red'>High risk of mortality within 3 years.</span>",
                unsafe_allow_html=True)

            st.write("Personalized Recommendations:")
            for feature, value in data.items():
                if feature in normal_ranges:
                    normal_min, normal_max = normal_ranges[feature]
                    if value < normal_min:
                        st.markdown(
                            f"<span style='color:red'>{feature}: Your value is {value}. It is below the normal range ({normal_min} - {normal_max}). Consider increasing it.</span>",
                            unsafe_allow_html=True)
                    elif value > normal_max:
                        st.markdown(
                            f"<span style='color:red'>{feature}: Your value is {value}. It is above the normal range ({normal_min} - {normal_max}). Consider decreasing it.</span>",
                            unsafe_allow_html=True)
                    else:
                        st.write(f"{feature}: Your value is within the normal range ({normal_min} - {normal_max}).")
        else:
            st.markdown(
                "<span style='color:green'>Low risk of mortality within 3 years.</span>",
                unsafe_allow_html=True)
    except Exception as e:
        st.write(f'Error: {str(e)}')
