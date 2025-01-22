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

# 列出所有需要安装的包
packages = ['pip']
for package in packages:
    install(package)

# 加载模型和标准化器
model_path = "random_forest_model.pkl"
scaler_path = "scaler.pkl"

with open(model_path, 'rb') as model_file, open(scaler_path, 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

# 特征名称直接在代码中定义
feature_names = ["age", "P", "respirator", "PCI", "β-blocker", "Glu"]

# 创建Web应用的标题
st.title('Machine learning-based model predicts 3-year mortality in elderly AMI patients')

# 添加介绍部分
st.markdown("""
## Introduction
This web-based calculator was developed based on the Random Forest model with an AUC of 0.92 (95% CI: 0.845 to 0.968) and the following additional metrics: Accuracy of 0.803, F1-score of 0.829, Brier score of 0.147, and AUC-PR of 0.93. Users can obtain the 3-year risk of death for elderly AMI patients by selecting the parameters and clicking on the "Predict" button.
""")

# 创建输入表单
st.markdown("## Selection Panel")
st.markdown("Picking up parameters")

with st.form("prediction_form"):
    age = st.slider('Age', min_value=75, max_value=100, value=75)  # 设置年龄最小值为75岁
    p = st.slider('P (bpm)', min_value=30, max_value=150, value=75)  # P 列代表心率，调整范围
    glu = st.slider('Glu (mmol/L)', min_value=3.0, max_value=15.0, value=5.5)  # 血糖正常值
    respirator = st.selectbox('Respirator', options=['No', 'Yes'])  # 分类变量用Yes/No
    pci = st.selectbox('PCI', options=['No', 'Yes'])  # 分类变量用Yes/No
    beta_blocker = st.selectbox('β-blocker', options=['No', 'Yes'])  # 分类变量用Yes/No

    # 提交按钮
    submit_button = st.form_submit_button("Predict")

# 定义正常值范围
normal_ranges = {
    "P (bpm)": (60, 100),  # 心率正常范围
    "Glu": (4.0, 7.0)  # 血糖正常范围
}

# 当用户提交表单时
if submit_button:
    # 构建请求数据，将 'P (bpm)' 对应为实际训练时的 'P'
    data = {
        "age": age,
        "P": p,  # 数据文件中的列名是 'P'
        "Glu": glu,  # 数据文件中的列名是 'Glu'
        "respirator": 1 if respirator == 'Yes' else 0,  # 转换成数值
        "PCI": 1 if pci == 'Yes' else 0,  # 转换成数值
        "β-blocker": 1 if beta_blocker == 'Yes' else 0  # 转换成数值
    }

    try:
        # 将数据转换为 DataFrame，并按特征文件中的顺序排列列
        data_df = pd.DataFrame([data], columns=feature_names)

        # 应用标准化
        data_scaled = scaler.transform(data_df)

        # 进行预测
        prediction = model.predict_proba(data_scaled)[:, 1][0]  # 获取类别为1的预测概率

        # 显示预测结果
        st.write(f'Prediction: {prediction * 100:.2f}%')  # 将概率转换为百分比

        # 提供个性化建议
        if prediction >= 0.482:
            st.markdown(
                "<span style='color:red'>This patient has a high probability of death within three years.</span>",
                unsafe_allow_html=True)
            st.write("Personalized Recommendations:")
            # 提供每个特征的调整建议
            for feature, (normal_min, normal_max) in normal_ranges.items():
                value = data[feature] if feature != "P (bpm)" else p  # 特别处理显示 'P (bpm)'
                if value < normal_min:
                    st.markdown(
                        f"<span style='color:red'>{feature}: Your value is {value}. It is lower than the normal range ({normal_min} - {normal_max}). Consider increasing it towards {normal_min}.</span>",
                        unsafe_allow_html=True)
                elif value > normal_max:
                    st.markdown(
                        f"<span style='color:red'>{feature}: Your value is {value}. It is higher than the normal range ({normal_min} - {normal_max}). Consider decreasing it towards {normal_max}.</span>",
                        unsafe_allow_html=True)
                else:
                    st.write(f"{feature}: Your value is within the normal range ({normal_min} - {normal_max}).")

            # 药物治疗建议
            if beta_blocker == 'No':
                st.write("Consider using β-blocker medication.")
            if pci == 'No':
                st.write("Consider undergoing PCI therapy.")
            if respirator == 'No':
                st.write("Respiratory support might be necessary in some cases.")
        else:
            st.markdown(
                "<span style='color:green'>This patient has a high probability of survival after three years.</span>",
                unsafe_allow_html=True)
    except Exception as e:
        st.write(f'Error: {str(e)}')