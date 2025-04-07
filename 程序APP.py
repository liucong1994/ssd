import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载Font Awesome图标库
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">',
            unsafe_allow_html=True)

# 自定义页面样式
st.markdown("""
<style>
    /* 修改结果卡片样式 */
    .result-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin: 1rem auto;
        padding: 1.5rem;
        max-width: 1200px;
    }
    .metric-value {
        font-size: 2.5rem !important;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .high-risk {
        border-color: #ff4b4b !important;
        background: #fff5f5 !important;
    }
</style>
""", unsafe_allow_html=True)


# 加载模型
@st.cache_resource
def load_model():
    model = joblib.load('rf_model.pkl')
    # 验证模型类别
    if model.classes_.tolist() != [0, 1]:
        st.error("模型类别定义异常，请确认训练标签顺序应为[0, 1]！")
        st.stop()
    return model


model = load_model()

# 特征定义
feature_ranges = {
    "Subtype": {
        "type": "categorical",
        "options": [0, 1, 2],
        "labels": ["LumA/B", "HER2+", "TNBC"],
        "display_name": "分子亚型"
    },
    "NLR": {
        "type": "numerical",
        "min": 0.0,
        "max": 10.0,
        "default": 5.0,
        "display_name": "中性粒细胞/淋巴细胞比"
    },
    "IL6": {
        "type": "numerical",
        "min": 0.0,
        "max": 100.0,
        "default": 5.0,
        "display_name": "白介素6"
    },
    "CAR": {
        "type": "numerical",
        "min": 0.0,
        "max": 5.0,
        "default": 0.2,
        "display_name": "C反应蛋白/白蛋白"
    },
    "VitD": {
        "type": "numerical",
        "min": 0.0,
        "max": 100.0,
        "default": 35.0,
        "display_name": "维生素D"
    },
    "FT4": {
        "type": "numerical",
        "min": 0.0,
        "max": 100.0,
        "default": 15.0,
        "display_name": "游离甲状腺素"
    },
}

# 页面标题
st.markdown("""
<div style="text-align: center; margin: 3rem 0 2rem; padding: 0 2rem;">
    <h1 style="color: var(--primary-color); 
              font-size: 2.0rem; 
              margin: 0 auto 1rem;
              max-width: 1200px;
              line-height: 1.2;
              padding: 1.5rem 0;
              border-bottom: 3px solid #4CAF50;
              display: inline-block;">
        <i class="fas fa-heartbeat" style="margin-right: 1rem;"></i>
        乳腺癌术后阈下抑郁风险评估系统
    </h1>
    <p style="color: #6c757d; 
             font-size: 1.1rem;
             max-width: 1200px;
             margin: 1rem auto;
             padding: 0 2rem;">
        基于机器学习与可解释性AI的临床决策支持系统
    </p>
</div>
""", unsafe_allow_html=True)

# 输入表单
with st.container():
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### 🧬 患者特征输入")

    col1, col2 = st.columns(2)
    feature_values = []

    with col1:
        # 分子亚型
        subtype = feature_ranges["Subtype"]
        val = st.selectbox(
            label=subtype["display_name"],
            options=subtype["options"],
            format_func=lambda x: subtype["labels"][x],
            help="根据免疫组化检测结果选择"
        )
        feature_values.append(val)

        # NLR
        nlr = feature_ranges["NLR"]
        val = st.slider(
            label=nlr["display_name"],
            min_value=float(nlr["min"]),
            max_value=float(nlr["max"]),
            value=float(nlr["default"]),
            help="正常范围：0.5-3.0"
        )
        feature_values.append(val)

        # IL6
        il6 = feature_ranges["IL6"]
        val = st.slider(
            label=il6["display_name"],
            min_value=float(il6["min"]),
            max_value=float(il6["max"]),
            value=float(il6["default"]),
            help="正常值：<7 pg/mL"
        )
        feature_values.append(val)

    with col2:
        # CAR
        car = feature_ranges["CAR"]
        val = st.slider(
            label=car["display_name"],
            min_value=float(car["min"]),
            max_value=float(car["max"]),
            value=float(car["default"]),
            help="正常值：<0.15"
        )
        feature_values.append(val)

        # VitD
        vitd = feature_ranges["VitD"]
        val = st.slider(
            label=vitd["display_name"],
            min_value=float(vitd["min"]),
            max_value=float(vitd["max"]),
            value=float(vitd["default"]),
            help="正常范围：30-100 ng/mL"
        )
        feature_values.append(val)

        # FT4
        ft4 = feature_ranges["FT4"]
        val = st.slider(
            label=ft4["display_name"],
            min_value=float(ft4["min"]),
            max_value=float(ft4["max"]),
            value=float(ft4["default"]),
            help="正常范围：10-31 pmol/L"
        )
        feature_values.append(val)

    st.markdown('</div>', unsafe_allow_html=True)

# 预测按钮
if st.button("🚀 开始风险评估", use_container_width=True):
    features = np.array([feature_values])

    try:
        proba_array = model.predict_proba(features)[0]
        if len(proba_array) != 2:
            raise ValueError("模型输出维度异常")

        probability_positive = proba_array[1] * 100
        probability_negative = proba_array[0] * 100
        predicted_class = 1 if probability_positive >= 50 else 0
        predicted_label = "高风险" if predicted_class == 1 else "低风险"

        # 结果展示
        st.markdown(f"""
        <div class="result-card {'high-risk' if predicted_class == 1 else ''}">
            <div style="position: relative; z-index: 3;">
                <div style="font-size: 1.4rem; margin-bottom: 1.5rem; color: {'#ff4b4b' if predicted_class == 1 else '#4CAF50'};">
                    <i class="fas fa-{'exclamation-triangle' if predicted_class == 1 else 'check-circle'}"></i>
                    风险评估结论
                </div>
                <div class="metric-value">
                    {predicted_label}
                </div>
                <div class="risk-details">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                        <div>
                            <div style="font-size: 1.1rem; color: #666; margin-bottom: 0.5rem;">
                                <i class="fas fa-arrow-up"></i>
                                阳性概率
                            </div>
                            <div style="font-size: 1.8rem; color: {'#ff4b4b' if predicted_class == 1 else '#4CAF50'}; font-weight: 700;">
                                {probability_positive:.1f}%
                            </div>
                        </div>
                        <div>
                            <div style="font-size: 1.1rem; color: #666; margin-bottom: 0.5rem;">
                                <i class="fas fa-arrow-down"></i>
                                阴性概率
                            </div>
                            <div style="font-size: 1.8rem; color: #6c757d; font-weight: 700;">
                                {probability_negative:.1f}%
                            </div>
                        </div>
                    </div>
                </div>
                <div class="risk-threshold" style="margin-top: 1.5rem;">
                    <i class="fas fa-info-circle"></i>
                    临床建议：{'建议立即启动心理干预流程' if predicted_class == 1 else '建议定期随访观察'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"预测过程出现异常：{str(e)}")
        st.stop()

    # SHAP可视化部分
    with st.spinner("生成可解释性分析..."):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        sample_data = pd.DataFrame([feature_values], columns=feature_ranges.keys())
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(sample_data)

        # 获取SHAP值
        if len(shap_values.shape) == 3:
            current_shap_values = shap_values[0, :, predicted_class].values
        else:
            current_shap_values = shap_values[0, :, 1].values

        current_shap_values = -current_shap_values  # 反转方向

        # 创建专业级可视化
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(feature_ranges))

        # 使用渐变色条
        colors = ['#ff6b6b' if val > 0 else '#4CAF50' for val in current_shap_values]
        bars = ax.barh(y_pos, current_shap_values, align='center', height=0.6, color=colors)

        # 添加数据标签
        for i, (val, name) in enumerate(
                zip(current_shap_values, [feature_ranges[f]['display_name'] for f in feature_ranges])):
            ax.text(val / 2 if val > 0 else val * 1.2, i,
                    f"{name}\n{val:.2f}",
                    va='center', ha='left' if val < 0 else 'right',
                    color='white' if abs(val) > 0.2 else '#666',
                    fontsize=10)

        # 美化坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.tick_params(axis='y', length=0)
        ax.set_xlabel('特征影响值', fontsize=12, color='#666')
        ax.set_title('特征影响分析',
                     fontsize=14, pad=20,
                     color='#2c3e50',
                     fontweight='bold')

        # 添加网格线
        ax.grid(axis='x', linestyle='--', alpha=0.4, color='#cccccc')

        # 调整布局
        plt.tight_layout(pad=2)

        st.pyplot(fig)
        plt.close()

        st.caption(f"""
        影响因素说明：
        • 红色特征：增加阈下抑郁风险的因素（SHAP值 > 0）
        • 蓝色特征：降低风险的保护性因素（SHAP值 < 0）
        """)

# 辅助信息
with st.expander("📚 临床指标参考指南", expanded=False):
    st.markdown("""
    **临床指标参考范围表**
    | 指标名称        | 正常范围       | 临床意义                  |
    |----------------|---------------|--------------------------|
    | NLR            | 0.5-3.0       | 全身炎症反应标志物        |
    | CAR            | <0.15         | 炎症/营养状态综合指标     |
    | IL-6           | <7 pg/mL      | 促炎细胞因子              |
    | 维生素D        | 30-100 ng/mL  | 免疫调节相关营养指标      |
    | FT4            | 10-31 pmol/L | 甲状腺功能核心指标        |
    """)

# 页脚
st.markdown("""
<hr style="margin: 4rem 0 2rem 0; border-top: 1px solid #e9ecef;"/>
<div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
    <p>淮北市人民医院肿瘤内科护理组</p>
    <p style="margin-top: 0.5rem;">
        <i class="fas fa-exclamation-triangle"></i> 
        *本系统预测结果仅供学术研究参考，不作为最终临床诊断依据*
    </p>
</div>
""", unsafe_allow_html=True)
