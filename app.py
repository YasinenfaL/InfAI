"""
DataAnalyzerApp: A Python application for data analysis, correlation analysis,
and conversational interaction using Streamlit and the LangChain library.
"""

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
import time
from matplotlib.colors import LinearSegmentedColormap
import streamlit.components.v1 as components

st.set_page_config(page_title="InfAI", layout="wide", initial_sidebar_state="collapsed")

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "show_api_input" not in st.session_state:
    st.session_state.show_api_input = False
if "page_loaded" not in st.session_state:
    st.session_state.page_loaded = False
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0
if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = False

def add_bg_gradient():
    if st.session_state.theme == "dark":
        bg_style = """
        <style>
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #f0f0f0;
        }
        </style>
        """
    else:
        bg_style = """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #eef2f5 100%);
        }
        </style>
        """
    return bg_style

def load_css():
    theme_color = "#4527A0" if st.session_state.theme == "light" else "#7C4DFF"
    bg_color = "white" if st.session_state.theme == "light" else "#1E1E2E"
    text_color = "#333" if st.session_state.theme == "light" else "#f0f0f0"
    secondary_color = "#673AB7" if st.session_state.theme == "light" else "#BB86FC"
    
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');
        
        * {{
            font-family: 'Poppins', sans-serif;
            transition: all 0.3s ease;
        }}
        
        .main-header {{
            font-size: 3.5rem !important;
            font-weight: 700;
            background: linear-gradient(90deg, {theme_color}, {secondary_color}, #3F51B5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.8rem;
            animation: fadeIn 1.5s ease-in-out, pulse 5s ease-in-out infinite;
        }}
        
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.03); }}
            100% {{ transform: scale(1); }}
        }}
        
        .subtitle {{
            text-align: center;
            font-size: 1.3rem;
            color: {text_color};
            margin-bottom: 2rem;
            animation: fadeIn 2s ease-in-out;
            opacity: 0.8;
        }}
        
        .subtitle span {{
            color: {secondary_color};
            font-weight: 600;
        }}
        
        @keyframes fadeIn {{
            0% {{ opacity: 0; transform: translateY(20px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @keyframes slideIn {{
            0% {{ opacity: 0; transform: translateX(-20px); }}
            100% {{ opacity: 1; transform: translateX(0); }}
        }}
        
        @keyframes slideUp {{
            0% {{ opacity: 0; transform: translateY(20px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .card {{
            border-radius: 16px;
            background-color: {bg_color};
            padding: 28px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.09);
            margin-bottom: 28px;
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            animation: fadeIn 0.8s ease-in-out;
            border: 1px solid rgba(200, 200, 200, 0.1);
        }}
        
        .card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 16px 32px rgba(0,0,0,0.15);
        }}
        
        .interactive-card {{
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }}
        
        .interactive-card::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, rgba(255,255,255,0) 70%, rgba(255,255,255,0.1) 100%);
            transition: all 0.3s ease;
        }}
        
        .interactive-card:hover::after {{
            transform: translateX(10px) translateY(-10px);
        }}
        
        .subheader {{
            font-size: 1.8rem;
            font-weight: 600;
            color: {theme_color};
            margin-bottom: 1.5rem;
            border-bottom: 2px solid rgba(240, 240, 240, 0.1);
            padding-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .subheader i {{
            font-size: 1.5rem;
            opacity: 0.9;
        }}
        
        h3 {{
            font-size: 1.2rem;
            font-weight: 600;
            color: {text_color};
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .success-msg {{
            color: #00C851;
            font-weight: 600;
            background-color: rgba(0, 200, 81, 0.1);
            padding: 14px 24px;
            border-radius: 12px;
            border-left: 4px solid #00C851;
            animation: slideIn 0.5s ease-in-out;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .info-msg {{
            background-color: rgba(33, 150, 243, 0.1);
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #2196F3;
            color: {text_color};
            animation: fadeIn 1s ease-in-out;
        }}
        
        .dataframe {{
            border-collapse: collapse !important;
            border: none !important;
            font-size: 0.9rem;
            background-color: {bg_color};
            color: {text_color};
        }}
        
        .dataframe th {{
            background-color: {theme_color} !important;
            color: white !important;
            font-weight: 600 !important;
            text-align: left !important;
            padding: 12px !important;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }}
        
        .dataframe td {{
            padding: 12px !important;
            border-bottom: 1px solid rgba(240, 240, 240, 0.1) !important;
        }}
        
        div.stButton > button {{
            background-color: {theme_color};
            color: white;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            border: none;
            font-weight: 500;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.9rem;
        }}
        
        div.stButton > button:hover {{
            background-color: {secondary_color};
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
            transform: translateY(-2px);
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 10px;
            border-radius: 10px;
            background-color: rgba(240, 240, 240, 0.1);
            padding: 5px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            white-space: pre-wrap;
            background-color: rgba(240, 240, 240, 0.05);
            border-radius: 8px;
            gap: 1px;
            padding: 10px 16px;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {theme_color} !important;
            color: white !important;
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        
        div[data-testid="stFileUploader"] {{
            border: 2px dashed rgba(150, 150, 150, 0.3);
            padding: 40px 30px;
            border-radius: 12px;
            text-align: center;
            background: linear-gradient(135deg, rgba(240, 240, 240, 0.05) 0%, rgba(240, 240, 240, 0.1) 100%);
            transition: all 0.3s ease;
        }}
        
        div[data-testid="stFileUploader"]:hover {{
            border-color: {secondary_color};
            background: linear-gradient(135deg, rgba(240, 240, 240, 0.1) 0%, rgba(240, 240, 240, 0.15) 100%);
        }}
        
        div[data-testid="metric-container"] {{
            background-color: rgba(240, 240, 240, 0.05);
            border-radius: 12px;
            padding: 20px 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.08);
            border: 1px solid rgba(200, 200, 200, 0.1);
            transition: all 0.3s ease;
        }}
        
        div[data-testid="metric-container"]:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.12);
        }}
        
        div[data-testid="metric-container"] label {{
            color: {text_color};
            opacity: 0.8;
            font-weight: 500;
        }}
        
        div[data-testid="metric-container"] .css-1wivap2 {{
            color: {secondary_color};
            font-weight: 700;
        }}
        
        div[data-testid="stMetricValue"] > div {{
            font-size: 2.2rem;
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .tooltip {{
            position: relative;
            display: inline-block;
            cursor: pointer;
        }}
        
        .tooltip .tooltiptext {{
            visibility: hidden;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 10px 15px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -80px;
            opacity: 0;
            transition: opacity 0.3s;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            font-size: 0.9rem;
            line-height: 1.4;
            width: 180px;
        }}
        
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
            animation: fadeIn 0.3s ease-in-out;
        }}

        .api-input-container {{
            background-color: {bg_color};
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border-left: 4px solid {theme_color};
            animation: slideIn 0.5s ease-in-out;
            border: 1px solid rgba(200, 200, 200, 0.1);
        }}
        
        .settings-icon {{
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: {bg_color};
            color: {theme_color};
            width: 48px;
            height: 48px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            z-index: 1000;
            font-size: 22px;
            transition: all 0.4s ease;
            border: 1px solid rgba(200, 200, 200, 0.1);
        }}
        
        .settings-icon:hover {{
            transform: rotate(45deg);
            box-shadow: 0 6px 15px rgba(0,0,0,0.25);
            background-color: {theme_color};
            color: white;
        }}
        
        .theme-switch {{
            position: fixed;
            top: 20px;
            left: 20px;
            background-color: {bg_color};
            color: {theme_color};
            width: 48px;
            height: 48px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            z-index: 1000;
            font-size: 22px;
            transition: all 0.4s ease;
            border: 1px solid rgba(200, 200, 200, 0.1);
        }}
        
        .theme-switch:hover {{
            transform: rotate(-45deg);
            box-shadow: 0 6px 15px rgba(0,0,0,0.25);
            background-color: {theme_color};
            color: white;
        }}
        
        .progress-container {{
            width: 100%;
            height: 8px;
            background-color: rgba(240, 240, 240, 0.1);
            border-radius: 4px;
            margin: 10px 0;
            overflow: hidden;
            position: relative;
        }}
        
        .progress-bar {{
            position: absolute;
            height: 100%;
            background: linear-gradient(90deg, {theme_color}, {secondary_color});
            border-radius: 4px;
            animation: progress-anim 2.5s ease-in-out infinite;
            background-size: 200% 100%;
        }}
        
        @keyframes progress-anim {{
            0% {{ background-position: 0% 0%; }}
            50% {{ background-position: 100% 0%; }}
            100% {{ background-position: 0% 0%; }}
        }}
        
        .download-button {{
            display: inline-block;
            padding: 12px 24px;
            background-color: {theme_color};
            color: white;
            text-decoration: none;
            border-radius: 10px;
            transition: all 0.3s ease;
            font-weight: 500;
            margin: 1rem 0;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }}
        
        .download-button:hover {{
            background-color: {secondary_color};
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
            transform: translateY(-3px);
        }}
        
        .download-button i {{
            transition: all 0.3s ease;
        }}
        
        .download-button:hover i {{
            transform: translateY(-3px);
        }}
        
        .welcome-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.85);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: fadeOut 0.5s ease-in-out 3s forwards;
        }}
        
        .welcome-content {{
            text-align: center;
            color: white;
            animation: welcomePulse 2s ease-in-out;
        }}
        
        .welcome-content h1 {{
            font-size: 4rem;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, {theme_color}, {secondary_color}, #3F51B5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .welcome-content p {{
            font-size: 1.5rem;
            margin-bottom: 2rem;
            opacity: 0.8;
        }}
        
        @keyframes welcomePulse {{
            0% {{ transform: scale(0.8); opacity: 0; }}
            50% {{ transform: scale(1.05); opacity: 1; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
        
        @keyframes fadeOut {{
            0% {{ opacity: 1; visibility: visible; }}
            100% {{ opacity: 0; visibility: hidden; }}
        }}
        
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: rgba(240, 240, 240, 0.1);
            border-radius: 10px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {theme_color};
            border-radius: 10px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {secondary_color};
        }}
        
        .file-uploader-content {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
        }}
        
        .file-uploader-icon {{
            font-size: 3rem;
            color: {theme_color};
            margin-bottom: 10px;
            animation: float 3s ease-in-out infinite;
        }}
        
        @keyframes float {{
            0% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-10px); }}
            100% {{ transform: translateY(0px); }}
        }}
        
        .upload-text {{
            font-size: 1.3rem;
            font-weight: 600;
            color: {text_color};
            margin-bottom: 5px;
        }}
        
        .upload-subtext {{
            font-size: 0.9rem;
            color: {text_color};
            opacity: 0.7;
        }}
        
        .tab-indicators {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 1rem 0;
        }}
        
        .tab-indicator {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: rgba(240, 240, 240, 0.3);
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .tab-indicator.active {{
            background-color: {theme_color};
            transform: scale(1.2);
        }}
    </style>
    """

def get_logo_html():
    theme_color = "#4527A0" if st.session_state.theme == "light" else "#7C4DFF"
    
    return f"""
    <div style="display: flex; justify-content: center; margin-bottom: 1rem; animation: fadeIn 1s ease-in-out;">
        <div style="background: linear-gradient(45deg, {theme_color}, #673AB7); width: 90px; height: 90px; 
                    border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                    box-shadow: 0 6px 12px rgba(0,0,0,0.2);">
            <span style="color: white; font-size: 2.5rem; font-weight: bold; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">AI</span>
        </div>
    </div>
    """

def get_file_uploader_html():
    return """
    <div class="file-uploader-content">
        <div class="file-uploader-icon">
            <i class="fas fa-file-csv"></i>
        </div>
        <div class="upload-text">CSV Dosyası Yükleyin</div>
        <div class="upload-subtext">veya dosyayı bu alana sürükleyip bırakın</div>
    </div>
    """

def show_welcome_animation():
    if not st.session_state.welcome_shown:
        welcome_html = """
        <div class="welcome-container">
            <div class="welcome-content">
                <h1>InfAI</h1>
                <p>Yapay Zeka Destekli Veri Analiz Platformu</p>
                <div class="progress-container">
                    <div class="progress-bar" style="width: 100%;"></div>
                </div>
            </div>
        </div>
        """
        st.markdown(welcome_html, unsafe_allow_html=True)
        st.session_state.welcome_shown = True

def show_theme_switch():
    if st.session_state.theme == "light":
        icon = "🌙"
    else:
        icon = "☀️"
        
    theme_html = f"""
    <div class="theme-switch" onclick="document.getElementById('theme-button').click()">
        {icon}
    </div>
    """
    st.markdown(theme_html, unsafe_allow_html=True)
    
    theme_clicked = st.button("Theme", key="theme-button", help="Tema Değiştir", type="primary")
    if theme_clicked:
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()

st.markdown(add_bg_gradient(), unsafe_allow_html=True)
st.markdown(load_css(), unsafe_allow_html=True)

show_welcome_animation()

show_theme_switch()

if not st.session_state.page_loaded:
    progress_bar = st.progress(0)
    for i in range(101):
        progress_bar.progress(i)
        time.sleep(0.01)
    st.session_state.page_loaded = True
    st.rerun()

st.markdown(
    """
    <div class="settings-icon" onclick="document.getElementById('settings-button').click()">
        <i class="fas fa-cog"></i>
    </div>
    """, 
    unsafe_allow_html=True
)

settings_clicked = st.button("Ayarlar", key="settings-button", help="API Ayarları", type="primary")
if settings_clicked:
    st.session_state.show_api_input = not st.session_state.show_api_input

if st.session_state.show_api_input:
    st.markdown("<div class='api-input-container'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0;'><i class='fas fa-key'></i> OpenAI API Ayarları</h3>", unsafe_allow_html=True)
    
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_api_key,
        placeholder="API anahtarınızı buraya girin...",
        help="OpenAI API anahtarınızı girin. Bu anahtar güvenli bir şekilde saklanacak ve yalnızca bu oturum için kullanılacaktır."
    )
    
    if api_key:
        if api_key != st.session_state.openai_api_key:
            st.session_state.openai_api_key = api_key
            st.markdown("<div class='success-msg'><i class='fas fa-check-circle'></i> API Key başarıyla kaydedildi!</div>", unsafe_allow_html=True)
            
    st.info("API anahtarınız, yapay zeka destekli analiz ve öngörüler için kullanılır. Anahtar olmadan yalnızca temel veri görselleştirme özellikleri kullanılabilir.")
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(get_logo_html(), unsafe_allow_html=True)
st.markdown("<h1 class='main-header'>InfAI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Yapay Zeka Destekli <span>Veri Analiz</span> Platformu</p>", unsafe_allow_html=True)

st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.15;'>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(get_file_uploader_html(), unsafe_allow_html=True)
        uploaded_file = st.file_uploader("CSV Dosyası Seçin", type="csv", label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"<div class='success-msg'>✅ '{uploaded_file.name}' başarıyla yüklendi!</div>", unsafe_allow_html=True)
        
        save_path = "data"
        os.makedirs(save_path, exist_ok=True)
        file_name = uploaded_file.name
        save_file = os.path.join(save_path, file_name)
        df.to_csv(save_file, index=False)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        total_rows = df.shape[0]
        total_cols = df.shape[1]
        memory_usage = df.memory_usage(deep=True).sum()
        
        def format_bytes(size):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.2f} {unit}"
                size /= 1024.0
            return f"{size:.2f} TB"
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Satır Sayısı", f"{total_rows:,}")
        col2.metric("Sütun Sayısı", total_cols)
        col3.metric("Bellek Kullanımı", format_bytes(memory_usage))
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h2 class='subheader'>📊 Veri Özeti</h2>", unsafe_allow_html=True)
            
            st.markdown("<h3>Veri Tipleri</h3>", unsafe_allow_html=True)
            dtypes_df = pd.DataFrame({
                "Sütun": df.dtypes.index,
                "Veri Tipi": [str(dtype) for dtype in df.dtypes.values]
            })
            st.dataframe(dtypes_df, use_container_width=True, height=min(250, 35 + 35 * len(dtypes_df)))
            
            st.markdown("<h3>Eksik Değerler</h3>", unsafe_allow_html=True)
            missing_data = pd.DataFrame({
                "Sütun": df.columns,
                "Eksik Değer": df.isnull().sum().values,
                "Yüzde": (df.isnull().sum().values / len(df) * 100).round(2)
            })
            missing_data["Yüzde"] = missing_data["Yüzde"].apply(lambda x: f"{x}%")
            st.dataframe(missing_data, use_container_width=True, height=min(250, 35 + 35 * len(missing_data)))
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h2 class='subheader'>👁️ Veri Önizleme</h2>", unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            
            # İstatistikler
            st.markdown("<h3>İstatistiksel Özet</h3>", unsafe_allow_html=True)
            st.dataframe(df.describe().round(2), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Görselleştirme bölümü
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='subheader'>📈 Veri Görselleştirme</h2>", unsafe_allow_html=True)
        
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        tabs = st.tabs(["📊 Dağılım", "🔄 Korelasyon", "📋 Kategorik Analiz"])
        
        with tabs[0]:
            if numeric_columns:
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    selected_column = st.selectbox("Sütun seçin:", numeric_columns)
                    color = st.color_picker("Renk seçin:", "#4527A0")
                    plot_type = st.radio("Grafik türü:", ["Histogram", "Box Plot", "Violin Plot"])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if plot_type == "Histogram":
                        sns.histplot(df[selected_column], kde=True, ax=ax, color=color)
                        ax.set_title(f"{selected_column} Dağılımı", fontsize=16)
                        ax.set_ylabel("Frekans", fontsize=12)
                    elif plot_type == "Box Plot":
                        sns.boxplot(x=df[selected_column], ax=ax, color=color)
                        ax.set_title(f"{selected_column} Box Plot", fontsize=16)
                    else:  
                        sns.violinplot(x=df[selected_column], ax=ax, color=color)
                        ax.set_title(f"{selected_column} Violin Plot", fontsize=16)
                    
                    ax.set_xlabel(selected_column, fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # İstatistik özeti
                    st.markdown(f"<h3>{selected_column} İstatistikleri</h3>", unsafe_allow_html=True)
                    col_stats = pd.DataFrame({
                        "Metrik": ["Ortalama", "Medyan", "Standart Sapma", "Min", "Max", "Count"],
                        "Değer": [
                            round(df[selected_column].mean(), 2),
                            round(df[selected_column].median(), 2),
                            round(df[selected_column].std(), 2),
                            round(df[selected_column].min(), 2),
                            round(df[selected_column].max(), 2),
                            df[selected_column].count()
                        ]
                    })
                    st.dataframe(col_stats, use_container_width=True, hide_index=True)
            else:
                st.warning("Dağılım görselleştirmesi için sayısal sütun bulunamadı.")
        
        with tabs[1]:
            if len(numeric_columns) > 1:
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    corr_columns = st.multiselect(
                        "Sütunları seçin (maks. 10):",
                        options=numeric_columns,
                        default=numeric_columns[:min(5, len(numeric_columns))]
                    )
                    
                    corr_method = st.selectbox(
                        "Korelasyon metodu:",
                        options=["pearson", "spearman", "kendall"],
                        index=0
                    )
                    
                    cmap_option = st.selectbox(
                        "Renk paleti:",
                        options=["coolwarm", "viridis", "plasma", "inferno", "magma", "cividis"],
                        index=0
                    )
                
                with col1:
                    if corr_columns and len(corr_columns) > 1:
                        if len(corr_columns) > 10:
                            st.warning("En iyi görselleştirme için maksimum 10 sütun seçin.")
                            corr_columns = corr_columns[:10]
                            
                        corr_matrix = df[corr_columns].corr(method=corr_method)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                        cmap = sns.color_palette(cmap_option, as_cmap=True)
                        
                        sns.heatmap(
                            corr_matrix, 
                            mask=mask, 
                            cmap=cmap, 
                            annot=True, 
                            center=0, 
                            square=True, 
                            linewidths=.5, 
                            fmt=".2f"
                        )
                        
                        plt.title(f"Korelasyon Matrisi ({corr_method.capitalize()})", fontsize=16)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Yüksek korelasyonu olan sütunlar
                        high_corr = corr_matrix.unstack().sort_values(ascending=False)
                        high_corr = high_corr[(high_corr < 1.0) & (abs(high_corr) >= 0.5)]
                        
                        if not high_corr.empty:
                            st.markdown("<h3>Dikkat Çeken Korelasyonlar (|r| ≥ 0.5)</h3>", unsafe_allow_html=True)
                            
                            high_corr_data = []
                            for (col1, col2), value in high_corr.items():
                                high_corr_data.append({
                                    "Sütun 1": col1,
                                    "Sütun 2": col2,
                                    "Korelasyon": round(value, 3),
                                    "İlişki": "Pozitif" if value > 0 else "Negatif"
                                })
                            
                            high_corr_df = pd.DataFrame(high_corr_data)
                            st.dataframe(high_corr_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("Korelasyon analizi için en az 2 sütun seçilmelidir.")
            else:
                st.warning("Korelasyon analizi için yeterli sayısal sütun bulunamadı.")
        
        with tabs[2]:
            if categorical_columns:
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    selected_cat_column = st.selectbox(
                        "Kategorik sütun seçin:", 
                        categorical_columns,
                        key="cat_column"
                    )
                    
                    count_limit = st.slider(
                        "Gösterilecek kategori sayısı:", 
                        min_value=3, 
                        max_value=min(20, df[selected_cat_column].nunique()),
                        value=min(10, df[selected_cat_column].nunique())
                    )
                    
                    plot_type = st.radio(
                        "Grafik türü:", 
                        ["Bar", "Pie"],
                        key="cat_plot_type"
                    )
                
                with col1:
                    value_counts = df[selected_cat_column].value_counts().nlargest(count_limit)
                    
                    if plot_type == "Bar":
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis", ax=ax)
                        ax.set_title(f"{selected_cat_column} Değer Dağılımı (Top {count_limit})", fontsize=16)
                        ax.set_xlabel("Frekans", fontsize=12)
                        ax.set_ylabel(selected_cat_column, fontsize=12)
                    else:  # Pie
                        colors = plt.cm.viridis(np.linspace(0, 1, len(value_counts)))
                        fig, ax = plt.subplots(figsize=(10, 8))
                        wedges, texts, autotexts = ax.pie(
                            value_counts.values, 
                            labels=value_counts.index, 
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=colors,
                            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
                        )
                        # Yazıları özelleştirme
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontsize(10)
                            autotext.set_fontweight('bold')
                        ax.set_title(f"{selected_cat_column} Dağılımı (Top {count_limit})", fontsize=16)
                        ax.axis('equal')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Kategori istatistikleri
                    st.markdown(f"<h3>{selected_cat_column} İstatistikleri</h3>", unsafe_allow_html=True)
                    cat_stats = pd.DataFrame({
                        "Kategori": value_counts.index,
                        "Frekans": value_counts.values,
                        "Yüzde": [(v/sum(value_counts.values)*100) for v in value_counts.values]
                    })
                    cat_stats["Yüzde"] = cat_stats["Yüzde"].apply(lambda x: f"{x:.2f}%")
                    st.dataframe(cat_stats, use_container_width=True, hide_index=True)
            else:
                st.warning("Kategorik analiz için kategorik sütun bulunamadı.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='subheader'>📥 Veri İndirme</h2>", unsafe_allow_html=True)
        
        # İşlenmiş veriyi indirme
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV olarak indirme
            csv = df.to_csv(index=False)
            b64_csv = base64.b64encode(csv.encode()).decode()
            href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="{os.path.splitext(file_name)[0]}_processed.csv" class="download-button">CSV İndir</a>'
            st.markdown(f"<div style='text-align: center; margin: 1rem 0;'>{href_csv}</div>", unsafe_allow_html=True)
        
        with col2:
            # Excel olarak indirme
            try:
                # Excel dosyası oluştur
                excel_buffer = pd.ExcelWriter(f"{os.path.splitext(file_name)[0]}_processed.xlsx", engine='xlsxwriter')
                df.to_excel(excel_buffer, index=False, sheet_name='Data')
                excel_buffer.save()
                
                with open(f"{os.path.splitext(file_name)[0]}_processed.xlsx", "rb") as f:
                    excel_data = f.read()
                
                b64_excel = base64.b64encode(excel_data).decode()
                href_excel = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="{os.path.splitext(file_name)[0]}_processed.xlsx" class="download-button">Excel İndir</a>'
                st.markdown(f"<div style='text-align: center; margin: 1rem 0;'>{href_excel}</div>", unsafe_allow_html=True)
                
                os.remove(f"{os.path.splitext(file_name)[0]}_processed.xlsx")
            except:
                st.warning("Excel indirme özelliği için xlsxwriter kütüphanesi gereklidir.")
        
        with col3:
            json_str = df.to_json(orient='records')
            b64_json = base64.b64encode(json_str.encode()).decode()
            href_json = f'<a href="data:file/json;base64,{b64_json}" download="{os.path.splitext(file_name)[0]}_processed.json" class="download-button">JSON İndir</a>'
            st.markdown(f"<div style='text-align: center; margin: 1rem 0;'>{href_json}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Hata oluştu: {str(e)}")
else:
    st.markdown("""
    <div class='info-msg'>
        <h3 style='margin-top: 0;'>Veri Analizi Başlatmak İçin:</h3>
        <ol style='margin-bottom: 0;'>
            <li>Yukarıdaki alandan CSV dosyanızı seçin.</li>
            <li>Dosyanız yüklendikten sonra otomatik analiz ve görselleştirmeler görüntülenecektir.</li>
            <li>Veri içerisinde farklı sütunları incelemek için seçenekleri kullanabilirsiniz.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .download-button {
        display: inline-block;
        padding: 10px 20px;
        background-color: #4527A0;
        color: white;
        text-decoration: none;
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .download-button:hover {
        background-color: #673AB7;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
