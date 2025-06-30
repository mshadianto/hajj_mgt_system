# =============================================================================
# 🔧 SOLUSI LENGKAP MASALAH ENCODING STREAMLIT
# =============================================================================

"""
MASALAH: Script compilation error - UnicodeDecodeError: 'utf-8' codec can't decode byte 0xed

PENYEBAB: File Python (.py) tidak disimpan dengan encoding UTF-8 yang benar

SOLUSI: 
1. Pastikan semua file .py disimpan dengan UTF-8
2. Hapus karakter non-ASCII yang bermasalah
3. Gunakan tool untuk konversi encoding
4. Setup editor dengan benar
"""

# =============================================================================
# STEP 1: TOOL UNTUK MEMPERBAIKI ENCODING FILE
# =============================================================================

import os
import chardet
import codecs
from pathlib import Path

def fix_file_encoding(file_path):
    """
    Memperbaiki encoding file Python
    """
    try:
        # Baca file dengan deteksi encoding otomatis
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        # Deteksi encoding
        detected = chardet.detect(raw_data)
        original_encoding = detected['encoding']
        
        print(f"File: {file_path}")
        print(f"Detected encoding: {original_encoding}")
        print(f"Confidence: {detected['confidence']:.2%}")
        
        # Decode dengan encoding yang terdeteksi
        try:
            content = raw_data.decode(original_encoding)
        except:
            # Fallback ke latin-1 jika gagal
            content = raw_data.decode('latin-1', errors='ignore')
        
        # Simpan ulang dengan UTF-8
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Fixed: {file_path} converted to UTF-8")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {str(e)}")
        return False

def fix_all_python_files(directory="."):
    """
    Perbaiki semua file Python dalam direktori
    """
    python_files = list(Path(directory).rglob("*.py"))
    
    print(f"🔍 Found {len(python_files)} Python files")
    
    for py_file in python_files:
        fix_file_encoding(str(py_file))

# =============================================================================
# STEP 2: CLEAN MAIN.PY - DIJAMIN UTF-8 BERSIH
# =============================================================================

# main.py - BERSIH DARI KARAKTER BERMASALAH
CLEAN_MAIN_PY = '''# -*- coding: utf-8 -*-
"""
Hajj Financial Sustainability Application
Author: Hajj Finance Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Konfigurasi halaman - HARUS DI AWAL
st.set_page_config(
    page_title="Hajj Financial Sustainability",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add path untuk import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #3CB371);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2E8B57;
    }
    .stButton > button {
        background-color: #2E8B57;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    """Buat data sample untuk testing"""
    years = list(range(2020, 2026))
    packages = ['Regular', 'Plus', 'VIP']
    
    data = []
    base_costs = {'Regular': 35000000, 'Plus': 45000000, 'VIP': 60000000}
    
    for year in years:
        for package in packages:
            cost = base_costs[package] * (1.05 ** (year - 2020))  # 5% inflasi
            data.append({
                'Year': year,
                'Package': package,
                'Cost_IDR': int(cost),
                'Cost_USD': int(cost / 15600)
            })
    
    return pd.DataFrame(data)

def safe_file_upload():
    """Upload file dengan error handling"""
    uploaded_file = st.file_uploader(
        "Upload Data Keuangan (CSV/Excel)", 
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Coba baca CSV dengan berbagai encoding
                encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        st.success(f"File berhasil dibaca dengan encoding: {encoding}")
                        return df
                    except UnicodeDecodeError:
                        continue
                
                st.error("Gagal membaca file CSV dengan semua encoding")
                return None
                
            else:
                # Excel files
                df = pd.read_excel(uploaded_file)
                st.success("File Excel berhasil dibaca!")
                return df
                
        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")
            return None
    
    return None

def main():
    """Fungsi utama aplikasi"""
    
    # Header aplikasi
    st.markdown("""
    <div class="main-header">
        <h1>🕌 Hajj Financial Sustainability Application</h1>
        <p>Sistem Cerdas Perencanaan & Optimalisasi Keuangan Haji</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### Navigation")
        
        # Logo placeholder
        st.markdown("🕌 **Hajj Finance**")
        st.markdown("---")
        
        page = st.selectbox(
            "Pilih Halaman:",
            [
                "🏠 Dashboard",
                "📊 Analytics", 
                "🎯 Optimization",
                "🧠 AI Assistant",
                "⚡ Simulation",
                "🌱 Sustainability",
                "💼 Planning"
            ]
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        st.metric("Target Haji", "Rp 42.5M")
        st.metric("Progress", "12%")
    
    # Main content area
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "📊 Analytics":
        st.info("📊 Analytics page - Under development")
        st.markdown("Fitur analisis mendalam akan tersedia segera!")
    elif page == "🎯 Optimization":
        st.info("🎯 Optimization page - Under development")
        st.markdown("AI optimization untuk strategi investasi!")
    elif page == "🧠 AI Assistant":
        st.info("🧠 AI Assistant - Under development")
        st.markdown("Chatbot cerdas untuk konsultasi haji!")
    elif page == "⚡ Simulation":
        st.info("⚡ Simulation page - Under development")
        st.markdown("Monte Carlo simulation untuk proyeksi keuangan!")
    elif page == "🌱 Sustainability":
        st.info("🌱 Sustainability page - Under development")
        st.markdown("Metrik ESG dan investasi berkelanjutan!")
    elif page == "💼 Planning":
        st.info("💼 Planning page - Under development")
        st.markdown("Perencanaan keuangan personal yang komprehensif!")

def dashboard_page():
    """Halaman dashboard utama"""
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Target Dana Haji",
            value="Rp 42.5M",
            delta="Regular Package"
        )
    
    with col2:
        st.metric(
            label="Dana Terkumpul", 
            value="Rp 5.2M",
            delta="12.2%"
        )
    
    with col3:
        st.metric(
            label="Sisa Waktu",
            value="4.2 Tahun",
            delta="-6 bulan"
        )
    
    with col4:
        st.metric(
            label="Target Bulanan",
            value="Rp 740K",
            delta="Optimal"
        )
    
    st.markdown("---")
    
    # Main content dalam 2 kolom
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Trend Biaya Haji")
        
        # Load sample data
        df = create_sample_data()
        
        # Tampilkan data
        st.dataframe(df, use_container_width=True)
        
        # Chart
        fig = px.line(
            df, 
            x='Year', 
            y='Cost_IDR', 
            color='Package',
            title="Proyeksi Biaya Haji (IDR)",
            labels={'Cost_IDR': 'Biaya (IDR)', 'Year': 'Tahun'}
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # File upload section
        st.markdown("### 📁 Upload Data Keuangan")
        uploaded_df = safe_file_upload()
        
        if uploaded_df is not None:
            st.markdown("#### Preview Data Uploaded:")
            st.dataframe(uploaded_df.head(), use_container_width=True)
    
    with col2:
        st.markdown("### 💰 Kalkulator Cepat")
        
        # Input form
        with st.form("calculator"):
            target_amount = st.number_input(
                "Target Dana Haji (IDR):", 
                min_value=1000000, 
                value=42500000, 
                step=1000000
            )
            
            current_savings = st.number_input(
                "Tabungan Saat Ini (IDR):", 
                min_value=0, 
                value=5000000, 
                step=500000
            )
            
            monthly_saving = st.number_input(
                "Tabungan per Bulan (IDR):", 
                min_value=100000, 
                value=1000000, 
                step=100000
            )
            
            annual_return = st.slider(
                "Expected Annual Return (%):",
                min_value=0.0,
                max_value=15.0,
                value=6.0,
                step=0.5
            ) / 100
            
            calculate = st.form_submit_button("🧮 Hitung", use_container_width=True)
        
        if calculate:
            # Perhitungan
            remaining = target_amount - current_savings
            
            if monthly_saving > 0:
                # Dengan compound interest
                monthly_rate = annual_return / 12
                if monthly_rate > 0:
                    months_needed = np.log(1 + (remaining * monthly_rate) / monthly_saving) / np.log(1 + monthly_rate)
                else:
                    months_needed = remaining / monthly_saving
            else:
                months_needed = float('inf')
            
            years_needed = months_needed / 12
            target_date = datetime.now() + timedelta(days=months_needed * 30)
            
            # Future value calculation
            if monthly_rate > 0:
                fv = current_savings * (1 + monthly_rate) ** months_needed + monthly_saving * (((1 + monthly_rate) ** months_needed - 1) / monthly_rate)
            else:
                fv = current_savings + (monthly_saving * months_needed)
            
            # Display results
            st.markdown(f"""
            <div class="metric-card">
                <h4>📈 Hasil Kalkulasi:</h4>
                <p><strong>Sisa yang dibutuhkan:</strong><br>
                Rp {remaining:,.0f}</p>
                <p><strong>Waktu yang dibutuhkan:</strong><br>
                {years_needed:.1f} tahun ({months_needed:.0f} bulan)</p>
                <p><strong>Target tercapai:</strong><br>
                {target_date.strftime('%B %Y')}</p>
                <p><strong>Total akhir (dengan return):</strong><br>
                Rp {fv:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar
            progress = min(current_savings / target_amount, 1.0)
            st.progress(progress)
            st.caption(f"Progress: {progress:.1%}")

if __name__ == "__main__":
    main()
'''

# =============================================================================
# STEP 3: REQUIREMENTS.TXT YANG BERSIH
# =============================================================================

CLEAN_REQUIREMENTS = '''streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
chardet>=5.1.0
requests>=2.31.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl>=3.1.0
xlrd>=2.0.0
python-dateutil>=2.8.0
'''

# =============================================================================
# STEP 4: SCRIPT UNTUK MEMBUAT FILE YANG BERSIH
# =============================================================================

def create_clean_files():
    """
    Buat semua file dengan encoding UTF-8 yang bersih
    """
    
    # Buat direktori jika belum ada
    os.makedirs("utils", exist_ok=True)
    os.makedirs("pages", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # File main.py
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(CLEAN_MAIN_PY)
    
    # Requirements.txt
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(CLEAN_REQUIREMENTS)
    
    # utils/__init__.py
    with open("utils/__init__.py", "w", encoding="utf-8") as f:
        f.write('# -*- coding: utf-8 -*-\n"""Hajj Finance Utils Package"""\n')
    
    # Simple config.py
    config_content = '''# -*- coding: utf-8 -*-
"""
Configuration file untuk Hajj Financial App
"""

class Config:
    # Encoding settings
    DEFAULT_ENCODING = "utf-8"
    
    # Financial constants
    USD_IDR_RATE = 15600
    HAJJ_COST_INFLATION = 0.05
    
    # Directories
    DATA_DIR = "data"
    MODELS_DIR = "models"
    LOGS_DIR = "logs"
'''
    
    with open("utils/config.py", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("✅ Semua file telah dibuat dengan encoding UTF-8 yang bersih!")

# =============================================================================
# STEP 5: LANGKAH-LANGKAH PERBAIKAN
# =============================================================================

def main_fix_procedure():
    """
    Prosedur utama untuk memperbaiki masalah encoding
    """
    
    print("🔧 MEMPERBAIKI MASALAH ENCODING STREAMLIT")
    print("=" * 50)
    
    print("\n1. 🧹 Membersihkan file yang ada...")
    fix_all_python_files(".")
    
    print("\n2. 📁 Membuat file baru yang bersih...")
    create_clean_files()
    
    print("\n3. ✅ Setup selesai!")
    print("\nLangkah selanjutnya:")
    print("- Jalankan: pip install -r requirements.txt")
    print("- Kemudian: streamlit run main.py")
    print("- Jika masih error, hapus cache: streamlit cache clear")

if __name__ == "__main__":
    main_fix_procedure()

# =============================================================================
# COMMAND LINE UNTUK MENJALANKAN SCRIPT INI
# =============================================================================

"""
CARA MENGGUNAKAN SCRIPT INI:

1. SAVE script ini sebagai "fix_encoding.py" dengan encoding UTF-8

2. JALANKAN di terminal/command prompt:
   python fix_encoding.py

3. ATAU jalankan langkah manual:
   
   WINDOWS Command Prompt:
   chcp 65001
   set PYTHONIOENCODING=utf-8
   streamlit run main.py
   
   WINDOWS PowerShell:
   $env:PYTHONIOENCODING="utf-8"
   streamlit run main.py
   
   macOS/Linux:
   export PYTHONIOENCODING=utf-8
   streamlit run main.py

4. JIKA MASIH ERROR:
   - Hapus semua file .py yang lama
   - Copy ulang code dari response ini
   - Pastikan editor (VS Code/Notepad++) set ke UTF-8
   - Save file dengan "Save with Encoding" → UTF-8

5. UNTUK VS CODE:
   - Klik kanan di file
   - "Save with Encoding"
   - Pilih "UTF-8"
   - Atau set default: File → Preferences → Settings → search "encoding" → pilih UTF-8
"""