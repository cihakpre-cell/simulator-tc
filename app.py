import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import unicodedata
import tempfile
from fpdf import FPDF

# --- 1. POMOCNÉ FUNKCE ---
def remove_accents(input_str):
    if not input_str: return ""
    nfkd_form = unicodedata.normalize('NFKD', str(input_str))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def load_tmy_robust(file):
    try:
        content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
        header_idx = -1
        for i, line in enumerate(content):
            if 'T2m' in line: header_idx = i; break
        if header_idx == -1: return None
        return pd.read_csv(io.StringIO("\n".join(content[header_idx:])))
    except: return None

def load_char(file):
    try:
        content = file.getvalue().decode('utf-8-sig', errors='ignore')
        sep = ';' if ';' in content.split('\n')[0] else ','
        df = pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
        df.columns = df.columns.str.strip()
        return df[['Teplota', 'Vykon_kW', 'COP']].copy()
    except: return None

# --- 2. KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="Energetický Simulátor TČ", layout="wide")
st.title("🚀 Profesionální simulátor kaskády TČ")

# --- 3. SIDEBAR (Zpřehledněné parametry) ---
st.sidebar.header("⚙️ Systémové parametry")
with st.sidebar:
    nazev_projektu = st.text_input("Název projektu", "SVJ Sladkovicova")
    ztrata = st.number_input("Tepelná ztráta objektu [kW]", value=54.0)
    t_design = st.number_input("Návrhová venkovní teplota [°C]", value=-12.0)
    
    st.markdown("### 🌡️ Otopná soustava")
    t_privod = st.number_input("Návrhová teplota přívodu (při T_design) [°C]", value=60.0)
    t_zpatecka = st.number_input("Návrhová teplota zpátečky (při T_design) [°C]", value=50.0)
    t_min_voda = st.number_input("Minimální teplota vody (při +15°C venku) [°C]", value=35.0)
    
    st.markdown("### 🚿 Příprava TUV")
    t_tuv_cilova = st.number_input("Požadovaná teplota TUV [°C]", value=55.0)
    spotreba_tuv = st.number_input("Roční potřeba tepla pro TUV [MWh/rok]", value=76.0)
    
    st.markdown("### 🏭 Provozní parametry")
    spotreba_ut = st.number_input("Roční potřeba tepla pro ÚT [MWh/rok]", value=124.0)
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 4)
    cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800.0)
    cena_gj_czt = st.number_input("Cena původního tepla (CZT) [Kč/GJ]", value=1284.0)
    investice = st.number_input("Celková investice do TČ [Kč]", value=3800000.0)

# --- 4. VÝPOČTOVÉ JÁDRO ---
if st.sidebar.button("▶️ Spustit simulaci") or 'df_sim' in st.session_state:
    # (Logika načítání dat zůstává stejná jako v předchozím kroku)
    # Pro účely ukázky předpokládáme nahraná data...
    pass # [Zde by následoval zbytek kódu s nahráváním souborů]

# --- 5. GRAF ROČNÍCH NÁKLADŮ (S POPISKY DLE VAŠEHO PŘÁNÍ) ---
def create_econ_plot(naklady_czt, naklady_tc, t_voda_max, t_voda_min, pocet_tc):
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['Puvodni CZT', f'Nove TC ({pocet_tc}ks)']
    costs = [naklady_czt, naklady_tc]
    
    bars = ax.bar(labels, costs, color=['#95a5a6', '#2ecc71'], width=0.6, edgecolor='white', linewidth=1)
    
    # Nastavení os a titulků
    ax.set_ylabel("Naklady [Kc/rok]", fontsize=10)
    ax.set_title(f"ROCNI NAKLADY (SPAD {int(t_voda_max)} / {int(t_voda_min)} deg C)", 
                 fontweight='bold', pad=20)
    
    # Formátování Y osy (tisíce)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' ')))
    
    # Přidání hodnot NAD sloupce
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (max(costs)*0.02),
                f'{int(height):,} Kc'.replace(',', ' '), 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    return fig

# --- 6. EXPORT PDF (SUMÁŘ ZADÁNÍ I VÝSTUPŮ) ---
def generate_pdf_final(params, results, figs):
    pdf = FPDF()
    pdf.add_page()
    
    # Titulek
    pdf.set_font("Helvetica", 'B', 18)
    pdf.cell(190, 15, f"REPORT: {remove_accents(params['nazev'])}", ln=True, align='C')
    pdf.line(10, 25, 200, 25)
    
    # Levý sloupec: Zadání
    pdf.ln(5)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(95, 10, "1. VSTUPNI PARAMETRY", ln=False)
    pdf.cell(95, 10, "2. EKONOMICKY VYSLEDEK", ln=True)
    
    pdf.set_font("Helvetica", '', 10)
    # Data zadání
    y_start = pdf.get_y()
    pdf.cell(95, 6, f"Tepelna ztrata: {params['ztrata']} kW", ln=True)
    pdf.cell(95, 6, f"Navrhovy spad: {params['t_privod']}/{params['t_zpatecka']} C", ln=True)
    pdf.cell(95, 6, f"Cilova teplota TUV: {params['t_tuv']} C", ln=True)
    pdf.cell(95, 6, f"Pocet jednotek v kaskade: {params['pocet_tc']} ks", ln=True)
    
    # Pravý sloupec: Výsledky (pomocí set_xy)
    pdf.set_xy(105, y_start)
    pdf.cell(95, 6, f"Rocni uspora: {int(results['uspora']):,} Kc".replace(',', ' '), ln=True)
    pdf.set_xy(105, pdf.get_y())
    pdf.cell(95, 6, f"Navratnost: {results['navratnost']:.1f} let", ln=True)
    pdf.set_xy(105, pdf.get_y())
    pdf.cell(95, 6, f"SCOP systemu: {results['scop']:.2f}", ln=True)
    
    # Grafy
    pdf.ln(10)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(190, 10, "3. GRAFICKE PREHLEDY", ln=True)
    
    # Vložení grafů (2 vedle sebe, 1 velký pod ně)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t1:
        figs['f1'].savefig(t1.name, dpi=150); pdf.image(t1.name, x=10, y=pdf.get_y(), w=90)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t2:
        figs['f2'].savefig(t2.name, dpi=150); pdf.image(t2.name, x=105, y=pdf.get_y(), w=90)
    
    pdf.set_y(pdf.get_y() + 65)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t3:
        figs['f_econ'].savefig(t3.name, dpi=150); pdf.image(t3.name, x=45, y=pdf.get_y(), w=120)
        
    return pdf.output()
