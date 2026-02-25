import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import urllib.request
from fpdf import FPDF
import tempfile

# --- KONFIGURACE CEST ---
URL_FONT = "https://github.com/reingart/pyfpdf/raw/master/fpdf/font/DejaVuSans.ttf"
URL_FONT_B = "https://github.com/reingart/pyfpdf/raw/master/fpdf/font/DejaVuSans-Bold.ttf"
PATH_FONT = "DejaVuSans.ttf"
PATH_FONT_B = "DejaVuSans-Bold.ttf"

def check_fonts():
    """Pokusí se stáhnout fonty, pokud neexistují."""
    try:
        if not os.path.exists(PATH_FONT):
            urllib.request.urlretrieve(URL_FONT, PATH_FONT)
        if not os.path.exists(PATH_FONT_B):
            urllib.request.urlretrieve(URL_FONT_B, PATH_FONT_B)
        return True
    except:
        return False

st.set_page_config(page_title="Simulátor TČ v4.9", layout="wide")
fonts_available = check_fonts()

# --- POMOCNÉ FUNKCE ---
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
        return pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
    except: return None

# --- SIDEBAR (S CSV CHARAKTERISTIKOU) ---
with st.sidebar:
    st.header("⚙️ Konfigurace")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    
    with st.expander("🏠 Budova a potřeba", expanded=True):
        ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
        t_vnitrni = st.number_input("Cílová teplota [°C]", value=20.0)
        t_design = st.number_input("Venkovní návrhová [°C]", value=-12.0)
        t_spad_max = st.number_input("Max. otopná voda [°C]", value=55.0)
        spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
        spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)

    with st.expander("🔧 Technologie", expanded=True):
        pocet_tc = st.slider("Počet TČ", 1, 10, 4)
        eta_biv = st.number_input("Účinnost bivalence [%]", value=98) / 100
        
        char_file = st.file_uploader("Nahrát CSV charakteristiku TČ", type="csv")
        default_data = {
            "Teplota [°C]": [-15, -7, 2, 7, 15],
            "Výkon [kW]": [7.5, 9.2, 11.5, 12.0, 13.5],
            "COP [-]": [2.1, 2.8, 3.5, 4.2, 5.1]
        }
        df_char = st.data_editor(load_char(char_file) if char_file else pd.DataFrame(default_data), num_rows="dynamic")

# --- VÝPOČET ---
tmy_file = st.file_uploader("1. Nahrát klimatická data (TMY CSV)", type="csv")

if tmy_file:
    tmy = load_tmy_robust(tmy_file)
    if tmy is not None:
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
        tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
        
        q_tuv_avg = (spotreba_tuv * 1000) / 8760
        potreba_ut_teorie = [max(0, ztrata * (t_vnitrni - t) / (t_vnitrni - t_design)) for t in tmy['T_smooth']]
        k_calib = spotreba_ut / (sum(potreba_ut_teorie) / 1000)
        
        res = []
        for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
            q_ut = max(0, ztrata * (t_vnitrni - t_sm) / (t_vnitrni - t_design) * k_calib)
            q_need = q_ut + q_tuv_avg
            p_max = np.interp(t_out, df_char.iloc[:,0], df_char.iloc[:,1]) * pocet_tc
            cop_base = np.interp(t_out, df_char.iloc[:,0], df_char.iloc[:,2])
            
            t_w = 25.0 + (t_spad_max - 25.0) * ((t_vnitrni - t_out) / (t_vnitrni - t_design))
            t_w = min(t_spad_max, max(25, t_w))
            cop_ekv = cop_base * (1 + 0.025 * (t_spad_max - t_w))
            
            q_tc = min(q_need, p_max)
            q_biv = max(0, q_need - q_tc)
            res.append([t_out, q_need, q_tc, q_biv, q_tc/cop_ekv, q_biv/eta_biv, t_w, cop_ekv])

        df_sim = pd.DataFrame(res, columns=['Teplota', 'Potreba_kW', 'Vykon_TC_kW', 'Vykon_Biv_kW', 'Prikon_TC_kW', 'Prikon_Biv_kW', 'T_Vody', 'COP_Ekv'])

        # --- EXPORT PDF (OPRAVENÝ) ---
        def create_pdf():
            pdf = FPDF()
            pdf.add_page()
            
            # Bezpečné nastavení fontu
            font_name = "Helvetica"
            if fonts_available:
                try:
                    pdf.add_font("DejaVu", "", PATH_FONT)
                    pdf.add_font("DejaVu", "B", PATH_FONT_B)
                    font_name = "DejaVu"
                except: pass

            pdf.set_font(font_name, "B", 16)
            pdf.cell(0, 10, f"Report: {nazev_projektu}", ln=True, align='C')
            
            pdf.ln(5)
            pdf.set_font(font_name, "B", 12)
            pdf.cell(0, 10, "Bilance bivalence (Tabulka)", ln=True)
            
            # Kreslení tabulky
            pdf.set_font(font_name, "B", 10)
            pdf.set_fill_color(230, 230, 230)
            pdf.cell(60, 8, "Zdroj", 1, 0, 'C', True)
            pdf.cell(60, 8, "Energie [MWh/rok]", 1, 0, 'C', True)
            pdf.cell(60, 8, "Podil [%]", 1, 1, 'C', True)
            
            q_tc_m = df_sim['Vykon_TC_kW'].sum() / 1000
            q_bi_m = df_sim['Vykon_Biv_kW'].sum() / 1000
            total = q_tc_m + q_bi_m
            
            pdf.set_font(font_name, "", 10)
            pdf.cell(60, 8, "Tepelna cerpadla", 1); pdf.cell(60, 8, f"{q_tc_m:.2f}", 1, 0, 'C'); pdf.cell(60, 8, f"{q_tc_m/total*100:.1f} %", 1, 1, 'C')
            pdf.cell(60, 8, "Bivalence", 1); pdf.cell(60, 8, f"{q_bi_m:.2f}", 1, 0, 'C'); pdf.cell(60, 8, f"{q_bi_m/total*100:.1f} %", 1, 1, 'C')

            # Přidání grafu s odsazením
            pdf.ln(10)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                plt.figure(figsize=(8, 4))
                plt.plot(df_sim['Teplota'], df_sim['Potreba_kW'], 'r', label='Potreba')
                plt.fill_between(df_sim['Teplota'], 0, df_sim['Vykon_TC_kW'], color='blue', alpha=0.3)
                plt.title("Dynamika provozu")
                plt.savefig(tmp.name, dpi=100)
                pdf.image(tmp.name, x=10, w=180)

            return pdf.output()

        # --- EXPORT EXCEL ---
        def create_excel():
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_sim.to_excel(writer, index=False, sheet_name='Simulace')
                df_char.to_excel(writer, sheet_name='Charakteristika')
            return output.getvalue()

        st.divider()
        c1, c2 = st.columns(2)
        c1.download_button("📥 Stáhnout PDF", create_pdf(), "Report.pdf")
        c2.download_button("📊 Stáhnout Excel", create_excel(), "Vypocet.xlsx")
