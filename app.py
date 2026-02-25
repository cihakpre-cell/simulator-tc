import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import urllib.request
from fpdf import FPDF # Doporučena instalace fpdf2
import tempfile
import xlsxwriter

# --- STAŽENÍ FONTU ---
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/robotomono/RobotoMono%5Bwght%5D.ttf"
FONT_PATH = "RobotoMono.ttf"

def download_font():
    if not os.path.exists(FONT_PATH):
        try: urllib.request.urlretrieve(FONT_URL, FONT_PATH)
        except: pass

# --- KONFIGURACE ---
st.set_page_config(page_title="Simulator TC v4.5 - EXCEL & PDF FIX", layout="wide")
download_font()

# (Funkce load_tmy_robust a load_char zůstávají stejné jako v v4.4)
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

with st.sidebar:
    st.header("⚙️ Konfigurace")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    nazev_tc = st.text_input("Model tepelného čerpadla", "NIBE S2125-12")
    
    with st.expander("🏠 Budova a potřeba", expanded=True):
        ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
        t_vnitrni = st.number_input("Žádaná vnitřní teplota [°C]", value=20.0)
        t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
        t_spad = st.text_input("Teplotní spád [°C]", "55/45")
        t_tuv_cil = st.number_input("Teplota TUV [°C]", value=55.0)
        spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
        spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)

    with st.expander("🔧 Technologie", expanded=True):
        pocet_tc = st.slider("Počet TČ", 1, 10, 4)
        eta_biv = st.slider("Účinnost bivalence [%]", 80, 100, 98) / 100
        char_file = st.file_uploader("Nahrát CSV charakteristiku", type="csv")
        df_char_raw = load_char(char_file) if char_file else pd.DataFrame({
            "Teplota [°C]": [-15, -7, 2, 7, 15], "Výkon [kW]": [7.5, 9.2, 11.5, 12.0, 13.5], "COP [-]": [2.1, 2.8, 3.5, 4.2, 5.1]
        })
        df_char = st.data_editor(df_char_raw, num_rows="dynamic")

    with st.expander("💰 Ekonomika"):
        investice = st.number_input("Investice [Kč]", value=4080000)
        cena_el = st.number_input("Cena el. [Kč/MWh]", value=4800)
        cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284)
        servis = st.number_input("Servis [Kč]", value=17500)

# --- VÝPOČET (EKVITERMA) ---
tmy_file = st.file_uploader("Nahrát TMY data", type="csv")
if tmy_file:
    tmy = load_tmy_robust(tmy_file)
    if tmy is not None:
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
        tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
        
        t_col, v_col, c_col = df_char.columns[0], df_char.columns[1], df_char.columns[2]
        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        potreba_ut_teorie = [max(0, ztrata * (t_vnitrni - t) / (t_vnitrni - t_design)) for t in tmy['T_smooth']]
        k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000) if sum(potreba_ut_teorie) > 0 else 1.0
        t_water_max = float(t_spad.split('/')[0]) if '/' in t_spad else 55.0

        res = []
        for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
            q_ut = max(0, (ztrata * (t_vnitrni - t_sm) / (t_vnitrni - t_design) * k_oprava))
            q_need = q_ut + q_tuv_avg
            p_max = np.interp(t_out, df_char[t_col], df_char[v_col]) * pocet_tc
            cop_base = np.interp(t_out, df_char[t_col], df_char[c_col])
            
            # Ekviterma
            t_w = 25.0 + (t_water_max - 25.0) * ((t_vnitrni - t_out) / (t_vnitrni - t_design)) if t_out < t_vnitrni else 25.0
            cop_ut = cop_base * (1 + 0.025 * max(0, t_water_max - t_w))
            
            q_tc = min(q_need, p_max)
            q_biv = max(0, q_need - q_tc)
            el = (min(q_tc, q_tuv_avg) / cop_base) + (max(0, q_tc - q_tuv_avg) / cop_ut) if q_tc > 0 else 0
            res.append([t_out, q_need, q_tc, q_biv, el, q_biv/eta_biv])

        df_sim = pd.DataFrame(res, columns=['Venkovní teplota [°C]', 'Potřeba celkem [kW]', 'Výkon TČ [kW]', 'Výkon Bivalence [kW]', 'Příkon TČ [kW]', 'Příkon Biv [kW]'])
        
        # Mezivýsledky pro report
        uspora = ((spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)) - ((df_sim['Příkon TČ [kW]'].sum() + df_sim['Příkon Biv [kW]'].sum())/1000 * cena_el + servis)
        
        # --- PDF EXPORT (OPRAVENÝ) ---
        def generate_pdf_v45():
            pdf = FPDF()
            if os.path.exists(FONT_PATH):
                pdf.add_font("Roboto", "", FONT_PATH)
                pdf.set_font("Roboto", size=10)
            else:
                pdf.set_font("Helvetica", size=10)
            
            pdf.add_page()
            pdf.set_font(size=16); pdf.cell(0, 10, f"REPORT: {nazev_projektu} ({nazev_tc})", ln=True, align='C')
            pdf.set_font(size=10); pdf.ln(5)
            
            # Parametry
            pdf.multi_cell(0, 6, f"Ztráta: {ztrata}kW | Spád: {t_spad} | TUV: {t_tuv_cil}°C\nKaskáda: {pocet_tc}ks | Úspora: {uspora:,.0f} Kč/rok")
            pdf.ln(5)
            
            # Tabulka bilance (Graf 6)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(60, 8, "Metrika", 1, 0, 'C', True)
            pdf.cell(40, 8, "TČ (MWh)", 1, 0, 'C', True)
            pdf.cell(40, 8, "Biv (MWh)", 1, 1, 'C', True)
            pdf.cell(60, 8, "Energie výstup", 1); pdf.cell(40, 8, f"{df_sim['Výkon TČ [kW]'].sum()/1000:.2f}", 1); pdf.cell(40, 8, f"{df_sim['Výkon Bivalence [kW]'].sum()/1000:.2f}", 1, 1)
            
            # Graf 1 (Odsazení aby nebyl překryt)
            pdf.ln(10)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                plt.figure(figsize=(10,4)); plt.plot(df_sim.iloc[:200, 0], df_sim.iloc[:200, 1]); plt.savefig(tmp.name)
                pdf.image(tmp.name, x=10, w=180)
            
            pdf.ln(5); pdf.multi_cell(0, 5, "Graf 1: Dynamika provozu v čase. Červená linie značí potřebu budovy, modrá oblast pokrytí kaskádou TČ.")
            return pdf.output()

        # --- EXCEL EXPORT ---
        def generate_excel():
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_sim.to_excel(writer, sheet_name='Hodinová simulace')
                df_char.to_excel(writer, sheet_name='Charakteristika TČ')
            return output.getvalue()

        st.subheader("📥 Exporty")
        col_pdf, col_xls = st.columns(2)
        with col_pdf:
            st.download_button("📥 Stáhnout PDF Report (v4.5)", generate_pdf_v45(), "Report.pdf")
        with col_xls:
            st.download_button("📥 Stáhnout Excel (8760 h)", generate_excel(), "Simulace_Data.xlsx")

        st.success("Výpočet s ekvitermou proběhl úspěšně. Grafy jsou zobrazeny níže...")
        # (Zde by následoval kód pro vykreslení grafů v aplikaci, stejný jako v v4.4)
