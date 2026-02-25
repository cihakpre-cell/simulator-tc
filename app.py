import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import urllib.request
from fpdf import FPDF
import tempfile

# --- KONFIGURACE A FONT ---
# Pro Streamlit Cloud je nutné mít v requirements.txt: fpdf2, xlsxwriter
FONT_URL = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"
FONT_BOLD_URL = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf"
FONT_PATH = "DejaVuSans.ttf"
FONT_BOLD_PATH = "DejaVuSans-Bold.ttf"

def download_fonts():
    if not os.path.exists(FONT_PATH):
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
    if not os.path.exists(FONT_BOLD_PATH):
        urllib.request.urlretrieve(FONT_BOLD_URL, FONT_BOLD_PATH)

st.set_page_config(page_title="Simulátor TČ v4.6", layout="wide")
download_fonts()

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

# --- SIDEBAR - VSTUPY ---
with st.sidebar:
    st.header("⚙️ Nastavení projektu")
    nazev_projektu = st.text_input("Název projektu", "Bytový dům - kaskáda")
    nazev_tc = st.text_input("Typ TČ", "NIBE S2125")
    
    with st.expander("🏠 Budova", expanded=True):
        ztrata = st.number_input("Tepelná ztráta [kW]", value=50.0)
        t_vnitrni = st.number_input("Vnitřní teplota [°C]", value=20.0)
        t_design = st.number_input("Venkovní návrhová [°C]", value=-12.0)
        t_spad_max = st.number_input("Max. teplota otopné vody [°C]", value=55.0)
        spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=120.0)
        spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=60.0)

    with st.expander("🔧 Technologie"):
        pocet_tc = st.slider("Počet strojů", 1, 10, 4)
        eta_biv = st.number_input("Účinnost bivalence [%]", value=98) / 100
        char_file = st.file_uploader("Nahrát CSV charakteristiku", type="csv")
        
        default_char = pd.DataFrame({
            "Teplota [°C]": [-15, -7, 2, 7, 15],
            "Výkon [kW]": [7.5, 9.2, 11.5, 12.0, 13.5],
            "COP [-]": [2.1, 2.8, 3.5, 4.2, 5.1]
        })
        df_char = st.data_editor(load_char(char_file) if char_file else default_char, num_rows="dynamic")

    with st.expander("💰 Ceny"):
        cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4500)
        cena_czt = st.number_input("Cena tepla CZT [Kč/GJ]", value=1200)
        investice = st.number_input("Investice [Kč]", value=3500000)

# --- VÝPOČETNÍ JÁDRO ---
tmy_file = st.file_uploader("1. Nahrajte klimatická data (TMY CSV)", type="csv")

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
            
            # Ekvitermní korekce COP
            t_w = 25.0 + (t_spad_max - 25.0) * ((t_vnitrni - t_out) / (t_vnitrni - t_design)) if t_out < t_vnitrni else 25.0
            t_w = min(t_spad_max, max(25, t_w))
            cop_ekv = cop_base * (1 + 0.025 * (t_spad_max - t_w))
            
            q_tc = min(q_need, p_max)
            q_biv = max(0, q_need - q_tc)
            
            # Příkon (Priorita TUV na cop_base, ÚT na cop_ekv)
            el_tuv = min(q_tc, q_tuv_avg) / cop_base
            el_ut = max(0, q_tc - q_tuv_avg) / cop_ekv
            el_biv = q_biv / eta_biv
            
            res.append([t_out, q_need, q_tc, q_biv, el_tuv + el_ut, el_biv, t_w, cop_ekv])

        df_sim = pd.DataFrame(res, columns=['Teplota', 'Potreba_kW', 'Vykon_TC_kW', 'Vykon_Biv_kW', 'Prikon_TC_kW', 'Prikon_Biv_kW', 'Teplota_Vody', 'COP_Ekv'])

        # --- VÝSLEDKY ---
        q_tc_total = df_sim['Vykon_TC_kW'].sum() / 1000
        q_biv_total = df_sim['Vykon_Biv_kW'].sum() / 1000
        el_total = (df_sim['Prikon_TC_kW'].sum() + df_sim['Prikon_Biv_kW'].sum()) / 1000
        
        naklady_czt = (spotreba_ut + spotreba_tuv) * cena_czt * 3.6
        naklady_tc = el_total * cena_el + 15000 # servis
        uspora = naklady_czt - naklady_tc
        
        st.metric("Roční úspora", f"{uspora:,.0f} Kč", delta=f"{uspora/naklady_czt*100:.1f} % oproti CZT")

        # --- PDF GENERÁTOR ---
        def generate_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.add_font("DejaVu", "", FONT_PATH)
            pdf.add_font("DejaVu", "B", FONT_BOLD_PATH)
            
            pdf.set_font("DejaVu", "B", 16)
            pdf.cell(0, 15, f"Technický report: {nazev_projektu}", ln=True, align='C')
            
            pdf.set_font("DejaVu", "", 10)
            pdf.cell(0, 8, f"Technologie: {pocet_tc}x {nazev_tc} | Max spád: {t_spad_max}°C", ln=True)
            pdf.cell(0, 8, f"Ekonomika: Úspora {uspora:,.0f} Kč/rok | Návratnost {investice/uspora:.1f} let", ln=True)
            
            pdf.ln(5)
            pdf.set_font("DejaVu", "B", 11)
            pdf.cell(0, 10, "Bilance bivalence (Roční)", ln=True)
            pdf.set_font("DejaVu", "", 10)
            
            # Tabulka
            pdf.set_fill_color(230, 230, 230)
            pdf.cell(60, 8, "Zdroj", 1, 0, 'L', True)
            pdf.cell(60, 8, "Dodaná energie [MWh]", 1, 0, 'R', True)
            pdf.cell(60, 8, "Podíl [%]", 1, 1, 'R', True)
            
            pdf.cell(60, 8, "Tepelná čerpadla", 1)
            pdf.cell(60, 8, f"{q_tc_total:.2f}", 1, 0, 'R')
            pdf.cell(60, 8, f"{q_tc_total/(q_tc_total+q_biv_total)*100:.1f} %", 1, 1, 'R')
            
            pdf.cell(60, 8, "Bivalentní zdroj", 1)
            pdf.cell(60, 8, f"{q_biv_total:.2f}", 1, 0, 'R')
            pdf.cell(60, 8, f"{q_biv_total/(q_tc_total+q_biv_total)*100:.1f} %", 1, 1, 'R')

            # Graf (přidáme jen jeden klíčový pro ukázku layoutu)
            pdf.ln(10)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_sim['Teplota'], df_sim['Potreba_kW'], color='red', label='Potřeba')
                ax.fill_between(df_sim['Teplota'], 0, df_sim['Vykon_TC_kW'], color='blue', alpha=0.3, label='Pokrytí TČ')
                ax.set_title("Hodinové krytí potřeby tepla")
                fig.savefig(tmp.name)
                pdf.image(tmp.name, x=10, y=pdf.get_y(), w=190)
            
            return pdf.output()

        # --- EXCEL GENERÁTOR ---
        def generate_excel():
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_sim.to_excel(writer, index=False, sheet_name='Simulace 8760h')
                # Přidání legendy a parametrů
                df_params = pd.DataFrame({"Parametr": ["Ztráta", "CZT", "Elektřina"], "Hodnota": [ztrata, cena_czt, cena_el]})
                df_params.to_excel(writer, sheet_name='Vstupy')
            return output.getvalue()

        # --- EXPORTNÍ TLAČÍTKA ---
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📂 Stáhnout technické PDF", generate_pdf(), f"Report_{nazev_projektu}.pdf")
        with c2:
            st.download_button("📊 Stáhnout kompletní Excel", generate_excel(), f"Data_{nazev_projektu}.xlsx")

        # --- ZOBRAZENÍ GRAFŮ V APP ---
        st.subheader("Vizualizace provozu")
        fig_web, ax_web = plt.subplots(figsize=(12, 5))
        ax_web.scatter(df_sim['Teplota'], df_sim['Potreba_kW'], s=1, color='gray', alpha=0.5)
        ax_web.plot(df_char.iloc[:,0], df_char.iloc[:,1] * pocet_tc, 'b-', label='Max. výkon kaskády')
        ax_web.set_xlabel("Venkovní teplota [°C]")
        ax_web.set_ylabel("Výkon [kW]")
        st.pyplot(fig_web)
