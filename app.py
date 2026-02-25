import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import urllib.request
import unicodedata
from fpdf import FPDF
import tempfile

# --- FUNKCE PRO ABSOLUTNÍ STABILITU TEXTU ---
def bez_diakritiky(text):
    """Odstraní české háčky a čárky pro stabilitu PDF."""
    return "".join(c for c in unicodedata.normalize('NFD', str(text)) if unicodedata.category(c) != 'Mn')

# --- DOWNLOAD FONTU S OVĚŘENÍM ---
URL_FONT = "https://github.com/google/fonts/raw/main/ofl/robotomono/RobotoMono%5Bwght%5D.ttf"
PATH_FONT = "RobotoMono.ttf"

def inicializace_fontu():
    if not os.path.exists(PATH_FONT):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(URL_FONT, headers=headers)
            with urllib.request.urlopen(req) as response:
                with open(PATH_FONT, 'wb') as f:
                    f.write(response.read())
            return True
        except:
            return False
    return True

st.set_page_config(page_title="Simulátor TČ v7.0", layout="wide")
font_ok = inicializace_fontu()

# --- POMOCNÉ FUNKCE PRO DATA ---
def load_char(file):
    try:
        content = file.getvalue().decode('utf-8-sig', errors='ignore')
        sep = ';' if ';' in content.split('\n')[0] else ','
        return pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Nastavení")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sladkovicova")
    
    with st.expander("🔧 Technologie a CSV", expanded=True):
        pocet_tc = st.slider("Počet TČ", 1, 10, 4)
        char_file = st.file_uploader("Nahrát CSV charakteristiku", type="csv")
        default_data = {"Teplota [°C]": [-15, -7, 2, 7, 15], "Výkon [kW]": [7.5, 9.2, 11.5, 12.0, 13.5], "COP [-]": [2.1, 2.8, 3.5, 4.2, 5.1]}
        df_char = st.data_editor(load_char(char_file) if char_file else pd.DataFrame(default_data), num_rows="dynamic")

    with st.expander("🏠 Budova"):
        ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
        t_vnitrni = st.number_input("Vnitřní teplota [°C]", value=20.0)
        t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
        t_spad_max = st.number_input("Max. teplota vody [°C]", value=55.0)
        spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
        spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)

# --- VÝPOČET ---
tmy_file = st.file_uploader("1. Nahrát klimatická data (TMY CSV)", type="csv")

if tmy_file:
    content = tmy_file.getvalue().decode('utf-8', errors='ignore').splitlines()
    header_idx = next((i for i, line in enumerate(content) if 'T2m' in line), -1)
    
    if header_idx != -1:
        tmy = pd.read_csv(io.StringIO("\n".join(content[header_idx:])))
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
        tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
        
        q_tuv_avg = (spotreba_tuv * 1000) / 8760
        potreba_ut_teorie = [max(0, ztrata * (t_vnitrni - t) / (t_vnitrni - t_design)) for t in tmy['T_smooth']]
        k_calib = spotreba_ut / (sum(potreba_ut_teorie) / 1000)
        
        res = []
        for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
            q_ut = max(0, (ztrata * (t_vnitrni - t_sm) / (t_vnitrni - t_design) * k_calib))
            q_need = q_ut + q_tuv_avg
            p_max = np.interp(t_out, df_char.iloc[:,0], df_char.iloc[:,1]) * pocet_tc
            cop_base = np.interp(t_out, df_char.iloc[:,0], df_char.iloc[:,2])
            
            t_w = 25.0 + (t_spad_max - 25.0) * ((t_vnitrni - t_out) / (t_vnitrni - t_design))
            cop_ekv = cop_base * (1 + 0.025 * (t_spad_max - min(t_spad_max, max(25, t_w))))
            
            q_tc = min(q_need, p_max)
            q_biv = max(0, q_need - q_tc)
            res.append([t_out, q_need, q_tc, q_biv, q_tc/cop_ekv, q_biv/0.98])

        df_sim = pd.DataFrame(res, columns=['Teplota', 'Potreba_kW', 'Vykon_TC_kW', 'Vykon_Biv_kW', 'Prikon_TC_kW', 'Prikon_Biv_kW'])

        # --- GENERÁTOR PDF ---
        def create_pdf_final():
            pdf = FPDF()
            pdf.add_page()
            
            # Bezpečné nastavení fontu
            used_font = "Helvetica"
            if font_ok:
                try:
                    pdf.add_font("Roboto", "", PATH_FONT)
                    used_font = "Roboto"
                except: pass
            
            # Texty - pokud nemáme Roboto, musíme odstranit diakritiku, jinak PDF spadne
            titulek = f"REPORT: {nazev_projektu}" if used_font == "Roboto" else bez_diakritiky(f"REPORT: {nazev_projektu}")
            
            pdf.set_font(used_font, "B", 16)
            pdf.cell(0, 10, titulek, ln=True, align='C')
            pdf.ln(10)
            
            # Tabulka bilance (nahrazuje graf č. 6)
            pdf.set_font(used_font, "B", 12)
            pdf.cell(0, 10, bez_diakritiky("1. Tabulka bilance bivalence") if used_font != "Roboto" else "1. Tabulka bilance bivalence", ln=True)
            
            pdf.set_font(used_font, "", 10)
            q_tc_mwh = df_sim['Vykon_TC_kW'].sum() / 1000
            q_biv_mwh = df_sim['Vykon_Biv_kW'].sum() / 1000
            total = q_tc_mwh + q_biv_mwh
            
            bilance_text = (
                f"Energie dodana TC: {q_tc_mwh:.2f} MWh ({q_tc_mwh/total*100:.1f} %)\n"
                f"Energie dodana bivalenci: {q_biv_mwh:.2f} MWh ({q_biv_mwh/total*100:.1f} %)\n"
                f"Celkova rocni potreba: {total:.2f} MWh"
            )
            pdf.multi_cell(0, 8, bez_diakritiky(bilance_text) if used_font != "Roboto" else bilance_text, border=1)
            
            # Graf s velkým odsazením
            pdf.ln(20)
            pdf.set_font(used_font, "B", 12)
            pdf.cell(0, 10, "2. Graf dynamiky provozu", ln=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                plt.figure(figsize=(10, 5))
                plt.plot(df_sim['Teplota'], df_sim['Potreba_kW'], 'r', label='Potreba budovy')
                plt.fill_between(df_sim['Teplota'], 0, df_sim['Vykon_TC_kW'], color='blue', alpha=0.3, label='Pokryti TC')
                plt.grid(True)
                plt.legend()
                plt.savefig(tmp.name, dpi=150)
                pdf.image(tmp.name, x=10, w=180)
            
            return bytes(pdf.output())

        # --- EXPORTY ---
        st.divider()
        c1, c2 = st.columns(2)
        
        with c1:
            st.download_button("📥 Stáhnout PDF Report", data=create_pdf_final(), file_name="Report.pdf", mime="application/pdf")
        
        with c2:
            excel_data = io.BytesIO()
            with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
                df_sim.to_excel(writer, index=False, sheet_name='Simulace')
                df_char.to_excel(writer, index=False, sheet_name='Charakteristika')
            st.download_button("📊 Stáhnout Excel (8760 h)", data=excel_data.getvalue(), file_name="Vypocet.xlsx")

        st.success("Vypocteno. Tabulka bilance a grafy jsou pripraveny v PDF.")
