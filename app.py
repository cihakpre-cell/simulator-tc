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

# --- OPRAVA DIAKRITIKY A STABILITY ---
def odstranit_diakritiku(text):
    return "".join(c for c in unicodedata.normalize('NFD', str(text)) if unicodedata.category(c) != 'Mn')

def safe_pdf_text(text, fonts_ok):
    if fonts_ok:
        return str(text)
    return odstranit_diakritiku(text)

# Přímé linky na prověřené fonty (Google Fonts / GitHub)
URL_FONT = "https://github.com/google/fonts/raw/main/ofl/robotomono/RobotoMono%5Bwght%5D.ttf"
PATH_FONT = "RobotoMono.ttf"

def download_font():
    if not os.path.exists(PATH_FONT):
        try:
            req = urllib.request.Request(URL_FONT, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(PATH_FONT, 'wb') as out_file:
                out_file.write(response.read())
            return True
        except: return False
    return True

st.set_page_config(page_title="Simulátor TČ v6.0", layout="wide")
fonts_ready = download_font()

# --- NAHRÁVÁNÍ DAT ---
def load_char(file):
    try:
        content = file.getvalue().decode('utf-8-sig', errors='ignore')
        sep = ';' if ';' in content.split('\n')[0] else ','
        return pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
    except: return None

# --- SIDEBAR KONFIGURACE ---
with st.sidebar:
    st.header("⚙️ Konfigurace projektu")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    
    with st.expander("🔧 Parametry TČ a CSV", expanded=True):
        pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 4)
        char_file = st.file_uploader("Nahrát CSV charakteristiku (Teplota, Výkon, COP)", type="csv")
        
        default_data = {"Teplota [°C]": [-15, -7, 2, 7, 15], "Výkon [kW]": [7.5, 9.2, 11.5, 12.0, 13.5], "COP [-]": [2.1, 2.8, 3.5, 4.2, 5.1]}
        df_char = st.data_editor(load_char(char_file) if char_file else pd.DataFrame(default_data), num_rows="dynamic")

    with st.expander("🏠 Budova a Spotřeba"):
        ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
        t_vnitrni = st.number_input("Vnitřní teplota [°C]", value=20.0)
        t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
        t_spad_max = st.number_input("Max. teplota vody [°C]", value=55.0)
        spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
        spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)

# --- VÝPOČETNÍ JÁDRO ---
tmy_file = st.file_uploader("1. Nahrajte klimatická data (TMY CSV)", type="csv")

if tmy_file:
    # (Zjednodušený robustní loader pro TMY)
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
            
            # Ekviterma a COP
            t_w = 25.0 + (t_spad_max - 25.0) * ((t_vnitrni - t_out) / (t_vnitrni - t_design))
            cop_ekv = cop_base * (1 + 0.025 * (t_spad_max - min(t_spad_max, max(25, t_w))))
            
            q_tc = min(q_need, p_max)
            q_biv = max(0, q_need - q_tc)
            res.append([t_out, q_need, q_tc, q_biv, q_tc/cop_ekv, q_biv/0.98])

        df_sim = pd.DataFrame(res, columns=['Teplota', 'Potreba_kW', 'Vykon_TC_kW', 'Vykon_Biv_kW', 'Prikon_TC_kW', 'Prikon_Biv_kW'])

        # --- EXPORT PDF ---
        def create_pdf_bytes():
            pdf = FPDF()
            pdf.add_page()
            
            f_family = "Helvetica"
            if fonts_ready:
                try:
                    pdf.add_font("Roboto", "", PATH_FONT)
                    f_family = "Roboto"
                except: pass

            # Záhlaví a parametry
            pdf.set_font(f_family, "B", 16)
            pdf.cell(0, 12, safe_pdf_text(f"TECHNICKÝ REPORT: {nazev_projektu}", fonts_ready), ln=True, align='C')
            pdf.ln(5)
            
            pdf.set_font(f_family, "B", 12)
            pdf.cell(0, 10, safe_pdf_text("1. Energetická bilance (Tabulka 6)", fonts_ready), ln=True)
            
            # Tabulka bilance
            pdf.set_fill_color(240, 240, 240)
            pdf.set_font(f_family, "B", 10)
            pdf.cell(60, 8, safe_pdf_text("Zdroj energie", fonts_ready), 1, 0, 'L', True)
            pdf.cell(60, 8, "MWh / rok", 1, 0, 'C', True)
            pdf.cell(60, 8, safe_pdf_text("Podíl", fonts_ready), 1, 1, 'C', True)
            
            q_tc_mwh = df_sim['Vykon_TC_kW'].sum() / 1000
            q_biv_mwh = df_sim['Vykon_Biv_kW'].sum() / 1000
            total = q_tc_mwh + q_biv_mwh
            
            pdf.set_font(f_family, "", 10)
            pdf.cell(60, 8, safe_pdf_text("Tepelná čerpadla", fonts_ready), 1)
            pdf.cell(60, 8, f"{q_tc_mwh:.2f}", 1, 0, 'C')
            pdf.cell(60, 8, f"{q_tc_mwh/total*100:.1f} %", 1, 1, 'C')
            
            pdf.cell(60, 8, safe_pdf_text("Bivalentní zdroj", fonts_ready), 1)
            pdf.cell(60, 8, f"{q_biv_mwh:.2f}", 1, 0, 'C')
            pdf.cell(60, 8, f"{q_biv_mwh/total*100:.1f} %", 1, 1, 'C')

            # Odsazení pro Graf 1, aby se nepřekrýval
            pdf.ln(15)
            pdf.set_font(f_family, "B", 12)
            pdf.cell(0, 10, safe_pdf_text("2. Dynamika provozu (Graf 1)", fonts_ready), ln=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                plt.figure(figsize=(9, 4.5))
                plt.plot(df_sim['Teplota'], df_sim['Potreba_kW'], 'r', label='Potřeba budovy')
                plt.fill_between(df_sim['Teplota'], 0, df_sim['Vykon_TC_kW'], color='skyblue', alpha=0.6, label='Kryto TČ')
                plt.xlabel("Teplota [°C]"); plt.ylabel("Výkon [kW]"); plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(tmp.name, dpi=120)
                pdf.image(tmp.name, x=10, w=185)
            
            # Musíme vrátit BYTES, ne bytearray!
            return bytes(pdf.output())

        # --- EXPORT EXCEL ---
        def create_excel_bytes():
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_sim.to_excel(writer, index=False, sheet_name='Hodinova_simulace')
                df_char.to_excel(writer, index=False, sheet_name='Charakteristika_TC')
                # Přidání parametrů
                pd.DataFrame({"Vlastnost": ["Ztráta [kW]", "T design [°C]", "Počet TČ"], "Hodnota": [ztrata, t_design, pocet_tc]}).to_excel(writer, sheet_name='Parametry')
            return output.getvalue()

        st.divider()
        st.subheader("📥 Exporty výsledků")
        c1, c2 = st.columns(2)
        
        # Klíčová oprava: Funkce voláme přímo uvnitř download_button a výsledek je v bytes
        c1.download_button(
            label="📂 Stáhnout PDF Report (v6.0)",
            data=create_pdf_bytes(),
            file_name=f"Report_{nazev_projektu}.pdf",
            mime="application/pdf"
        )
        
        c2.download_button(
            label="📊 Stáhnout Excel (8760 h)",
            data=create_excel_bytes(),
            file_name=f"Data_{nazev_projektu}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.success("Simulace proběhla. Exporty jsou připraveny výše.")
