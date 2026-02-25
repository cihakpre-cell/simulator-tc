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

# --- KONFIGURACE FONTŮ ---
# Cesty k souborům, které jste nainstaloval/nahrál na GitHub
FONT_REGULAR = "DejaVuSans.ttf"
FONT_BOLD = "DejaVuSans-Bold.ttf"

def load_tmy_robust(file):
    try:
        content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
        header_idx = -1
        for i, line in enumerate(content):
            if 'T2m' in line:
                header_idx = i
                break
        if header_idx == -1: return None
        return pd.read_csv(io.StringIO("\n".join(content[header_idx:])))
    except: return None

def load_char(file):
    try:
        content = file.getvalue().decode('utf-8-sig', errors='ignore')
        sep = ';' if ';' in content.split('\n')[0] else ','
        return pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
    except: return None

# --- STREAMLIT UI ---
st.set_page_config(page_title="Simulator TC v4.4 - FIXED", layout="wide")

with st.sidebar:
    st.header("⚙️ Konfigurace")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    nazev_tc = st.text_input("Model tepelného čerpadla", "NIBE S2125-12")
    
    with st.expander("🏠 Budova a potřeba", expanded=True):
        ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
        t_vnitrni = st.number_input("Žádaná vnitřní teplota [°C]", value=20.0)
        t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
        t_spad = st.text_input("Teplotní spád soustavy [°C]", "55/45")
        t_tuv_cil = st.number_input("Teplota TUV [°C]", value=55.0)
        spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
        spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)

    with st.expander("🔧 Technologie & Charakteristika", expanded=True):
        pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 4)
        eta_biv = st.slider("Účinnost bivalence [%]", 80, 100, 98) / 100
        char_file = st.file_uploader("Nahrát CSV charakteristiku TČ", type="csv")
        if char_file:
            df_char_raw = load_char(char_file)
        else:
            df_char_raw = pd.DataFrame({
                "Teplota [°C]": [-15, -7, 2, 7, 15],
                "Výkon [kW]": [7.5, 9.2, 11.5, 12.0, 13.5],
                "COP [-]": [2.1, 2.8, 3.5, 4.2, 5.1]
            })
        df_char = st.data_editor(df_char_raw, num_rows="dynamic")

    with st.expander("💰 Ekonomika", expanded=True):
        investice = st.number_input("Investice celkem [Kč]", value=4080000)
        cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800)
        cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284)
        servis = st.number_input("Roční servis [Kč]", value=17500)

# --- VÝPOČTY ---
tmy_file = st.file_uploader("Nahrát TMY data (venkovní teploty)", type="csv")

if tmy_file:
    tmy = load_tmy_robust(tmy_file)
    if tmy is not None and df_char is not None:
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
        tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
        
        t_col, v_col, c_col = df_char.columns[0], df_char.columns[1], df_char.columns[2]
        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        potreba_ut_teorie = [max(0, ztrata * (t_vnitrni - t) / (t_vnitrni - t_design)) for t in tmy['T_smooth']]
        k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000) if sum(potreba_ut_teorie) > 0 else 1.0

        try: t_water_max = float(t_spad.split('/')[0])
        except: t_water_max = 55.0

        res = []
        for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
            q_ut = max(0, (ztrata * (t_vnitrni - t_sm) / (t_vnitrni - t_design) * k_oprava))
            q_need = q_ut + q_tuv_avg
            p_max = np.interp(t_out, df_char[t_col], df_char[v_col]) * pocet_tc
            cop_base = np.interp(t_out, df_char[t_col], df_char[c_col])
            
            t_water_actual = 25.0 + (t_water_max - 25.0) * ((t_vnitrni - t_out) / (t_vnitrni - t_design)) if t_out < t_vnitrni else 25.0
            cop_ut = cop_base * (1 + 0.025 * max(0, t_water_max - t_water_actual))
            
            q_tc = min(q_need, p_max)
            q_biv = max(0, q_need - q_tc)
            
            el_tc = (min(q_tc, q_tuv_avg) / cop_base) + (max(0, q_tc - q_tuv_avg) / cop_ut) if cop_base > 0 else 0
            el_biv = q_biv / eta_biv
            res.append([t_out, q_need, q_tc, q_biv, el_tc, el_biv])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])

        # Bod bivalence
        t_biv_val = -12.0
        for t in np.linspace(15, -15, 500):
            q_req = max(0, (ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava)) + q_tuv_avg
            if (np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc) < q_req:
                t_biv_val = t
                break

        # Ekonomika
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        mwh_el_total = (df_sim['El_tc'].sum() + df_sim['El_biv'].sum()) / 1000
        naklady_tc = (mwh_el_total * cena_el) + servis
        uspora = naklady_czt - naklady_tc
        navratnost = investice / uspora if uspora > 0 else 0

        q_tc_s, q_bv_s = df_sim['Q_tc'].sum()/1000, df_sim['Q_biv'].sum()/1000
        el_tc_s, el_bv_s = df_sim['El_tc'].sum()/1000, df_sim['El_biv'].sum()/1000
        df_biv_table = pd.DataFrame({
            "Metrika": ["Tepelná energie (Výstup)", "Spotřeba elektřiny (Vstup)"],
            "TČ [MWh]": [round(q_tc_s, 2), round(el_tc_s, 2)],
            "Bivalence [MWh]": [round(q_bv_s, 2), round(el_bv_s, 2)],
            "Podíl bivalence [%]": [round(q_bv_s/(q_tc_s+q_bv_s)*100, 1) if (q_tc_s+q_bv_s)>0 else 0, 
                                    round(el_bv_s/(el_tc_s+el_bv_s)*100, 1) if (el_tc_s+el_bv_s)>0 else 0]
        })

        # --- ZOBRAZENÍ GRAFŮ (Vaše původní logika) ---
        fig12, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        # ... (zde jsou vaše ploty pro fig12, fig34, fig5, fig6, fig7 - zkráceno pro přehlednost, v kódu zůstávají)
        # Poznámka: V kompletním kódu zde nechte celé definice fig12 až fig7, jak jste je zaslal.
        
        # Simulace vykreslení pro potřeby PDF generátoru níže:
        tr = np.linspace(-15, 18, 100)
        q_p = np.array([max(0, (ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava)) + q_tuv_avg for t in tr])
        p_p = np.array([np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tr])
        ax1.plot(tr, q_p, 'r-', label='Potřeba'); ax1.plot(tr, p_p, 'b--', label='TČ'); ax1.legend()
        
        st.pyplot(fig12)
        st.table(df_biv_table)

        # --- OPRAVENÝ PDF GENERÁTOR ---
        def generate_pdf_v44_fixed():
            pdf = FPDF()
            
            # 1. Kontrola fontů přímo na disku
            has_unicode_font = os.path.exists(FONT_REGULAR) and os.path.exists(FONT_BOLD)
            
            if has_unicode_font:
                try:
                    pdf.add_font("DejaVu", "", FONT_REGULAR) # Odstraněno uni=True, v fpdf2 je to automatické
                    pdf.add_font("DejaVu", "B", FONT_BOLD)
                    current_font = "DejaVu"
                except:
                    has_unicode_font = False
                    current_font = "Helvetica"
            else:
                current_font = "Helvetica"

            # Funkce pro ošetření textu
            def cz(txt):
                if has_unicode_font: return str(txt)
                return "".join([c for c in unicodedata.normalize('NFKD', str(txt)) if not unicodedata.combining(c)])

            pdf.add_page()
            
            # Nadpis
            pdf.set_font(current_font, "B", 16)
            pdf.cell(0, 10, cz(f"TECHNICKÝ REPORT: {nazev_projektu.upper()}"), ln=True, align="C")
            
            pdf.set_font(current_font, "B", 12)
            pdf.cell(0, 10, cz(f"Model TČ: {nazev_tc}"), ln=True, align="C")
            
            pdf.ln(5)
            pdf.set_font(current_font, "B", 11)
            pdf.cell(0, 8, cz("1. VSTUPNÍ PARAMETRY ZADÁNÍ"), ln=True)
            
            pdf.set_font(current_font, "", 10)
            pdf.cell(0, 6, cz(f"- Tepelná ztráta objektu: {ztrata} kW"), ln=True)
            pdf.cell(0, 6, cz(f"- Roční spotřeba: ÚT {spotreba_ut} MWh | TUV {spotreba_tuv} MWh"), ln=True)
            pdf.cell(0, 6, cz(f"- Bod bivalence: {t_biv_val:.1f} °C"), ln=True)
            pdf.cell(0, 6, cz(f"- Roční úspora: {uspora:,.0f} Kč"), ln=True)

            # Vložení grafu 1 (Dynamika)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1:
                fig12.savefig(f1.name, dpi=100)
                pdf.image(f1.name, x=10, y=pdf.get_y()+10, w=180)
            
            # Finalizace
            return pdf.output()

        # Tlačítko v sidebar
        st.sidebar.markdown("---")
        if st.sidebar.button("🚀 GENEROVAT PDF"):
            pdf_bytes = generate_pdf_v44_fixed()
            st.sidebar.download_button(
                label="📥 Stáhnout PDF Report",
                data=pdf_bytes,
                file_name=f"Report_{nazev_projektu}.pdf",
                mime="application/pdf"
            )
