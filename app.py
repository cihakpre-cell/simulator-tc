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

def load_tmy_pvgis(file):
    try:
        content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
        # Najdeme řádek, kde začínají data (obsahuje T2m)
        data_start_idx = -1
        for i, line in enumerate(content):
            if 'T2m' in line and 'time' in line:
                data_start_idx = i
                break
        
        if data_start_idx == -1:
            st.error("V souboru nebyla nalezena datová hlavička (T2m).")
            return None
            
        # Načteme data od nalezeného řádku
        df = pd.read_csv(io.StringIO("\n".join(content[data_start_idx:])))
        df.columns = df.columns.str.strip()

        # PVGIS používá formát času 20180101:0000 (rok-měsíc-den:hodina)
        # Vytiskneme měsíc z prvních 6 znaků řetězce času
        if 'time(UTC)' in df.columns:
            df['month'] = df['time(UTC)'].str[4:6].astype(int)
        elif 'time' in df.columns:
            df['month'] = df['time'].str[4:6].astype(int)
        
        return df
    except Exception as e:
        st.error(f"Chyba při zpracování PVGIS souboru: {e}")
        return None

def load_char(file):
    try:
        content = file.getvalue().decode('utf-8-sig', errors='ignore')
        sep = ';' if ';' in content.split('\n')[0] else ','
        df = pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
        df.columns = df.columns.str.strip()
        return df[['Teplota', 'Vykon_kW', 'COP']].copy()
    except: return None

# --- 2. KONFIGURACE A SIDEBAR ---
st.set_page_config(page_title="Energetický Simulátor TČ", layout="wide")
st.title("🚀 Profesionální simulátor kaskády TČ")

with st.sidebar:
    st.header("⚙️ Systémové parametry")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sladkovičova")
    ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
    t_design = st.number_input("Návrhová venkovní teplota [°C]", value=-12.0)
    st.markdown("---")
    t_voda_max = st.number_input("TV_Max_Navrh [°C]", value=60.0)
    t_voda_min = st.number_input("TV_Min_Navrh [°C]", value=35.0)
    limit_voda_tc = st.number_input("Limit_Voda_TC (Max z TČ) [°C]", value=55.0)
    st.markdown("---")
    t_tuv_cil = st.number_input("Cílová teplota TUV [°C]", value=55.0)
    spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)
    st.markdown("---")
    spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 4)
    cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800.0)
    cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284.0)
    investice = st.number_input("Investice celkem [Kč]", value=3800000.0)

# --- 3. NAHRÁNÍ DAT ---
st.subheader("📁 1. Krok: Nahrání dat")
col_f1, col_f2 = st.columns(2)
with col_f1: tmy_file = st.file_uploader("Nahrajte TMY (z PVGIS)", type="csv")
with col_f2: char_file = st.file_uploader("Nahrajte Charakteristiku TČ", type="csv")

if tmy_file and char_file:
    tmy = load_tmy_pvgis(tmy_file)
    df_char = load_char(char_file)

    if tmy is not None and df_char is not None:
        st.success("✅ Data z PVGIS úspěšně načtena.")
        
        # Výpočetní logika
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce').fillna(0)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        
        potreba_ut_teorie = [ztrata * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
        k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000)

        res = []
        for _, row in tmy.iterrows():
            t_out, t_sm, m = row['T2m'], row['T_smooth'], int(row['month'])
            t_v_req = np.interp(t_sm, [t_design, 15], [t_voda_max, t_voda_min]) if t_sm < 20 else t_voda_min
            
            k_p = 1 - (max(0, t_v_req - 35.0) * 0.01)
            k_cop_ut = 1 - (max(0, t_v_req - 35.0) * 0.025)
            k_cop_tuv = 1 - (max(0, t_tuv_cil - 35.0) * 0.025)
            
            q_need = (max(0, (ztrata * (20 - t_sm) / (20 - t_design))) * k_oprava) + q_tuv_avg
            p_tc_max = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc * k_p
            
            if t_v_req > limit_voda_tc or t_tuv_cil > limit_voda_tc:
                q_tc, q_biv = 0, q_need
            else:
                q_tc = min(q_need, p_tc_max)
                q_biv = q_need - q_tc
            
            cop_b = np.interp(t_out, df_char['Teplota'], df_char['COP'])
            el_tc = (q_tc * 0.7 / (cop_b * k_cop_ut)) + (q_tc * 0.3 / (cop_b * k_cop_tuv)) if q_tc > 0 else 0
            res.append([t_out, m, q_need, q_tc, q_biv, el_tc, q_biv/0.98])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Month', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])

        # Sumáře
        q_tc_s, q_biv_s = df_sim['Q_tc'].sum()/1000, df_sim['Q_biv'].sum()/1000
        el_tc_s, el_biv_s = df_sim['El_tc'].sum()/1000, df_sim['El_biv'].sum()/1000
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        naklady_tc = (el_tc_s + el_biv_s) * cena_el + 18000
        uspora = naklady_czt - naklady_tc
        scop = q_tc_s / el_tc_s if el_tc_s > 0 else 0

        # --- GRAFY ---
        tab1, tab2 = st.tabs(["📉 Bilance", "💰 Ekonomika & Export"])
        
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            tx = np.sort(df_sim['Temp'].unique())
            qy = [max(0, (ztrata * (20 - t) / (20 - t_design) * k_oprava)) + q_tuv_avg for t in tx]
            ax1.plot(tx, qy, 'r-', label='Potřeba objektu')
            ax1.set_title("Výkonová rovnováha"); ax1.grid(True, alpha=0.3); ax1.legend()
            st.pyplot(fig1)

            fig3, ax3 = plt.subplots(figsize=(10, 4))
            df_sim.groupby('Month')[['Q_tc', 'Q_biv']].sum().plot(kind='bar', stacked=True, ax=ax3, color=['#3498db', '#e74c3c'])
            ax3.set_xticklabels(['Led','Úno','Bře','Dub','Kvě','Čer','Čec','Srp','Zář','Říj','Lis','Pro'], rotation=0)
            st.pyplot(fig3)

        with tab2:
            st.metric("Roční úspora", f"{int(uspora):,} Kč".replace(',',' '))
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            ax4.bar(['CZT', 'TČ'], [naklady_czt, naklady_tc], color=['#95a5a6', '#2ecc71'])
            st.pyplot(fig4)

            if st.button("📄 Generovat PDF report"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", 'B', 16)
                pdf.cell(190, 10, f"REPORT PVGIS: {remove_accents(nazev_projektu)}", ln=True, align='C')
                
                pdf.set_font("Helvetica", '', 10); pdf.ln(10)
                pdf.cell(90, 7, f"Ztrata: {ztrata} kW"); pdf.cell(90, 7, f"Spad: {t_voda_max}/{t_voda_min} C", ln=True)
                pdf.cell(90, 7, f"T_TUV: {t_tuv_cil} C"); pdf.cell(90, 7, f"Pocet TC: {pocet_tc} ks", ln=True)
                pdf.cell(90, 7, f"Uspora: {int(uspora):,} Kc".replace(',',' ')); pdf.cell(90, 7, f"SCOP: {scop:.2f}", ln=True)

                for f in [fig1, fig3, fig4]:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        f.savefig(tmp.name, dpi=110); pdf.image(tmp.name, x=20, w=170); pdf.ln(5)
                
                st.download_button("⬇️ Stáhnout PDF", data=pdf.output(dest='S').encode('latin-1', 'replace'), file_name="Report_PVGIS.pdf")
else:
    st.info("Nahrajte soubor z PVGIS (T2m) a charakteristiku TČ.")
