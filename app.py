import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import unicodedata
from fpdf import FPDF
import tempfile

# --- POMOCNÉ FUNKCE ---
def remove_accents(input_str):
    if not input_str: return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

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

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="Expertní simulátor TČ v3.1", layout="wide")

with st.sidebar:
    st.header("⚙️ Konfigurace")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    
    with st.expander("🏠 Budova a potřeba", expanded=True):
        ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
        t_vnitrni = st.number_input("Žádaná vnitřní teplota [°C]", value=20.0)
        t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
        spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
        spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)

    with st.expander("🔧 Technologie a Ekonomika", expanded=True):
        pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 3)
        eta_biv = st.slider("Účinnost bivalence [%]", 80, 100, 98) / 100
        investice = st.number_input("Investice celkem [Kč]", value=3800000)
        cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800)
        cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284)
        servis = st.number_input("Roční servis [Kč]", value=17000)

# --- VÝPOČTY ---
st.subheader("📁 Nahrání dat")
c1, c2 = st.columns(2)
with c1: tmy_file = st.file_uploader("1. Nahrajte TMY (CSV)", type="csv")
with c2: char_file = st.file_uploader("2. Nahrajte Charakteristiku TČ (CSV)", type="csv")

if tmy_file and char_file:
    tmy = load_tmy_robust(tmy_file)
    df_char = load_char(char_file)

    if tmy is not None and df_char is not None:
        tmy.columns = tmy.columns.str.strip()
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
        tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
        
        df_char.columns = df_char.columns.str.strip()
        t_col, v_col, c_col = 'Teplota', 'Vykon_kW', 'COP'

        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        potreba_ut_teorie = [ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) for t in tmy['T_smooth']]
        k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000)

        res = []
        for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
            q_need = max(0, (ztrata * (t_vnitrni - t_sm) / (t_vnitrni - t_design) * k_oprava)) + q_tuv_avg
            p_max = np.interp(t_out, df_char[t_col], df_char[v_col]) * pocet_tc
            cop_val = np.interp(t_out, df_char[t_col], df_char[c_col])
            q_tc = min(q_need, p_max)
            q_biv = max(0, q_need - q_tc)
            res.append([t_out, q_need, q_tc, q_biv, q_tc/cop_val if q_tc > 0 else 0, q_biv/eta_biv])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
        
        # --- TABULKA BIVALENCE (Kterou jsme řešili) ---
        q_tc_total = df_sim['Q_tc'].sum() / 1000
        q_biv_total = df_sim['Q_biv'].sum() / 1000
        el_tc_total = df_sim['El_tc'].sum() / 1000
        el_biv_total = df_sim['El_biv'].sum() / 1000

        data_biv = {
            "Metrika": ["Tepelná energie (Výstup)", "Spotřeba elektřiny (Vstup)"],
            "TČ [MWh]": [round(q_tc_total, 2), round(el_tc_total, 2)],
            "Bivalence [MWh]": [round(q_biv_total, 2), round(el_biv_total, 2)],
            "Podíl bivalence [%]": [
                round((q_biv_total/(q_tc_total+q_biv_total))*100, 1),
                round((el_biv_total/(el_tc_total+el_biv_total))*100, 1)
            ]
        }
        df_biv_table = pd.DataFrame(data_biv)

        # --- ZOBRAZENÍ GRAFŮ 1-5 ---
        st.header(f"📊 Projekt: {nazev_projektu}")
        
        # Graf 1 a 2
        fig12, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        tr = np.linspace(-15, 18, 100)
        q_p = np.array([(ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava) + q_tuv_avg for t in tr])
        p_p = np.array([np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tr])
        ax1.plot(tr, q_p, 'r-', lw=2, label='Potřeba')
        ax1.plot(tr, p_p, 'b--', alpha=0.4, label='Max kaskáda')
        ax1.fill_between(tr, p_p, q_p, where=(q_p > p_p), color='red', alpha=0.2, hatch='XXXX', label='Bivalence')
        ax1.set_title("1. DYNAMIKA PROVOZU", fontweight='bold')
        ax1.legend(); ax1.grid(alpha=0.2)

        df_sim['Temp_R'] = df_sim['Temp'].round()
        df_t = df_sim.groupby('Temp_R')[['Q_tc', 'Q_biv']].sum().sort_index()
        ax2.bar(df_t.index, df_t['Q_tc'], color='#3498db', label='TČ')
        ax2.bar(df_t.index, df_t['Q_biv'], bottom=df_t['Q_tc'], color='#e74c3c', label='Bivalence')
        ax2.set_title("2. ENERGIE DLE TEPLOTY", fontweight='bold')
        ax2.legend(); ax2.grid(alpha=0.1, axis='y')
        st.pyplot(fig12)

        # Graf 3 a 4
        fig34, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 7))
        df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1
        m_df = df_sim.groupby('Month').agg({'Q_tc': 'sum', 'Q_biv': 'sum'}).reset_index()
        ax3.bar(m_df['Month'], m_df['Q_tc']/1000, color='#3498db')
        ax3.bar(m_df['Month'], m_df['Q_biv']/1000, bottom=m_df['Q_tc']/1000, color='#e74c3c')
        ax3.set_title("3. MĚSÍČNÍ BILANCE [MWh]", fontweight='bold')

        q_sorted = np.sort(df_sim['Q_need'].values)[::-1]
        ax4.plot(range(8760), q_sorted, 'r-', lw=2)
        ax4.set_title("4. TRVÁNÍ POTŘEBY VÝKONU", fontweight='bold')
        st.pyplot(fig34)

        # Graf 5
        fig5, ax5 = plt.subplots(figsize=(18, 5))
        df_sorted_t = df_sim.sort_values('Temp').reset_index(drop=True)
        ax5.plot(df_sorted_t.index, df_sorted_t['Q_need'], 'r')
        ax5.plot(df_sorted_t.index, df_sorted_t['Q_tc'], 'b')
        ax5.set_title("5. ČETNOST TEPLOT (TEPLOTNÍ MONOTÓNA)", fontweight='bold')
        st.pyplot(fig5)

        # --- NOVÝ BLOK: TABULKA A DOPLŇKOVÉ GRAFY ---
        st.markdown("---")
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("6. Bilance bivalence (Vstup vs. Výstup)")
            st.table(df_biv_table)
            
            fig6, ax6 = plt.subplots(figsize=(8, 6))
            ax6.pie([q_tc_total, q_biv_total], labels=['TČ', 'Biv'], autopct='%1.1f%%', colors=['#3498db', '#e74c3c'])
            ax6.set_title("Podíl na dodané tepelné energii")
            st.pyplot(fig6)

        with col_right:
            st.subheader("7. Ekonomické porovnání")
            naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
            naklady_tc = (el_tc_total + el_biv_total) * cena_el + servis
            
            fig7, ax7 = plt.subplots(figsize=(8, 6))
            ax7.bar(['CZT', 'TČ'], [naklady_czt, naklady_tc], color=['#95a5a6', '#2ecc71'])
            ax7.set_ylabel("Kč / rok")
            st.pyplot(fig7)
            st.metric("Roční úspora", f"{naklady_czt - naklady_tc:,.0f} Kč")

        # --- PDF REPORT ---
        def generate_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, f"REPORT: {remove_accents(nazev_projektu).upper()}", ln=True, align="C")
            
            pdf.set_font("Helvetica", "", 10)
            pdf.ln(10)
            pdf.cell(0, 8, f"Bilance bivalence (Vystupni energie): {data_biv['Podíl bivalence [%]'][0]} %", ln=True)
            pdf.cell(0, 8, f"Bilance bivalence (Vstupni elektrina): {data_biv['Podíl bivalence [%]'][1]} %", ln=True)
            
            # Export grafů
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t1:
                fig12.savefig(t1.name); pdf.image(t1.name, x=10, y=50, w=190)
            pdf.add_page()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t2:
                fig34.savefig(t2.name); pdf.image(t2.name, x=10, y=20, w=190)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t3:
                fig5.savefig(t3.name); pdf.image(t3.name, x=10, y=140, w=190)
            
            return bytes(pdf.output())

        st.sidebar.markdown("---")
        if st.sidebar.button("Pripravit PDF"):
            st.sidebar.download_button("Stahnout PDF", generate_pdf(), f"Report_{remove_accents(nazev_projektu)}.pdf")
