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
st.set_page_config(page_title="Finální Simulátor TČ v3.2", layout="wide")

# --- SIDEBAR S KOMPLETNÍMI PARAMETRY ---
with st.sidebar:
    st.header("⚙️ Konfigurace")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sladkovicova")
    
    with st.expander("🏠 Budova a potřeba", expanded=True):
        ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
        t_vnitrni = st.number_input("Žádaná vnitřní teplota [°C]", value=20.0)
        t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
        spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
        spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)

    with st.expander("🔧 Technologie a Teploty", expanded=True):
        pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 3)
        t_spad_ut = st.text_input("Teplotní spád ÚT", value="60/50")
        t_tuv_vystup = st.number_input("Výstupní teplota TUV [°C]", value=55)
        eta_biv = st.slider("Účinnost bivalence [%]", 80, 100, 98) / 100

    with st.expander("💰 Ekonomika", expanded=True):
        investice = st.number_input("Investice celkem [Kč]", value=3800000)
        dotace = st.number_input("Dotace [Kč]", value=0)
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
        
        # Ekonomické výsledky
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        naklady_tc = ((df_sim['El_tc'].sum() + df_sim['El_biv'].sum()) / 1000) * cena_el + servis
        uspora = naklady_czt - naklady_tc
        navratnost = (investice - dotace) / uspora if uspora > 0 else 0

        # --- GRAFY 1-5 (STÁVAJÍCÍ PODOBA) ---
        st.header(f"📊 Report: {nazev_projektu}")
        
        fig12, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        # Graf 1
        tr = np.linspace(-15, 18, 100)
        q_p = np.array([(ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava) + q_tuv_avg for t in tr])
        p_p = np.array([np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tr])
        ax1.plot(tr, q_p, 'r-', label='Potřeba')
        ax1.plot(tr, p_p, 'b--', alpha=0.4)
        ax1.fill_between(tr, p_p, q_p, where=(q_p > p_p), color='red', alpha=0.2, hatch='XXXX', label='Bivalence')
        ax1.set_title("1. DYNAMIKA PROVOZU")
        ax1.legend()

        # Graf 2
        df_sim['Temp_R'] = df_sim['Temp'].round()
        df_t = df_sim.groupby('Temp_R')[['Q_tc', 'Q_biv']].sum()
        ax2.bar(df_t.index, df_t['Q_tc'], color='#3498db', label='TČ')
        ax2.bar(df_t.index, df_t['Q_biv'], bottom=df_t['Q_tc'], color='#e74c3c', label='Biv')
        ax2.set_title("2. ENERGIE DLE TEPLOTY")
        st.pyplot(fig12)

        # Graf 3 a 4 (OPRAVA BAREV V GRAFU 3)
        fig34, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 7))
        df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1
        m_df = df_sim.groupby('Month').agg({'Q_tc': 'sum', 'Q_biv': 'sum'})
        ax3.bar(m_df.index, m_df['Q_tc']/1000, color='#add8e6', label='TČ (bleděmodrá)') # Opraveno
        ax3.bar(m_df.index, m_df['Q_biv']/1000, bottom=m_df['Q_tc']/1000, color='#ff0000', label='Bivalence (červená)') # Opraveno
        ax3.set_title("3. MĚSÍČNÍ BILANCE [MWh]")
        ax3.legend()

        q_sorted = np.sort(df_sim['Q_need'].values)[::-1]
        ax4.plot(range(8760), q_sorted, 'r-')
        ax4.set_title("4. TRVÁNÍ POTŘEBY VÝKONU")
        st.pyplot(fig34)

        # Graf 5
        fig5, ax5 = plt.subplots(figsize=(18, 5))
        df_sort_t = df_sim.sort_values('Temp').reset_index(drop=True)
        ax5.plot(df_sort_t.index, df_sort_t['Q_need'], 'r')
        ax5.plot(df_sort_t.index, df_sort_t['Q_tc'], 'b')
        ax5.set_title("5. TEPLOTNÍ MONOTÓNA")
        st.pyplot(fig5)

        # --- TABULKA BIVALENCE + EKONOMIKA (GRAFY 6-7) ---
        st.markdown("---")
        col_l, col_r = st.columns(2)
        
        with col_l:
            st.subheader("6. Bilance bivalence (Vstup/Výstup)")
            q_tc_s = df_sim['Q_tc'].sum()/1000
            q_bv_s = df_sim['Q_biv'].sum()/1000
            el_tc_s = df_sim['El_tc'].sum()/1000
            el_bv_s = df_sim['El_biv'].sum()/1000
            
            df_res_biv = pd.DataFrame({
                "Metrika": ["Tepelná energie (Výstup)", "Spotřeba el. (Vstup)"],
                "TČ [MWh]": [round(q_tc_s, 2), round(el_tc_s, 2)],
                "Biv [MWh]": [round(q_bv_s, 2), round(el_bv_s, 2)],
                "Podíl [%]": [round(q_bv_s/(q_tc_s+q_bv_s)*100, 1), round(el_bv_s/(el_tc_s+el_bv_s)*100, 1)]
            })
            st.table(df_res_biv)

        with col_r:
            st.subheader("7. Ekonomické srovnání")
            fig7, ax7 = plt.subplots(figsize=(8, 6))
            ax7.bar(['Původní CZT', 'Nové TČ'], [naklady_czt, naklady_tc], color=['gray', 'green'])
            st.pyplot(fig7)
            st.write(f"**Roční úspora:** {uspora:,.0f} Kč | **Návratnost:** {navratnost:.1f} let")

        # --- KOMPLETNÍ PDF REPORT ---
        def generate_pdf_full():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, f"TECHNICKY REPORT: {remove_accents(nazev_projektu).upper()}", ln=True, align="C")
            
            # SEKCE: VSTUPNÍ PARAMETRY
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, "1. Vstupni parametry projektu", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 7, f"Tepelna ztrata: {ztrata} kW | Navrhova teplota: {t_design} C", ln=True)
            pdf.cell(0, 7, f"Spotreba UT: {spotreba_ut} MWh | TUV: {spotreba_tuv} MWh", ln=True)
            pdf.cell(0, 7, f"Kaskada: {pocet_tc} ks TC | Teplotni spad: {remove_accents(t_spad_ut)}", ln=True)
            
            # SEKCE: EKONOMICKÝ SUMÁŘ
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, "2. Ekonomicky sumar", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 7, f"Celkova investice (po dotaci): {investice-dotace:,.0f} Kc", ln=True)
            pdf.cell(0, 7, f"Rocni naklady CZT: {naklady_czt:,.0f} Kc | Rocni naklady TC: {naklady_tc:,.0f} Kc", ln=True)
            pdf.cell(0, 7, f"Rocni uspora: {uspora:,.0f} Kc | Navratnost: {navratnost:.1f} let", ln=True)
            
            # GRAFY
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t1:
                fig12.savefig(t1.name, dpi=100); pdf.image(t1.name, x=10, y=90, w=190)
            
            pdf.add_page()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t2:
                fig34.savefig(t2.name, dpi=100); pdf.image(t2.name, x=10, y=20, w=190)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t3:
                fig5.savefig(t3.name, dpi=100); pdf.image(t3.name, x=10, y=140, w=190)
                
            return bytes(pdf.output())

        st.sidebar.markdown("---")
        if st.sidebar.button("🚀 Vygenerovat finální PDF"):
            pdf_data = generate_pdf_full()
            st.sidebar.download_button("📥 Stáhnout PDF", data=pdf_data, file_name=f"Report_{remove_accents(nazev_projektu)}.pdf")
