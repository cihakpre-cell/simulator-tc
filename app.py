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

# --- KONFIGURACE ---
st.set_page_config(page_title="Energetický Simulátor TČ v3.0", layout="wide")

with st.sidebar:
    st.header("⚙️ Vstupní parametry")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    
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
col1, col2 = st.columns(2)
with col1: tmy_file = st.file_uploader("1. Nahrajte TMY (CSV)", type="csv")
with col2: char_file = st.file_uploader("2. Nahrajte Charakteristiku TČ (CSV)", type="csv")

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

        # Simulace
        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        potreba_ut_teorie = [ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * 1.0 for t in tmy['T_smooth']]
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
        
        # Bod bivalence
        t_biv = -12.0
        for t in np.linspace(15, -15, 500):
            q_req = (ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava) + q_tuv_avg
            if (np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc) < q_req:
                t_biv = t
                break

        # Ekonomická bilance
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        mwh_tc_total = (df_sim['El_tc'].sum() + df_sim['El_biv'].sum()) / 1000
        naklady_tc_provoz = (mwh_tc_total * cena_el) + servis
        uspora = naklady_czt - naklady_tc_provoz
        navratnost = (investice - dotace) / uspora if uspora > 0 else 0

        # --- ZOBRAZENÍ GRAFŮ (ZACHOVÁNÍ STÁVAJÍCÍCH 1-5) ---
        st.header(f"📊 Výsledky simulace: {nazev_projektu}")
        
        # Horní blok (1 a 2)
        fig_top, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        tr = np.linspace(-15, 18, 100)
        q_p = np.array([(ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava) + q_tuv_avg for t in tr])
        p_p = np.array([np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tr])
        ax1.plot(tr, q_p, 'r-', lw=2, label='Potřeba (ÚT+TUV)')
        ax1.plot(tr, p_p, 'b--', alpha=0.4, label='Max kaskáda')
        ax1.plot(tr, np.minimum(q_p, p_p), 'g-', lw=5, alpha=0.4, label='Krytí TČ')
        ax1.fill_between(tr, p_p, q_p, where=(q_p > p_p), color='red', alpha=0.2, hatch='XXXX', label='Bivalence')
        ax1.axvline(t_biv, color='k', ls=':', label=f'Bivalence {t_biv:.1f}°C')
        ax1.set_title("1. DYNAMIKA PROVOZU A MODULACE", fontweight='bold')
        ax1.set_xlabel("Teplota [°C]"); ax1.set_ylabel("Výkon [kW]"); ax1.legend(); ax1.grid(alpha=0.2)

        df_sim['Temp_R'] = df_sim['Temp'].round()
        df_t = df_sim.groupby('Temp_R')[['Q_tc', 'Q_biv']].sum().sort_index()
        ax2.bar(df_t.index, df_t['Q_tc'], color='#3498db', label='Energie TČ')
        ax2.bar(df_t.index, df_t['Q_biv'], bottom=df_t['Q_tc'], color='#e74c3c', label='Bivalence')
        ax2.set_title("2. ROZDĚLENÍ ENERGIE DLE TEPLOTY", fontweight='bold')
        ax2.set_xlabel("Teplota [°C]"); ax2.set_ylabel("Energie [kWh]"); ax2.legend(); ax2.grid(alpha=0.1, axis='y')
        st.pyplot(fig_top)

        # Střední blok (3 a 4)
        fig_mid, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 7))
        df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1
        m_df = df_sim.groupby('Month').agg({'Q_tc': 'sum', 'Q_biv': 'sum'}).reset_index()
        ax3.bar(m_df['Month'], m_df['Q_tc']/1000, label='TČ', color='#3498db')
        ax3.bar(m_df['Month'], m_df['Q_biv']/1000, bottom=m_df['Q_tc']/1000, label='Bivalence', color='#e74c3c')
        ax3.set_title("3. MĚSÍČNÍ BILANCE ENERGIE [MWh]", fontweight='bold')
        ax3.set_xticks(range(1,13)); ax3.set_ylabel("MWh"); ax3.legend(); ax3.grid(alpha=0.1, axis='y')

        q_sorted = np.sort(df_sim['Q_need'].values)[::-1]
        p_lim_biv = np.interp(t_biv, df_char[t_col], df_char[v_col]) * pocet_tc
        ax4.plot(range(8760), q_sorted, 'r-', lw=2)
        ax4.fill_between(range(8760), p_lim_biv, q_sorted, where=(q_sorted > p_lim_biv), color='#e74c3c', alpha=0.4, label='Bivalence')
        ax4.fill_between(range(8760), 0, np.minimum(q_sorted, p_lim_biv), color='#3498db', alpha=0.2, label='Kryto TČ')
        ax4.set_title("4. TRVÁNÍ POTŘEBY VÝKONU", fontweight='bold')
        ax4.set_xlabel("Hodin v roce"); ax4.set_ylabel("Výkon [kW]"); ax4.legend(); ax4.grid(alpha=0.2)
        st.pyplot(fig_mid)

        # Graf 5 (Monotóna dle teploty - přes celou šířku)
        fig_mono_t, ax5 = plt.subplots(figsize=(18, 5))
        df_sorted_t = df_sim.sort_values('Temp').reset_index(drop=True)
        ax5.plot(df_sorted_t.index, df_sorted_t['Q_need'], 'r', label='Potřeba domu (ÚT+TUV)')
        ax5.plot(df_sorted_t.index, df_sorted_t['Q_tc'], 'b', label='Krytí TČ')
        biv_idx = df_sorted_t[df_sorted_t['Q_biv'] > 0.1].index
        if len(biv_idx) > 0:
            ax5.fill_between(df_sorted_t.index[:max(biv_idx)], df_sorted_t['Q_tc'][:max(biv_idx)], 
                             df_sorted_t['Q_need'][:max(biv_idx)], color='red', alpha=0.3, label='Oblast bivalence')
        ax5.set_title("5. ČETNOST TEPLOT A BOD BIVALENCE V ROCE", fontweight='bold')
        ax5.set_ylabel("Výkon [kW]"); ax5.set_xlabel("Hodin v roce (seřazeno dle teploty)"); ax5.legend(); ax5.grid(alpha=0.2)
        st.pyplot(fig_mono_t)

        # --- NOVÝ BLOK: GRAF 6 (Výseč) a GRAF 7 (Ekonomika) ---
        st.markdown("---")
        fig_extra, (ax6, ax7) = plt.subplots(1, 2, figsize=(18, 7))
        
        # 6. Výsečový graf (Podíl energie)
        labels = ['Tepelné čerpadlo', 'Bivalence']
        sizes = [df_sim['Q_tc'].sum(), df_sim['Q_biv'].sum()]
        ax6.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c'], 
                textprops={'fontsize': 12}, pctdistance=0.85)
        ax6.set_title("6. PODÍL KRYTÍ ENERGIE TČ VS. BIVALENCE", fontweight='bold')
        
        # 7. Porovnání nákladů CZT vs TČ
        ax7.bar(['Původní CZT', 'Nové s TČ'], [naklady_czt, naklady_tc_provoz], color=['#95a5a6', '#2ecc71'])
        ax7.set_ylabel("Roční náklady [Kč]"); ax7.set_title("7. POROVNÁNÍ ROČNÍCH NÁKLADŮ", fontweight='bold')
        for i, v in enumerate([naklady_czt, naklady_tc_provoz]):
            ax7.text(i, v + 50000, f"{v:,.0f} Kč", ha='center', fontweight='bold')
        
        st.pyplot(fig_extra)

        # --- PDF REPORT (bytes fix) ---
        def generate_pdf_final():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, f"TECHNICKY REPORT: {remove_accents(nazev_projektu).upper()}", ln=True, align="C")
            
            pdf.set_font("Helvetica", "", 10)
            pdf.ln(5)
            # Tabulka parametrů
            pdf.cell(0, 7, f"Tepelna ztrata: {ztrata} kW | Navrhova teplota: {t_design} C | Bivalence: {t_biv:.1f} C", ln=True)
            pdf.cell(0, 7, f"Investice: {investice:,.0f} Kc | Dotace: {dotace:,.0f} Kc | Navratnost: {navratnost:.1f} let", ln=True)
            pdf.cell(0, 7, f"Uspora: {uspora:,.0f} Kc/rok | Podil TC: {(sizes[0]/sum(sizes))*100:.1f} %", ln=True)
            
            # Vkládání grafů
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t1:
                fig_top.savefig(t1.name, dpi=100); pdf.image(t1.name, x=10, y=50, w=190)
            
            pdf.add_page()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t2:
                fig_mid.savefig(t2.name, dpi=100); pdf.image(t2.name, x=10, y=20, w=190)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t3:
                fig_mono_t.savefig(t3.name, dpi=100); pdf.image(t3.name, x=10, y=140, w=190)
            
            pdf.add_page()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t4:
                fig_extra.savefig(t4.name, dpi=100); pdf.image(t4.name, x=10, y=20, w=190)
                
            return bytes(pdf.output())

        # Stažení
        st.sidebar.markdown("---")
        if st.sidebar.button("🚀 Vygenerovat PDF Report"):
            pdf_bytes = generate_pdf_final()
            st.sidebar.download_button("📥 Stáhnout PDF", data=pdf_bytes, file_name=f"Report_{remove_accents(nazev_projektu)}.pdf", mime="application/pdf")
