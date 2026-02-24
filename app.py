import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import unicodedata
from fpdf import FPDF
import tempfile

# --- FUNKCE PRO ODSTRANĚNÍ DIAKRITIKY ---
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
st.set_page_config(page_title="Expertní simulátor TČ v2.3", layout="wide")

# --- SIDEBAR: VŠECHNY PARAMETRY ---
with st.sidebar:
    st.header("⚙️ Konfigurace projektu")
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

# --- VÝPOČETNÍ JÁDRO ---
st.subheader("📁 Datové podklady")
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

        # Simulace
        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        potreba_ut_teorie = [ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) if t < t_vnitrni else 0 for t in tmy['T_smooth']]
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

        # --- VIZUALIZACE ---
        st.header(f"📊 Report projektu: {nazev_projektu}")
        
        # GRAFY 1 & 2
        fig_top, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        tr = np.linspace(-15, 18, 100)
        q_p = np.array([(ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava) + q_tuv_avg for t in tr])
        p_p = np.array([np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tr])
        
        ax1.plot(tr, q_p, 'r-', lw=2, label='Potreba (UT+TUV)')
        ax1.plot(tr, p_p, 'b--', alpha=0.4, label='Max kaskada')
        ax1.plot(tr, np.minimum(q_p, p_p), 'g-', lw=5, alpha=0.4, label='Kryti TC')
        ax1.fill_between(tr, p_p, q_p, where=(q_p > p_p), color='red', alpha=0.2, hatch='XXXX', label='Bivalence')
        ax1.axvline(t_biv, color='k', ls=':', label=f'Bivalence {t_biv:.1f}C')
        ax1.set_title("1. DYNAMIKA PROVOZU A MODULACE", fontweight='bold')
        ax1.set_xlabel("Venkovni teplota [C]"); ax1.set_ylabel("Vykon [kW]"); ax1.legend(); ax1.grid(alpha=0.2)

        df_sim['Temp_R'] = df_sim['Temp'].round()
        df_t = df_sim.groupby('Temp_R')[['Q_tc', 'Q_biv']].sum().sort_index()
        ax2.bar(df_t.index, df_t['Q_tc'], color='#3498db', label='Energie TC')
        ax2.bar(df_t.index, df_t['Q_biv'], bottom=df_t['Q_tc'], color='#e74c3c', label='Bivalence')
        ax2.set_title("2. ROZDELENI ENERGIE DLE TEPLOTY", fontweight='bold')
        ax2.set_xlabel("Teplota [C]"); ax2.set_ylabel("Energie [kWh]"); ax2.legend(); ax2.grid(alpha=0.1, axis='y')
        st.pyplot(fig_top)

        # GRAFY 3 & 4
        fig_bot, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 7))
        df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1
        m_df = df_sim.groupby('Month').agg({'Q_tc': 'sum', 'Q_biv': 'sum'}).reset_index()
        ax3.bar(m_df['Month'], m_df['Q_tc']/1000, label='TC', color='#3498db')
        ax3.bar(m_df['Month'], m_df['Q_biv']/1000, bottom=m_df['Q_tc']/1000, label='Bivalence', color='#e74c3c')
        ax3.set_title("3. MESICNI BILANCE ENERGIE [MWh]", fontweight='bold')
        ax3.set_xticks(range(1,13)); ax3.set_ylabel("MWh"); ax3.legend(); ax3.grid(alpha=0.1, axis='y')

        q_sorted = np.sort(df_sim['Q_need'].values)[::-1]
        p_lim_biv = np.interp(t_biv, df_char[t_col], df_char[v_col]) * pocet_tc
        ax4.plot(range(8760), q_sorted, 'r-', lw=2)
        ax4.fill_between(range(8760), p_lim_biv, q_sorted, where=(q_sorted > p_lim_biv), color='#e74c3c', alpha=0.4, label='Bivalence')
        ax4.fill_between(range(8760), 0, np.minimum(q_sorted, p_lim_biv), color='#3498db', alpha=0.2, label='Kryto TC')
        ax4.set_title("4. TRVANI POTREBY VYKONU", fontweight='bold')
        ax4.set_xlabel("Hodin v roce"); ax4.set_ylabel("Vykon [kW]"); ax4.legend(); ax4.grid(alpha=0.2)
        st.pyplot(fig_bot)

        # GRAF 5
        st.markdown("---")
        fig_mono_t, ax5 = plt.subplots(figsize=(18, 5))
        df_sorted_t = df_sim.sort_values('Temp').reset_index(drop=True)
        ax5.plot(df_sorted_t.index, df_sorted_t['Q_need'], 'r', label='Potreba domu (UT+TUV)')
        ax5.plot(df_sorted_t.index, df_sorted_t['Q_tc'], 'b', label='Kryti TC')
        biv_idx = df_sorted_t[df_sorted_t['Q_biv'] > 0.1].index
        if len(biv_idx) > 0:
            ax5.fill_between(df_sorted_t.index[:max(biv_idx)], df_sorted_t['Q_tc'][:max(biv_idx)], 
                             df_sorted_t['Q_need'][:max(biv_idx)], color='red', alpha=0.3, label='Oblast bivalence')
        ax5.set_title("5. CETNOST TEPLOT A BOD BIVALENCE V ROCE", fontweight='bold')
        ax5.set_ylabel("Vykon [kW]"); ax5.set_xlabel("Hodin v roce (vzestupne dle teploty)"); ax5.legend(); ax5.grid(alpha=0.2)
        st.pyplot(fig_mono_t)

        # --- PDF GENERÁTOR ---
        def generate_pdf_bytes():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, f"Technicko-ekonomicky report: {remove_accents(nazev_projektu)}", ln=True, align="C")
            
            pdf.set_font("Helvetica", "", 10)
            pdf.ln(5)
            pdf.cell(0, 8, f"Tepelna ztrata: {ztrata} kW | Navrhova teplota: {t_design} C", ln=True)
            pdf.cell(0, 8, f"Rocni spotreba UT: {spotreba_ut} MWh | TUV: {spotreba_tuv} MWh", ln=True)
            pdf.cell(0, 8, f"Pocet TC v kaskade: {pocet_tc} | Teplotni spad: {remove_accents(t_spad_ut)}", ln=True)
            pdf.cell(0, 8, f"Vypocteny bod bivalence: {t_biv:.1f} C", ln=True)
            
            pdf.ln(5)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t1:
                fig_top.savefig(t1.name, dpi=100)
                pdf.image(t1.name, x=10, y=60, w=190)
            
            pdf.add_page()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t2:
                fig_bot.savefig(t2.name, dpi=100)
                pdf.image(t2.name, x=10, y=20, w=190)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t3:
                fig_mono_t.savefig(t3.name, dpi=100)
                pdf.image(t3.name, x=10, y=140, w=190)
                
            # KLÍČOVÁ OPRAVA: Výstup musí být bytes
            return bytes(pdf.output())

        # --- SIDEBAR DOWNLOAD ---
        st.sidebar.markdown("---")
        try:
            pdf_data = generate_pdf_bytes()
            st.sidebar.download_button(
                label="📥 Stahnout PDF Report",
                data=pdf_data,
                file_name=f"Report_{remove_accents(nazev_projektu)}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.sidebar.error(f"Chyba pri generovani PDF: {e}")
