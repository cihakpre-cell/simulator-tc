import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import re
import unicodedata
import tempfile
from fpdf import FPDF

# --- POMOCNÉ FUNKCE ---
def remove_accents(input_str):
    if not input_str: return ""
    nfkd_form = unicodedata.normalize('NFKD', str(input_str))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def load_tmy_robust(file):
    try:
        content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
        start = -1
        for i, line in enumerate(content):
            if 'T2m' in line and 'time' in line:
                start = i
                break
        if start == -1: return None
        df = pd.read_csv(io.StringIO("\n".join(content[start:])))
        df.columns = df.columns.str.strip()
        df = df[df[df.columns[0]].apply(lambda x: str(x)[:4].isdigit() if pd.notnull(x) else False)].copy()
        df['T2m'] = pd.to_numeric(df['T2m'], errors='coerce')
        return df.dropna(subset=['T2m']).reset_index(drop=True)
    except: return None

def load_char(file):
    try:
        content = file.getvalue().decode('utf-8-sig', errors='ignore')
        sep = ';' if ';' in content.split('\n')[0] else ','
        df = pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
        return df[['Teplota', 'Vykon_kW', 'COP']].copy()
    except: return None

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="Expertní simulátor TČ", layout="wide")
st.title("📊 Energetická analýza kaskády TČ")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Vstupní parametry")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sladkovičova")
    ztrata_celkova = st.number_input("Tepelná ztráta [kW]", value=54.0)
    t_design = st.number_input("Návrhová venkovní teplota [°C]", value=-12.0)
    st.markdown("---")
    fakt_ut = st.number_input("Reálná spotřeba ÚT [MWh/rok]", value=124.0)
    f_tuv = st.number_input("Reálná spotřeba TUV [MWh/rok]", value=76.0)
    st.markdown("---")
    t_privod = st.number_input("Teplota přívod [°C]", value=60)
    t_zpatecka = st.number_input("Teplota zpátečka [°C]", value=50)
    spad_text = f"{int(t_privod)} / {int(t_zpatecka)} °C"
    st.markdown("---")
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 3)
    cena_el_mwh = st.number_input("Cena elektřiny [Kč/MWh]", value=4800.0)
    cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284.0)
    investice = st.number_input("Investice CAPEX [Kč]", value=3800000.0)
    servis = st.number_input("Roční servis [Kč]", value=17000.0)

col_u1, col_u2 = st.columns(2)
with col_u1: tmy_up = st.file_uploader("Nahrajte TMY (CSV)", type="csv")
with col_u2: char_up = st.file_uploader("Nahrajte Charakteristiku TČ (CSV)", type="csv")

if tmy_up and char_up:
    tmy = load_tmy_robust(tmy_up)
    df_char = load_char(char_up)

    if tmy is not None and df_char is not None:
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
        q_tuv_avg = (f_tuv / 8760) * 1000
        
        # Kalibrace
        potreba_ut_teorie = [ztrata_celkova * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
        k_oprava = fakt_ut / (sum(potreba_ut_teorie) / 1000)
        naklady_czt_rok = (fakt_ut + f_tuv) * (cena_gj_czt * 3.6)

        # Hledání bodu bivalence
        t_biv = -99
        for t in np.linspace(15, -15, 500):
            q_need = (ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg
            p_tc = np.interp(t, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc
            if p_tc < q_need:
                t_biv = t
                break

        # Simulace
        res = []
        for t_out, t_smooth in zip(tmy['T2m'], tmy['T_smooth']):
            q_total = max(0, (ztrata_celkova * (20 - t_smooth) / (20 - t_design) * k_oprava)) + q_tuv_avg
            p_max = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc
            cop = np.interp(t_out, df_char['Teplota'], df_char['COP'])
            q_tc = min(q_total, p_max)
            q_biv = max(0, q_total - q_tc)
            res.append([t_out, q_total, q_tc, q_biv, q_tc/cop if q_tc > 0 else 0, q_biv/0.98, p_max])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need_kW', 'Q_tc_kW', 'Q_biv_kW', 'El_tc_kW', 'El_biv_kW', 'P_tc_max'])
        df_sorted = df_sim.sort_values('Temp').reset_index(drop=True)
        
        el_tc_mwh = df_sim['El_tc_kW'].sum() / 1000
        el_biv_mwh = df_sim['El_biv_kW'].sum() / 1000
        naklady_tc = (el_tc_mwh + el_biv_mwh) * cena_el_mwh + servis
        uspora = naklady_czt_rok - naklady_tc

        # --- GRAFICKÝ REPORT 2x2 ---
        fig = plt.figure(figsize=(16, 12))
        plt.suptitle(f"EXPERTNÍ ANALÝZA: {nazev_projektu}", fontsize=18, fontweight='bold')

        # A. Modulace vs Teplota (Původní)
        ax1 = plt.subplot(2, 2, 1)
        t_range = np.linspace(-15, 18, 100)
        q_domu = [(ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg for t in t_range]
        p_kaskady = [np.interp(t, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc for t in t_range]
        ax1.plot(t_range, q_domu, color='red', lw=2, label='Potřeba domu (UT + TUV)')
        ax1.plot(t_range, p_kaskady, color='blue', lw=1, ls='--', alpha=0.5, label='Max. limit kaskády')
        ax1.fill_between(t_range, 0, [min(q,p) for q,p in zip(q_domu, p_kaskady)], color='green', alpha=0.2, label='Pracovní oblast TČ')
        ax1.axvline(t_biv, color='black', ls=':', lw=2, label=f'Bod bivalence {t_biv:.1f}°C')
        ax1.set_title("DYNAMIKA PROVOZU A MODULACE", fontweight='bold')
        ax1.set_xlabel("Teplota [°C]"); ax1.set_ylabel("Výkon [kW]"); ax1.legend(fontsize=9); ax1.grid(alpha=0.2)

        # B. VÝKONOVÁ ROVNOVÁHA (NÁHRADA ZA HISTOGRAM)
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(df_sorted.index, df_sorted['Q_need_kW'], color='red', label='Potřeba objektu', lw=1.5)
        ax2.plot(df_sorted.index, df_sorted['P_tc_max'], color='blue', ls='--', alpha=0.4, label='Max. výkon TČ')
        ax2.fill_between(df_sorted.index, 0, df_sorted['Q_tc_kW'], color='#3498db', alpha=0.3, label='Kryto TČ')
        ax2.fill_between(df_sorted.index, df_sorted['Q_tc_kW'], df_sorted['Q_need_kW'], color='#e74c3c', alpha=0.3, label='Bivalence')
        ax2.set_ylabel("Výkon [kW]"); ax2.set_xlabel("Hodinový průběh (seřazeno dle teploty)")
        
        ax2b = ax2.twinx()
        ax2b.plot(df_sorted.index, df_sorted['Temp'], color='black', alpha=0.3, lw=1, label='Venkovní teplota')
        ax2b.set_ylabel("Teplota [°C]")
        ax2.set_title("VÝKONOVÁ ROVNOVÁHA V ČASE", fontweight='bold')
        ax2.grid(alpha=0.1)

        # C. Ekonomika
        ax3 = plt.subplot(2, 2, 3)
        ax3.bar(['Původní CZT', f'Nové TČ ({pocet_tc}ks)'], [naklady_czt_rok, naklady_tc], color=['#95a5a6', '#27ae60'])
        ax3.set_title(f"ROČNÍ NÁKLADY (SPÁD {spad_text})", fontweight='bold')
        for i, v in enumerate([naklady_czt_rok, naklady_tc]): 
            ax3.text(i, v + 10000, f"{v:,.0f} Kč", ha='center', fontweight='bold')

        # D. Tabulka
        ax4 = plt.subplot(2, 2, 4); ax4.axis('off')
        summary = [["Projekt", nazev_projektu], ["Teplotní spád", spad_text], ["Bod bivalence", f"{t_biv:.1f} °C"], ["Roční úspora", f"{uspora:,.0f} Kč"], ["Návratnost", f"{investice/uspora:.1f} let"]]
        tbl = ax4.table(cellText=summary, loc='center', cellLoc='left', colWidths=[0.4, 0.6])
        tbl.scale(1, 3); tbl.set_fontsize(12)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)

        # --- EXPORTY ---
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📄 Exportovat PDF"):
                pdf = FPDF(); pdf.add_page(); pdf.set_font("Helvetica", 'B', 16)
                pdf.cell(190, 10, f"ANALYZNI REPORT: {remove_accents(nazev_projektu)}", ln=True, align='C')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    fig.savefig(tmp.name, dpi=150); pdf.image(tmp.name, x=10, y=30, w=190)
                st.download_button("⬇️ Stáhnout PDF", data=pdf.output(dest='S').encode('latin-1', 'replace'), file_name="Report.pdf")
        
        with c2:
            output = io.BytesIO()
            df_sim.to_excel(output, index=False)
            st.download_button("⬇️ Stáhnout Excel", data=output.getvalue(), file_name="Analyza.xlsx", mime="application/vnd.ms-excel")
