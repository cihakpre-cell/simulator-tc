import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import re
import unicodedata

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
st.title("🚀 Expertní analýza kaskády TČ (Colab Edition)")

# --- SIDEBAR (Architektura vstupů) ---
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

# --- NAHRÁNÍ DAT ---
col_u1, col_u2 = st.columns(2)
with col_u1: tmy_up = st.file_uploader("Nahrajte TMY (CSV)", type="csv")
with col_u2: char_up = st.file_uploader("Nahrajte Charakteristiku TČ (CSV)", type="csv")

if tmy_up and char_up:
    tmy = load_tmy_robust(tmy_up)
    df_char = load_char(char_up)

    if tmy is not None and df_char is not None:
        # 1. KALIBRACE A PŘÍPRAVA (Dle Colabu)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
        q_tuv_avg = (f_tuv / 8760) * 1000
        
        # Výpočet kalibračního faktoru k_oprava
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

        # 2. CELOROČNÍ SIMULACE
        res = []
        for t_out, t_smooth in zip(tmy['T2m'], tmy['T_smooth']):
            q_total = max(0, (ztrata_celkova * (20 - t_smooth) / (20 - t_design) * k_oprava)) + q_tuv_avg
            p_max = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc
            cop = np.interp(t_out, df_char['Teplota'], df_char['COP'])
            q_tc = min(q_total, p_max)
            q_biv = max(0, q_total - q_tc)
            res.append([t_out, q_total, q_tc, q_biv, q_tc/cop if q_tc > 0 else 0, q_biv/0.98])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need_kW', 'Q_tc_kW', 'Q_biv_kW', 'El_tc_kW', 'El_biv_kW'])
        el_tc_mwh = df_sim['El_tc_kW'].sum() / 1000
        el_biv_mwh = df_sim['El_biv_kW'].sum() / 1000
        naklady_tc = (el_tc_mwh + el_biv_mwh) * cena_el_mwh + servis
        uspora = naklady_czt_rok - naklady_tc

        # --- 3. GRAFICKÝ REPORT (IDENTICKÝ S COLABEM) ---
        fig = plt.figure(figsize=(16, 12))
        plt.suptitle(f"EXPERTNÍ ANALÝZA: {nazev_projektu}", fontsize=18, fontweight='bold')

        # A. Výkonová charakteristika
        ax1 = plt.subplot(2, 2, 1)
        t_range = np.linspace(-15, 18, 100)
        q_domu = [(ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg for t in t_range]
        p_kaskady = [np.interp(t, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc for t in t_range]
        p_provoz_tc = [min(q, p) for q, p in zip(q_domu, p_kaskady)]

        ax1.plot(t_range, q_domu, color='red', lw=1.5, label='Potřeba domu (UT + TUV)')
        ax1.plot(t_range, p_kaskady, color='blue', lw=1, ls='--', alpha=0.3, label='Max. limit kaskády')
        ax1.plot(t_range, p_provoz_tc, color='green', lw=5, alpha=0.5, label='Skutečný výkon TČ (modulace)')

        t_mraz = np.linspace(-15, t_biv, 50)
        q_mraz = [(ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg for t in t_mraz]
        p_mraz = [np.interp(t, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc for t in t_mraz]
        ax1.fill_between(t_mraz, p_mraz, q_mraz, color='red', alpha=0.2, hatch='\\\\\\', label='Bivalentní dohřev')

        info_text = (f"Roční bilance:\n"
                     f"Spotřeba TČ: {el_tc_mwh:.1f} MWh\n"
                     f"Spotřeba Biv: {el_biv_mwh:.1f} MWh\n"
                     f"Úspora vs CZT: {uspora:,.0f} Kč/rok")
        ax1.text(0.05, 0.95, info_text, transform=ax1.transAxes, fontsize=10, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        ax1.axvline(t_biv, color='black', ls=':', lw=2, label=f'Bod bivalence {t_biv:.1f}°C')
        ax1.set_title("DYNAMIKA PROVOZU A MODULACE", fontweight='bold')
        ax1.legend(loc='lower right', fontsize=9); ax1.grid(alpha=0.2)
        ax1.set_xlabel("Venkovní teplota [°C]"); ax1.set_ylabel("Výkon [kW]")

        # B. Histogram četnosti teplot
        ax2 = plt.subplot(2, 2, 2)
        n, bins, patches = ax2.hist(tmy['T2m'], bins=np.arange(-15, 20, 1), color='skyblue', edgecolor='white')
        for i in range(len(patches)):
            if bins[i] < t_biv: patches[i].set_facecolor('#ff4444')
        ax2.annotate('Zóna bivalence\n(dohřev)', xy=(t_biv-1, 50), xytext=(t_biv-8, 400),
                     arrowprops=dict(facecolor='red', shrink=0.05), color='red', fontweight='bold')
        ax2.set_title("ROZDĚLENÍ TEPLOT V ROCE", fontweight='bold')

        # C. Ekonomické porovnání
        ax3 = plt.subplot(2, 2, 3)
        ax3.bar(['Původní CZT', f'Nové TČ ({pocet_tc}ks)'], [naklady_czt_rok, naklady_tc], color=['#95a5a6', '#27ae60'])
        ax3.set_title(f"ROČNÍ NÁKLADY (SPÁD {spad_text})", fontweight='bold')
        for i, v in enumerate([naklady_czt_rok, naklady_tc]): 
            ax3.text(i, v + 10000, f"{v:,.0f} Kč", ha='center', fontweight='bold', fontsize=11)

        # D. Tabulka výsledků
        ax4 = plt.subplot(2, 2, 4); ax4.axis('off')
        summary = [
            ["Projekt", nazev_projektu], 
            ["Teplotní spád", spad_text], 
            ["Bod bivalence", f"{t_biv:.1f} °C"], 
            ["Roční úspora", f"{uspora:,.0f} Kč"], 
            ["Návratnost investice", f"{investice/uspora:.1f} let"]
        ]
        tbl = ax4.table(cellText=summary, loc='center', cellLoc='left', colWidths=[0.45, 0.55])
        tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 3.2)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor('lightgray')
            if col == 0: cell.set_text_props(fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)

        # --- EXPORTY ---
        st.markdown("---")
        st.subheader("📥 Export výsledků")
        
        # Excel export
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame(summary, columns=['Parametr', 'Hodnota']).to_excel(writer, sheet_name='Souhrn', index=False)
            df_sim.to_excel(writer, sheet_name='Hodinova_Simulace', index=False)
        
        st.download_button(
            label="⬇️ Stáhnout Excel analýzu",
            data=output.getvalue(),
            file_name=f"Analyza_{remove_accents(nazev_projektu)}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
