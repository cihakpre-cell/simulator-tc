import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import unicodedata
import tempfile
from fpdf import FPDF

# --- POMOCNÉ FUNKCE ---
def remove_accents(input_str):
    if not input_str: return ""
    nfkd_form = unicodedata.normalize('NFKD', str(input_str))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# --- STREAMLIT UI & LOGIKA ---
st.set_page_config(page_title="Expertní simulátor TČ", layout="wide")
st.title("📊 Profesionální analýza kaskády TČ")

with st.sidebar:
    st.header("⚙️ Vstupní parametry")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sladkovičova")
    ztrata_celkova = st.number_input("Tepelná ztráta [kW]", value=54.0)
    t_design = st.number_input("Návrhová venkovní teplota [°C]", value=-12.0)
    fakt_ut = st.number_input("Reálná spotřeba ÚT [MWh/rok]", value=124.0)
    f_tuv = st.number_input("Reálná spotřeba TUV [MWh/rok]", value=76.0)
    t_privod = st.number_input("Teplota přívod [°C]", value=60)
    t_zpatecka = st.number_input("Teplota zpátečka [°C]", value=50)
    spad_text = f"{int(t_privod)} / {int(t_zpatecka)} °C"
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 3)
    cena_el_mwh = st.number_input("Cena elektřiny [Kč/MWh]", value=4800.0)
    cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284.0)
    investice = st.number_input("Investice [Kč]", value=3800000.0)
    servis = st.number_input("Roční servis [Kč]", value=17000.0)

tmy_up = st.file_uploader("TMY (CSV)", type="csv")
char_up = st.file_uploader("Charakteristika TČ (CSV)", type="csv")

if tmy_up and char_up:
    # (Předpokládáme load_tmy_robust a load_char z předchozích kroků - funkční)
    # Zkráceno pro přehlednost, v ostrém kódu ponechat načítací funkce
    tmy = pd.read_csv(tmy_up, skiprows=17) # Zjednodušený loading pro ukázku fixace
    tmy.columns = tmy.columns.str.strip()
    char = pd.read_csv(char_up) 

    if 'T2m' in tmy.columns:
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
        q_tuv_avg = (f_tuv / 8760) * 1000
        
        # Kalibrace
        potreba_ut_teorie = [ztrata_celkova * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
        k_oprava = fakt_ut / (sum(potreba_ut_teorie) / 1000)
        naklady_czt_rok = (fakt_ut + f_tuv) * (cena_gj_czt * 3.6)

        # Bod bivalence
        t_biv = -12.0 # Default
        for t in np.linspace(15, -15, 500):
            q_need = (ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg
            p_tc = np.interp(t, char['Teplota'], char['Vykon_kW']) * pocet_tc
            if p_tc < q_need:
                t_biv = t
                break

        # Simulace
        res = []
        for t_out, t_smooth in zip(tmy['T2m'], tmy['T_smooth']):
            q_total = max(0, (ztrata_celkova * (20 - t_smooth) / (20 - t_design) * k_oprava)) + q_tuv_avg
            p_max = np.interp(t_out, char['Teplota'], char['Vykon_kW']) * pocet_tc
            cop = np.interp(t_out, char['Teplota'], char['COP'])
            q_tc = min(q_total, p_max)
            q_biv = max(0, q_total - q_tc)
            res.append([round(t_out), q_total, q_tc, q_biv, q_tc/cop if q_tc > 0 else 0, q_biv/0.98])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
        
        # Agregace pro koláčový graf
        total_tc_mwh = df_sim['Q_tc'].sum() / 1000
        total_biv_mwh = df_sim['Q_biv'].sum() / 1000
        total_energy = total_tc_mwh + total_biv_mwh
        perc_tc = (total_tc_mwh / total_energy) * 100
        perc_biv = (total_biv_mwh / total_energy) * 100

        naklady_tc = ((df_sim['El_tc'].sum() + df_sim['El_biv'].sum()) / 1000) * cena_el_mwh + servis
        uspora = naklady_czt_rok - naklady_tc

        # --- GRAFICKÝ VÝSTUP ---
        fig = plt.figure(figsize=(16, 12))
        plt.suptitle(f"EXPERTNÍ ANALÝZA: {nazev_projektu}", fontsize=18, fontweight='bold')

        # 1. GRAF (ZAFIXOVÁN) - Dynamika
        ax1 = plt.subplot(2, 2, 1)
        t_range = np.linspace(-15, 18, 100)
        q_plot = [(ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg for t in t_range]
        p_plot = [np.interp(t, char['Teplota'], char['Vykon_kW']) * pocet_tc for t in t_range]
        ax1.plot(t_range, q_plot, color='red', lw=1.5, label='Potřeba domu')
        ax1.plot(t_range, p_plot, color='blue', lw=1, ls='--', alpha=0.3, label='Max limit')
        ax1.plot(t_range, [min(q,p) for q,p in zip(q_plot, p_plot)], color='green', lw=5, alpha=0.5, label='Modulace TČ')
        ax1.axvline(t_biv, color='black', ls=':', label=f'Bod biv. {t_biv:.1f}°C')
        ax1.set_title("DYNAMIKA PROVOZU A MODULACE", fontweight='bold')
        ax1.legend(loc='lower right', fontsize=8); ax1.grid(alpha=0.2)

        # 2. GRAF (ZAFIXOVÁN) - Sloupcový mix
        ax2 = plt.subplot(2, 2, 2)
        df_temp = df_sim.groupby('Temp')[['Q_tc', 'Q_biv']].sum().sort_index()
        ax2.bar(df_temp.index, df_temp['Q_tc'], color='#3498db', label='Energie TČ')
        ax2.bar(df_temp.index, df_temp['Q_biv'], bottom=df_temp['Q_tc'], color='#e74c3c', label='Bivalence')
        ax2.set_title("DODANÁ ENERGIE DLE TEPLOT", fontweight='bold')
        ax2.legend(fontsize=8); ax2.grid(alpha=0.1, axis='y')

        # 3. GRAF (NOVÝ) - Podíl bivalence (Donut chart)
        ax3 = plt.subplot(2, 2, 3)
        wedges, texts, autotexts = ax3.pie([total_tc_mwh, total_biv_mwh], 
                                          labels=['Tepelná čerpadla', 'Bivalentní zdroj'],
                                          autopct='%1.1f%%', startangle=90, 
                                          colors=['#3498db', '#e74c3c'],
                                          wedgeprops=dict(width=0.4, edgecolor='w'))
        plt.setp(autotexts, size=10, weight="bold", color="white")
        ax3.set_title("ROČNÍ ENERGETICKÉ KRYTÍ", fontweight='bold')
        # Středový text
        ax3.text(0, 0, f"{int(total_energy)} MWh\ncelkem", ha='center', va='center', fontweight='bold')

        # 4. GRAF (ZAFIXOVÁN) - Tabulka
        ax4 = plt.subplot(2, 2, 4); ax4.axis('off')
        summary = [["Projekt", nazev_projektu], ["Bod bivalence", f"{t_biv:.1f} °C"], 
                   ["Podíl TČ", f"{perc_tc:.1f} %"], ["Podíl Biv.", f"{perc_biv:.1f} %"],
                   ["Úspora", f"{uspora:,.0f} Kč/rok"]]
        tbl = ax4.table(cellText=summary, loc='center', cellLoc='left', colWidths=[0.4, 0.6])
        tbl.scale(1, 3.2); tbl.set_fontsize(11)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)
