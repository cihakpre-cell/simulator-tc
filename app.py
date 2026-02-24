import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- FUNKCE NAČÍTÁNÍ (Robustní) ---
def load_tmy_robust(file):
    try:
        content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
        start_idx = -1
        for i, line in enumerate(content):
            if 'time(UTC)' in line or 'T2m' in line:
                start_idx = i
                break
        if start_idx == -1: return None
        df = pd.read_csv(io.StringIO("\n".join(content[start_idx:])))
        df.columns = df.columns.str.strip()
        df['T2m'] = pd.to_numeric(df['T2m'], errors='coerce')
        return df.dropna(subset=['T2m']).reset_index(drop=True)
    except: return None

def load_char(file):
    try:
        content = file.getvalue().decode('utf-8-sig', errors='ignore')
        sep = ';' if ';' in content.split('\n')[0] else ','
        df = pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
        return df[['Teplota', 'Vykon_kW', 'COP']].apply(pd.to_numeric, errors='coerce').dropna()
    except: return None

# --- UI ---
st.set_page_config(page_title="Expertní simulátor TČ", layout="wide")

with st.sidebar:
    st.header("⚙️ Vstupní parametry")
    nazev_projektu = st.text_input("Název projektu", "Analýza kaskády")
    ztrata_celkova = st.number_input("Tepelná ztráta [kW]", value=54.0)
    t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
    fakt_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
    f_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 3)
    cena_el_mwh = st.number_input("Cena elektřiny [Kč/MWh]", value=4800.0)

tmy_up = st.file_uploader("Nahrajte TMY (CSV)", type="csv")
char_up = st.file_uploader("Nahrajte Charakteristiku (CSV)", type="csv")

if tmy_up and char_up:
    tmy = load_tmy_robust(tmy_up)
    char = load_char(char_up)

    if tmy is not None and char is not None:
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
        q_tuv_avg = (f_tuv / 8760) * 1000
        potreba_ut_teorie = [ztrata_celkova * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
        k_oprava = fakt_ut / (sum(potreba_ut_teorie) / 1000)

        # Simulace
        res = []
        for t_out, t_smooth in zip(tmy['T2m'], tmy['T_smooth']):
            q_total = max(0, (ztrata_celkova * (20 - t_smooth) / (20 - t_design) * k_oprava)) + q_tuv_avg
            p_max = np.interp(t_out, char['Teplota'], char['Vykon_kW']) * pocet_tc
            cop = np.interp(t_out, char['Teplota'], char['COP'])
            q_tc = min(q_total, p_max)
            q_biv = max(0, q_total - q_tc)
            res.append([t_out, q_total, q_tc, q_biv, q_tc/cop if q_tc > 0 else 0, q_biv/0.98])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
        
        # Bod bivalence
        t_biv = -12.0
        for t in np.linspace(15, -15, 500):
            if (np.interp(t, char['Teplota'], char['Vykon_kW']) * pocet_tc) < ((ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg):
                t_biv = t
                break

        # --- GRAFICKÝ VÝSTUP ---
        fig = plt.figure(figsize=(18, 14))
        
        # 1. DYNAMIKA PROVOZU (FIXNÍ)
        ax1 = plt.subplot(2, 2, 1)
        tr = np.linspace(-15, 18, 100)
        q_p = [(ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg for t in tr]
        p_p = [np.interp(t, char['Teplota'], char['Vykon_kW']) * pocet_tc for t in tr]
        ax1.plot(tr, q_p, color='red', lw=1.5, label='Potřeba domu')
        ax1.plot(tr, p_p, color='blue', lw=1, ls='--', alpha=0.3, label='Max limit kaskády')
        ax1.plot(tr, [min(q,p) for q,p in zip(q_p, p_p)], color='green', lw=5, alpha=0.5, label='Skutečný výkon TČ')
        t_mraz = np.linspace(-15, t_biv, 50)
        q_mraz = [(ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg for t in t_mraz]
        p_mraz = [np.interp(t, char['Teplota'], char['Vykon_kW']) * pocet_tc for t in t_mraz]
        ax1.fill_between(t_mraz, p_mraz, q_mraz, color='red', alpha=0.2, hatch='\\\\\\', label='Bivalentní dohřev')
        ax1.axvline(t_biv, color='black', ls=':', label=f'Bod bivalence {t_biv:.1f}°C')
        ax1.set_title("DYNAMIKA PROVOZU A MODULACE", fontweight='bold')
        ax1.legend(loc='lower right', fontsize=8); ax1.grid(alpha=0.2)

        # 2. ENERGETICKÝ MIX (FIXNÍ)
        ax2 = plt.subplot(2, 2, 2)
        df_sim['Temp_R'] = df_sim['Temp'].round()
        df_t = df_sim.groupby('Temp_R')[['Q_tc', 'Q_biv']].sum().sort_index()
        ax2.bar(df_t.index, df_t['Q_tc'], color='#3498db', label='Energie TČ')
        ax2.bar(df_t.index, df_t['Q_biv'], bottom=df_t['Q_tc'], color='#e74c3c', label='Bivalence')
        ax2.set_title("ROZDĚLENÍ ENERGIE DLE VENKOVNÍ TEPLOTY", fontweight='bold')
        ax2.legend(fontsize=8); ax2.grid(alpha=0.1, axis='y')

        # 3. VÝSEČOVÝ GRAF + BILANCE (ZAFIXOVÁNO)
        ax3 = plt.subplot(2, 2, 3)
        q_tc_s, q_bv_s = df_sim['Q_tc'].sum()/1000, df_sim['Q_biv'].sum()/1000
        el_tc_s, el_bv_s = df_sim['El_tc'].sum()/1000, df_sim['El_biv'].sum()/1000
        total_q, total_el = q_tc_s + q_bv_s, el_tc_s + el_bv_s
        ax3.pie([q_tc_s, q_bv_s], labels=['TČ', 'Biv.'], autopct='%1.1f%%', startangle=90, 
                colors=['#3498db', '#e74c3c'], explode=(0, 0.1), shadow=True)
        ax3.set_title("PODÍL NA DODANÉM TEPLE", fontweight='bold')
        
        table_data = [
            ["Zdroj", "Teplo [MWh]", "Teplo [%]", "El. [MWh]", "El. [%]"],
            ["Tepelná čerpadla", f"{q_tc_s:.1f}", f"{(q_tc_s/total_q)*100:.1f}%", f"{el_tc_s:.1f}", f"{(el_tc_s/total_el)*100:.1f}%"],
            ["Bivalentní zdroj", f"{q_bv_s:.1f}", f"{(q_bv_s/total_q)*100:.1f}%", f"{el_bv_s:.1f}", f"{(el_bv_s/total_el)*100:.1f}%"],
            ["CELKEM", f"{total_q:.1f}", "100%", f"{total_el:.1f}", "100%"]
        ]
        tbl = ax3.table(cellText=table_data, loc='bottom', cellLoc='center', bbox=[0, -0.45, 1, 0.35])
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)
        for i in range(5): tbl[(0, i)].set_facecolor("#f2f2f2")

        # 4. ČETNOST TEPLOT V ROCE (FIXNÍ - OSA X = HODINY)
        ax4 = plt.subplot(2, 2, 4)
        temps_sorted = np.sort(df_sim['Temp'].values) # Seřazení teplot od nejnižší po nejvyšší
        hours = np.arange(len(temps_sorted))
        
        # Vybarvení oblastí
        ax4.fill_between(hours, temps_sorted, t_biv, where=(temps_sorted < t_biv), color='#e74c3c', alpha=0.3, label='Bivalentní provoz')
        ax4.fill_between(hours, temps_sorted, t_biv, where=(temps_sorted >= t_biv), color='#3498db', alpha=0.3, label='Monovalentní provoz')
        ax4.plot(hours, temps_sorted, color='black', lw=1.5)
        
        ax4.axhline(t_biv, color='black', ls='--', lw=2, label=f'Bod bivalence {t_biv:.1f}°C')
        ax4.set_title("ČETNOST TEPLOT V ROCE", fontweight='bold')
        ax4.set_xlabel("Hodin v roce"); ax4.set_ylabel("Venkovní teplota [°C]")
        ax4.set_xlim(0, 8760); ax4.grid(alpha=0.2); ax4.legend(loc='upper left', fontsize=8)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        st.pyplot(fig)
