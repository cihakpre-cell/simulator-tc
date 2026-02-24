import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- POMOCNÉ FUNKCE ---
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

# --- KONFIGURACE ---
st.set_page_config(page_title="Expertní simulátor TČ", layout="wide")

# --- SIDEBAR - KOMPLETNÍ VSTUPY ---
with st.sidebar:
    st.header("⚙️ Konfigurace projektu")
    nazev_projektu = st.text_input("Název projektu", "Analýza Sladkovičova")
    
    with st.expander("🏠 Budova a potřeba", expanded=True):
        ztrata_celkova = st.number_input("Tepelná ztráta [kW]", value=54.0)
        t_vnitrni = st.number_input("Žádaná vnitřní teplota [°C]", value=20.0)
        t_design = st.number_input("Návrhová venkovní teplota [°C]", value=-12.0)
        fakt_ut = st.number_input("Reálná spotřeba ÚT [MWh/rok]", value=124.0)
        f_tuv = st.number_input("Reálná spotřeba TUV [MWh/rok]", value=76.0)

    with st.expander("🔧 Technologie TČ", expanded=True):
        pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 3)
        t_tuv_zadana = st.number_input("Teplota TUV [°C]", value=55)
        t_privod = st.number_input("Teplota přívod ÚT [°C]", value=60)
        t_zpatecka = st.number_input("Teplota zpátečka ÚT [°C]", value=50)
        eta_biv = st.slider("Účinnost bivalence [%]", 80, 100, 98) / 100

    with st.expander("💰 Ekonomika a CAPEX", expanded=True):
        investice = st.number_input("Investice celková (CAPEX) [Kč]", value=3800000)
        dotace = st.number_input("Dotace [Kč]", value=0)
        cena_el_mwh = st.number_input("Cena elektřiny [Kč/MWh]", value=4800.0)
        cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284.0)
        servis = st.number_input("Roční servis [Kč]", value=17000.0)

# --- VÝPOČETNÍ JÁDRO ---
tmy_up = st.file_uploader("Nahrajte TMY (CSV)", type="csv")
char_up = st.file_uploader("Nahrajte Charakteristiku TČ (CSV)", type="csv")

if tmy_up and char_up:
    tmy = load_tmy_robust(tmy_up)
    char = load_char(char_up)

    if tmy is not None and char is not None:
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
        q_tuv_avg = (f_tuv / 8760) * 1000
        
        # Korekce na reálnou spotřebu
        potreba_ut_teorie = [ztrata_celkova * (t_vnitrni - t) / (t_vnitrni - t_design) if t < t_vnitrni else 0 for t in tmy['T_smooth']]
        k_oprava = fakt_ut / (sum(potreba_ut_teorie) / 1000)

        res = []
        for t_out, t_smooth in zip(tmy['T2m'], tmy['T_smooth']):
            q_total = max(0, (ztrata_celkova * (t_vnitrni - t_smooth) / (t_vnitrni - t_design) * k_oprava)) + q_tuv_avg
            p_max = np.interp(t_out, char['Teplota'], char['Vykon_kW']) * pocet_tc
            cop = np.interp(t_out, char['Teplota'], char['COP'])
            
            q_tc = min(q_total, p_max)
            q_biv = max(0, q_total - q_tc)
            res.append([t_out, q_total, q_tc, q_biv, q_tc/cop if q_tc > 0 else 0, q_biv/eta_biv])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
        
        # Bod bivalence
        t_biv = -12.0
        for t in np.linspace(15, -15, 500):
            q_req = (ztrata_celkova * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava) + q_tuv_avg
            if (np.interp(t, char['Teplota'], char['Vykon_kW']) * pocet_tc) < q_req:
                t_biv = t
                break

        # --- GRAFICKÁ ČÁST ---
        fig = plt.figure(figsize=(18, 14))
        
        # 1. DYNAMIKA (Zafixováno)
        ax1 = plt.subplot(2, 2, 1)
        tr = np.linspace(-15, 18, 100)
        q_p = [(ztrata_celkova * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava) + q_tuv_avg for t in tr]
        p_p = [np.interp(t, char['Teplota'], char['Vykon_kW']) * pocet_tc for t in tr]
        ax1.plot(tr, q_p, 'r-', lw=1.5, label='Potřeba domu')
        ax1.plot(tr, p_p, 'b--', lw=1, alpha=0.3, label='Max kaskáda')
        ax1.plot(tr, [min(q,p) for q,p in zip(q_p, p_p)], 'g-', lw=5, alpha=0.5, label='Výkon TČ')
        t_m = np.linspace(-15, t_biv, 50)
        ax1.fill_between(t_m, np.interp(t_m, char['Teplota'], char['Vykon_kW'])*pocet_tc, 
                         [(ztrata_celkova*(t_vnitrni-tx)/(t_vnitrni-t_design)*k_oprava)+q_tuv_avg for tx in t_m], 
                         color='red', alpha=0.2, hatch='\\\\\\')
        ax1.axvline(t_biv, color='k', ls=':', label=f'Bivalence {t_biv:.1f}°C')
        ax1.set_title("DYNAMIKA PROVOZU A MODULACE", fontweight='bold')
        ax1.legend(loc='lower right', fontsize=8); ax1.grid(alpha=0.2)

        # 2. SLOUPCOVÝ MIX (Zafixováno)
        ax2 = plt.subplot(2, 2, 2)
        df_sim['Temp_R'] = df_sim['Temp'].round()
        df_t = df_sim.groupby('Temp_R')[['Q_tc', 'Q_biv']].sum().sort_index()
        ax2.bar(df_t.index, df_t['Q_tc'], color='#3498db', label='Energie TČ')
        ax2.bar(df_t.index, df_t['Q_biv'], bottom=df_t['Q_tc'], color='#e74c3c', label='Bivalence')
        ax2.set_title("ROZDĚLENÍ ENERGIE DLE VENKOVNÍ TEPLOTY", fontweight='bold')
        ax2.legend(fontsize=8); ax2.grid(alpha=0.1, axis='y')

        # 3. VÝSEČOVÝ + TABULKA (Zafixováno, bez stínů, oprava přetékání)
        ax3 = plt.subplot(2, 2, 3)
        q_tc_s, q_bv_s = df_sim['Q_tc'].sum()/1000, df_sim['Q_biv'].sum()/1000
        el_tc_s, el_bv_s = df_sim['El_tc'].sum()/1000, df_sim['El_biv'].sum()/1000
        total_q = q_tc_s + q_bv_s
        total_el = el_tc_s + el_bv_s
        ax3.pie([q_tc_s, q_bv_s], labels=['TČ', 'Biv.'], autopct='%1.1f%%', startangle=90, 
                colors=['#3498db', '#e74c3c'], explode=(0, 0.1), shadow=False)
        ax3.set_title("PODÍL NA DODANÉM TEPLE", fontweight='bold')
        
        table_data = [
            ["Zdroj", "Teplo [MWh]", "Teplo [%]", "El. [MWh]", "El. [%]"],
            ["Tepelná kaskáda", f"{q_tc_s:.1f}", f"{(q_tc_s/total_q)*100:.1f}%", f"{el_tc_s:.1f}", f"{(el_tc_s/total_el)*100:.1f}%"],
            ["Biv. zdroj", f"{q_bv_s:.1f}", f"{(q_bv_s/total_q)*100:.1f}%", f"{el_bv_s:.1f}", f"{(el_bv_s/total_el)*100:.1f}%"],
            ["CELKEM", f"{total_q:.1f}", "100%", f"{total_el:.1f}", "100%"]
        ]
        tbl = ax3.table(cellText=table_data, loc='bottom', cellLoc='center', bbox=[0, -0.5, 1, 0.35])
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)
        for i in range(5): tbl[(0, i)].set_facecolor("#f2f2f2")

        # 4. VÝKONOVÁ MONOTÓNA (Zafixováno, osa Y v kW)
        ax4 = plt.subplot(2, 2, 4)
        q_sorted = np.sort(df_sim['Q_need'].values)[::-1]
        hours = np.arange(len(q_sorted))
        p_lim = np.interp(t_biv, char['Teplota'], char['Vykon_kW']) * pocet_tc
        ax4.plot(hours, q_sorted, 'r-', lw=2, label='Potřebný výkon')
        ax4.fill_between(hours, p_lim, q_sorted, where=(q_sorted > p_lim), color='#e74c3c', alpha=0.4, label='Bivalence')
        ax4.fill_between(hours, 0, np.minimum(q_sorted, p_lim), color='#3498db', alpha=0.3, label='Kryto TČ')
        ax4.set_title("TRVÁNÍ POTŘEBY VÝKONU (MONOTONA)", fontweight='bold')
        ax4.set_xlabel("Hodin v roce"); ax4.set_ylabel("Výkon [kW]")
        ax4.set_xlim(0, 8760); ax4.grid(alpha=0.2); ax4.legend(loc='upper right', fontsize=8)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        st.pyplot(fig)
        
        # --- DOPLŇKOVÁ EKONOMIKA ---
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        naklady_czt = (fakt_ut + f_tuv) * cena_gj_czt * 3.6
        naklady_tc = total_el * cena_el_mwh + servis
        uspora = naklady_czt - naklady_tc
        prostah_navratnost = (investice - dotace) / uspora if uspora > 0 else 0
        
        c1.metric("Roční úspora", f"{uspora:,.0f} Kč")
        c2.metric("Provozní SCOP", f"{total_q/total_el:.2f}")
        c3.metric("Návratnost (prostá)", f"{prostah_navratnost:.1f} let")
