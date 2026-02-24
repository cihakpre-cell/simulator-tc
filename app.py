import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- POMOCNÉ FUNKCE ---
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
st.set_page_config(page_title="Expertní simulátor TČ", layout="wide")

with st.sidebar:
    st.header("⚙️ Konfigurace projektu")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    
    with st.expander("🏠 Budova a potřeba", expanded=True):
        ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
        t_vnitrni = st.number_input("Žádaná vnitřní teplota [°C]", value=20.0)
        t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
        spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
        spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)

    with st.expander("🔧 Technologie", expanded=True):
        pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 3)
        eta_biv = st.slider("Účinnost bivalence [%]", 80, 100, 98) / 100

# --- VÝPOČETNÍ JÁDRO ---
tmy_file = st.file_uploader("1. Nahrajte TMY (CSV)", type="csv")
char_file = st.file_uploader("2. Nahrajte Charakteristiku TČ (CSV)", type="csv")

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
        
        # Bod bivalence pro grafy
        t_biv = -12.0
        for t in np.linspace(15, -15, 500):
            q_req = (ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava) + q_tuv_avg
            if (np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc) < q_req:
                t_biv = t
                break

        # --- GRAFICKÝ VÝSTUP 2x2 ---
        fig = plt.figure(figsize=(18, 14))

        # 1. DYNAMIKA PROVOZU (Vlevo nahoře - FIXNÍ)
        ax1 = plt.subplot(2, 2, 1)
        tr = np.linspace(-15, 18, 100)
        q_p = [(ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava) + q_tuv_avg for t in tr]
        p_p = [np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tr]
        ax1.plot(tr, q_p, 'r-', lw=2, label='Potřeba domu (ÚT+TUV)')
        ax1.plot(tr, p_p, 'b--', lw=1, alpha=0.4, label='Max kaskáda')
        ax1.plot(tr, [min(q,p) for q,p in zip(q_p, p_p)], 'g-', lw=5, alpha=0.5, label='Výkon TČ')
        t_mraz = np.linspace(-15, t_biv, 50)
        q_mraz = [(ztrata * (t_vnitrni - tx) / (t_vnitrni - t_design) * k_oprava) + q_tuv_avg for tx in t_mraz]
        p_mraz = [np.interp(tx, df_char[t_col], df_char[v_col]) * pocet_tc for tx in t_mraz]
        ax1.fill_between(t_mraz, p_mraz, q_mraz, color='red', alpha=0.2, hatch='\\\\\\', label='Bivalence')
        ax1.axvline(t_biv, color='k', ls=':', label=f'Bivalence {t_biv:.1f}°C')
        ax1.set_title("DYNAMIKA PROVOZU A MODULACE", fontweight='bold')
        ax1.set_xlabel("Venkovní teplota [°C]"); ax1.set_ylabel("Výkon [kW]")
        ax1.legend(loc='lower right'); ax1.grid(alpha=0.2)

        # 2. ENERGETICKÝ MIX DLE TEPLOTY (Vpravo nahoře - FIXNÍ)
        ax2 = plt.subplot(2, 2, 2)
        df_sim['Temp_R'] = df_sim['Temp'].round()
        df_t = df_sim.groupby('Temp_R')[['Q_tc', 'Q_biv']].sum().sort_index()
        ax2.bar(df_t.index, df_t['Q_tc'], color='#3498db', label='Energie TČ')
        ax2.bar(df_t.index, df_t['Q_biv'], bottom=df_t['Q_tc'], color='#e74c3c', label='Bivalence')
        ax2.set_title("ROZDĚLENÍ ENERGIE DLE VENKOVNÍ TEPLOTY", fontweight='bold')
        ax2.set_xlabel("Venkovní teplota [°C]"); ax2.set_ylabel("Energie [kWh]"); ax2.legend(); ax2.grid(alpha=0.1, axis='y')

        # 3. MĚSÍČNÍ ENERGIE (Vlevo dole - FIXNÍ)
        ax3 = plt.subplot(2, 2, 3)
        df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1
        df_sim['Month'] = df_sim['Month'].clip(1, 12)
        m_df = df_sim.groupby('Month').agg({'Q_tc': 'sum', 'Q_biv': 'sum'}).reset_index()
        ax3.bar(m_df['Month'], m_df['Q_tc']/1000, label='TČ', color='#3498db')
        ax3.bar(m_df['Month'], m_df['Q_biv']/1000, bottom=m_df['Q_tc']/1000, label='Bivalence', color='#e74c3c')
        ax3.set_title("MĚSÍČNÍ BILANCE ENERGIE [MWh]", fontweight='bold')
        ax3.set_xlabel("Měsíc"); ax3.set_ylabel("MWh"); ax3.legend(); ax3.grid(alpha=0.1, axis='y')

        # 4. VÝKONOVÁ MONOTÓNA (Vpravo dole - FIXNÍ)
        ax4 = plt.subplot(2, 2, 4)
        q_sorted = np.sort(df_sim['Q_need'].values)[::-1]
        hours = np.arange(len(q_sorted))
        p_lim = np.interp(t_biv, df_char[t_col], df_char[v_col]) * pocet_tc
        ax4.plot(hours, q_sorted, 'r-', lw=2, label='Potřeba')
        ax4.fill_between(hours, p_lim, q_sorted, where=(q_sorted > p_lim), color='#e74c3c', alpha=0.4, label='Bivalence')
        ax4.fill_between(hours, 0, np.minimum(q_sorted, p_lim), color='#3498db', alpha=0.2, label='Kryto TČ')
        ax4.set_title("TRVÁNÍ POTŘEBY VÝKONU (MONOTONA)", fontweight='bold')
        ax4.set_xlabel("Hodin v roce"); ax4.set_ylabel("Výkon [kW]"); ax4.set_xlim(0, 8760); ax4.grid(alpha=0.2)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        st.pyplot(fig)
