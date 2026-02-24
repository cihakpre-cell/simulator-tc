import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- FUNKCE PRO NAČÍTÁNÍ ---
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

    with st.expander("🔧 Technologie a Teploty", expanded=True):
        pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 3)
        t_tuv_vystup = st.number_input("Výstupní teplota TUV [°C]", value=55)
        t_spad_ut = st.text_input("Teplotní spád ÚT", value="60/50")
        eta_biv = st.slider("Účinnost bivalence [%]", 80, 100, 98) / 100

    with st.expander("💰 Ekonomika", expanded=True):
        investice = st.number_input("Investice celkem [Kč]", value=3800000)
        dotace = st.number_input("Dotace [Kč]", value=0)
        cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800)
        cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284)

# --- VÝPOČETNÍ JÁDRO ---
tmy_file = st.file_uploader("1. Nahrajte TMY (soubor tmy_...)", type="csv")
char_file = st.file_uploader("2. Nahrajte Charakteristiku TČ (vstupy_TC.csv)", type="csv")

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
        
        # Měsíční agregace pro Graf 3
        df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1
        df_sim['Month'] = df_sim['Month'].clip(1, 12)
        mesicni_df = df_sim.groupby('Month').agg({'Q_tc': 'sum', 'Q_biv': 'sum'}).reset_index()
        mesicni_df[['Q_tc', 'Q_biv']] /= 1000 # na MWh

        # --- GRAFICKÁ ČÁST (2x2) ---
        fig = plt.figure(figsize=(18, 14))

        # 1. Četnost teplot a bod bivalence (Z tvého kódu)
        ax1 = plt.subplot(2, 2, 1)
        df_sorted = df_sim.sort_values('Temp').reset_index(drop=True)
        biv_idx = df_sorted[df_sorted['Q_biv'] > 0.1].index
        ax1.plot(df_sorted.index, df_sorted['Q_need'], 'r', label='Potřeba domu (ÚT+TUV)')
        ax1.plot(df_sorted.index, df_sorted['Q_tc'], 'b', label='Krytí TČ')
        if len(biv_idx) > 0:
            ax1.fill_between(df_sorted.index[:max(biv_idx)], df_sorted['Q_tc'][:max(biv_idx)], 
                             df_sorted['Q_need'][:max(biv_idx)], color='red', alpha=0.3, label='Oblast bivalence')
        ax1.set_title("ČETNOST TEPLOT A BOD BIVALENCE", fontweight='bold')
        ax1.set_ylabel("Výkon [kW]"); ax1.set_xlabel("Hodin v roce (seřazeno od nejnižší teploty)")
        ax1.legend(); ax1.grid(alpha=0.2)

        # 2. Výkonová křivka vs Teplota (Zafixováno)
        ax2 = plt.subplot(2, 2, 2)
        tx = np.linspace(-15, 20, 100)
        qy = [ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava + q_tuv_avg for t in tx]
        py = [np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tx]
        ax2.plot(tx, qy, 'r', lw=2, label='Potřeba')
        ax2.plot(tx, py, 'b--', lw=2, label='Výkon kaskády')
        ax2.set_title("VÝKONOVÁ ROVNOVÁHA (kW vs °C)", fontweight='bold')
        ax2.set_xlabel("Teplota [°C]"); ax2.set_ylabel("Výkon [kW]"); ax2.legend(); ax2.grid(alpha=0.2)

        # 3. Měsíční energie v MWh (Z tvého kódu)
        ax3 = plt.subplot(2, 2, 3)
        ax3.bar(mesicni_df['Month'], mesicni_df['Q_tc'], label='TČ', color='#3498db')
        ax3.bar(mesicni_df['Month'], mesicni_df['Q_biv'], bottom=mesicni_df['Q_tc'], label='Bivalence', color='#e74c3c')
        ax3.set_title("MĚSÍČNÍ ENERGIE [MWh]", fontweight='bold')
        ax3.set_xlabel("Měsíc"); ax3.set_ylabel("MWh"); ax3.legend(); ax3.grid(alpha=0.1, axis='y')

        # 4. Podíl na dodaném teple - Pie (Zafixováno)
        ax4 = plt.subplot(2, 2, 4)
        q_tc_s, q_bv_s = df_sim['Q_tc'].sum()/1000, df_sim['Q_biv'].sum()/1000
        el_tc_s, el_bv_s = df_sim['El_tc'].sum()/1000, df_sim['El_biv'].sum()/1000
        total_q = q_tc_s + q_bv_s
        ax4.pie([q_tc_s, q_bv_s], labels=['TČ', 'Biv.'], autopct='%1.1f%%', startangle=90, 
                colors=['#3498db', '#e74c3c'], explode=(0, 0.1))
        ax4.set_title("PODÍL NA DODANÉM TEPLE", fontweight='bold')
        
        table_data = [
            ["Zdroj", "Teplo [MWh]", "Teplo [%]", "El. [MWh]"],
            ["Tepelná kaskáda", f"{q_tc_s:.1f}", f"{(q_tc_s/total_q)*100:.1f}%", f"{el_tc_s:.1f}"],
            ["Biv. zdroj", f"{q_bv_s:.1f}", f"{(q_bv_s/total_q)*100:.1f}%", f"{el_bv_s:.1f}"],
            ["CELKEM", f"{total_q:.1f}", "100%", f"{(el_tc_s+el_bv_s):.1f}"]
        ]
        tbl = ax4.table(cellText=table_data, loc='bottom', cellLoc='center', bbox=[0, -0.45, 1, 0.3])
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        st.pyplot(fig)
