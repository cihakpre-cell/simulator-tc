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
st.set_page_config(page_title="Expertní simulátor TČ v2.0", layout="wide")

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
        eta_biv = st.slider("Účinnost bivalence [%]", 80, 100, 98) / 100

    with st.expander("💰 Ekonomika", expanded=True):
        investice = st.number_input("Investice celkem [Kč]", value=3800000)
        cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800)
        cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284)

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
            if (np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc) < (ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava + q_tuv_avg):
                t_biv = t
                break

        # --- VÝSTUPNÍ GRAFY ---
        st.header(f"📈 Analýza: {nazev_projektu}")
        
        # Horní řada (1 a 2)
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # 1. Dynamika (Modulace)
        tr = np.linspace(-15, 18, 100)
        q_p = [(ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava) + q_tuv_avg for t in tr]
        p_p = [np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tr]
        ax1.plot(tr, q_p, 'r-', lw=2, label='Potřeba (ÚT+TUV)')
        ax1.plot(tr, p_p, 'b--', alpha=0.4, label='Max výkon kaskády')
        ax1.plot(tr, [min(q,p) for q,p in zip(q_p, p_p)], 'g-', lw=5, alpha=0.4, label='Skutečné krytí TČ')
        ax1.axvline(t_biv, color='k', ls=':', label=f'Bivalence {t_biv:.1f}°C')
        ax1.set_title("1. DYNAMIKA PROVOZU A MODULACE", fontweight='bold')
        ax1.set_xlabel("Venkovní teplota [°C]"); ax1.set_ylabel("Výkon [kW]"); ax1.legend(); ax1.grid(alpha=0.2)

        # 2. Mix dle teploty
        df_sim['Temp_R'] = df_sim['Temp'].round()
        df_t = df_sim.groupby('Temp_R')[['Q_tc', 'Q_biv']].sum().sort_index()
        ax2.bar(df_t.index, df_t['Q_tc'], color='#3498db', label='Energie TČ')
        ax2.bar(df_t.index, df_t['Q_biv'], bottom=df_t['Q_tc'], color='#e74c3c', label='Bivalence')
        ax2.set_title("2. ENERGETICKÝ MIX DLE VENKOVNÍ TEPLOTY", fontweight='bold')
        ax2.set_xlabel("Teplota [°C]"); ax2.set_ylabel("Energie [kWh]"); ax2.legend(); ax2.grid(alpha=0.1, axis='y')
        st.pyplot(fig1)

        # Prostřední řada (3 a 4)
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 6))
        
        # 3. Měsíční bilance
        df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1
        m_df = df_sim.groupby('Month').agg({'Q_tc': 'sum', 'Q_biv': 'sum'}).reset_index()
        ax3.bar(m_df['Month'], m_df['Q_tc']/1000, label='TČ', color='#3498db')
        ax3.bar(m_df['Month'], m_df['Q_biv']/1000, bottom=m_df['Q_tc']/1000, label='Bivalence', color='#e74c3c')
        ax3.set_title("3. MĚSÍČNÍ BILANCE ENERGIE [MWh]", fontweight='bold')
        ax3.set_xticks(range(1, 13)); ax3.set_ylabel("MWh"); ax3.legend(); ax3.grid(alpha=0.1, axis='y')

        # 4. Výkonová monotóna (dle výkonu)
        q_sorted = np.sort(df_sim['Q_need'].values)[::-1]
        p_lim = np.interp(t_biv, df_char[t_col], df_char[v_col]) * pocet_tc
        ax4.plot(range(8760), q_sorted, 'r-', lw=2, label='Potřeba')
        ax4.fill_between(range(8760), p_lim, q_sorted, where=(q_sorted > p_lim), color='#e74c3c', alpha=0.4, label='Bivalence')
        ax4.fill_between(range(8760), 0, np.minimum(q_sorted, p_lim), color='#3498db', alpha=0.2, label='TČ')
        ax4.set_title("4. TRVÁNÍ POTŘEBY VÝKONU (DLE VÝKONU)", fontweight='bold')
        ax4.set_xlabel("Hodin v roce"); ax4.set_ylabel("Výkon [kW]"); ax4.legend(); ax4.grid(alpha=0.2)
        st.pyplot(fig2)

        # Spodní graf (5 - Nový dle vašeho kódu)
        st.markdown("---")
        st.subheader("5. Četnost teplot a bod bivalence v roce (Seřazeno dle teploty)")
        df_sorted_t = df_sim.sort_values('Temp').reset_index(drop=True)
        fig3, ax5 = plt.subplots(figsize=(18, 5))
        ax5.plot(df_sorted_t.index, df_sorted_t['Q_need'], 'r', label='Potřeba domu (ÚT+TUV)')
        ax5.plot(df_sorted_t.index, df_sorted_t['Q_tc'], 'b', label='Krytí TČ')
        biv_area = df_sorted_t[df_sorted_t['Q_biv'] > 0.1].index
        if len(biv_area) > 0:
            ax5.fill_between(df_sorted_t.index[:max(biv_area)], df_sorted_t['Q_tc'][:max(biv_area)], 
                             df_sorted_t['Q_need'][:max(biv_area)], color='red', alpha=0.3, label='Oblast bivalence')
        ax5.set_ylabel("Výkon [kW]"); ax5.set_xlabel("Hodin v roce (od nejnižší po nejvyšší teplotu)")
        ax5.legend(); ax5.grid(True, alpha=0.2)
        st.pyplot(fig3)
