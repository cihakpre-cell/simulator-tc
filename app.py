import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- 1. KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="Energetický Simulátor TČ", layout="wide")
st.title("🚀 Profesionální simulátor kaskády TČ")

# --- 2. SIDEBAR ---
st.sidebar.header("⚙️ Systémové parametry")
with st.sidebar:
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
    t_design = st.number_input("Návrhová venkovní teplota [°C]", value=-12.0)
    st.markdown("---")
    t_voda_max = st.number_input("Teplota vody při návrhové t. [°C]", value=60.0)
    t_voda_min = st.number_input("Teplota vody při +15°C [°C]", value=35.0)
    st.markdown("---")
    spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
    spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 3)
    st.markdown("---")
    cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800.0)
    cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284.0)
    investice = st.number_input("Investice celkem [Kč]", value=3800000.0)

# --- 3. POMOCNÉ FUNKCE ---
def load_tmy_robust(file):
    try:
        content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
        header_idx = -1
        for i, line in enumerate(content):
            if 'T2m' in line: header_idx = i; break
        if header_idx == -1: return None
        return pd.read_csv(io.StringIO("\n".join(content[header_idx:])))
    except: return None

def load_char(file):
    try:
        content = file.getvalue().decode('utf-8-sig', errors='ignore')
        sep = ';' if ';' in content.split('\n')[0] else ','
        df = pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
        df.columns = df.columns.str.strip()
        return df[['Teplota', 'Vykon_kW', 'COP']].copy()
    except: return None

# --- 4. NAHRÁNÍ DAT ---
st.subheader("📁 1. Krok: Nahrání dat")
col_f1, col_f2 = st.columns(2)
with col_f1: tmy_file = st.file_uploader("Nahrajte TMY", type="csv")
with col_f2: char_file = st.file_uploader("Nahrajte Charakteristiku (vstupy_TC.csv)", type="csv")

if tmy_file and char_file:
    tmy_raw = load_tmy_robust(tmy_file)
    df_char_raw = load_char(char_file)

    if tmy_raw is not None and df_char_raw is not None:
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Charakteristika TČ (editovatelná)")
        df_char = st.sidebar.data_editor(df_char_raw, num_rows="dynamic", hide_index=True, key="tc_editor")

        # Příprava TMY
        tmy = tmy_raw.copy()
        tmy.columns = tmy.columns.str.strip()
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
        tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()

        # Výpočet
        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        potreba_ut_teorie = [ztrata * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
        k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000)

        res = []
        for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
            t_voda_req = np.interp(t_sm, [t_design, 15], [t_voda_max, t_voda_min]) if t_sm < 20 else t_voda_min
            delta_t = max(0, t_voda_req - 35.0)
            k_cop, k_p = 1 - (delta_t * 0.025), 1 - (delta_t * 0.01)

            q_need = max(0, (ztrata * (20 - t_sm) / (20 - t_design) * k_oprava)) + q_tuv_avg
            p_base = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc
            cop_base = np.interp(t_out, df_char['Teplota'], df_char['COP'])
            
            p_real = p_base * k_p
            cop_real = cop_base * k_cop
            
            q_tc = min(q_need, p_real)
            q_biv = max(0, q_need - q_tc)
            res.append([t_out, q_need, q_tc, q_biv, q_tc/cop_real if q_tc > 0 else 0, q_biv/0.98])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])

        # --- BILANCE ---
        q_tc_s, q_biv_s = df_sim['Q_tc'].sum()/1000, df_sim['Q_biv'].sum()/1000
        el_tc_s, el_biv_s = df_sim['El_tc'].sum()/1000, df_sim['El_biv'].sum()/1000
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        naklady_tc = (el_tc_s + el_biv_s) * cena_el + 17000
        uspora = naklady_czt - naklady_tc
        navratnost = investice / uspora if uspora > 0 else 0

        # --- ZOBRAZENÍ ---
        tab1, tab2, tab3 = st.tabs(["💰 Ekonomika", "📈 Bilance a Grafy", "📅 Měsíční přehled"])
        
        with tab1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Roční úspora", f"{uspora:,.0f} Kč")
            c2.metric("Návratnost", f"{navratnost:.1f} let")
            c3.metric("SCOP systému", f"{q_tc_s / el_tc_s:.2f}")
            c4.metric("Bod bivalence", f"{df_sim[df_sim['Q_biv'] > 0.1]['Temp'].max() if q_biv_s > 0 else -20:.1f} °C")
            
            st.markdown("---")
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("Podíl spotřebované elektřiny")
                fig_pie, ax_pie = plt.subplots(figsize=(6,4))
                ax_pie.pie([el_tc_s, el_biv_s], labels=['TČ (Kompresor)', 'Bivalence (Patrona)'], 
                           autopct='%1.1f%%', colors=['#3498db','#e74c3c'], startangle=90)
                st.pyplot(fig_pie)
                st.caption("Pozn: Bivalence spotřebuje více elektřiny na 1 kWh tepla než TČ (COP 1 vs 3.5).")

        with tab2:
            st.subheader("Výkonová rovnováha kaskády")
            # GENERUJEME ČISTÝ GRAF DLE TEPLOTY (NE HODIN)
            tx = np.linspace(-15, 20, 100)
            qy = [max(0, (ztrata * (20 - t) / (20 - t_design) * k_oprava)) + q_tuv_avg for t in tx]
            
            # Korekce výkonu pro graf (přibližná ekvitermní korekce pro vizualizaci)
            py = []
            for t in tx:
                tv = np.interp(t, [t_design, 15], [t_voda_max, t_voda_min]) if t < 20 else t_voda_min
                kp = 1 - (max(0, tv - 35.0) * 0.01)
                p_base = np.interp(t, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc
                py.append(p_base * kp)

            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(tx, qy, 'r-', label='Potřeba domu + TUV [kW]', linewidth=2)
            ax1.plot(tx, py, 'b--', label=f'Maximální výkon kaskády ({pocet_tc}ks TČ) [kW]', linewidth=2)
            
            # Vybarvení bivalence (tam kde je potřeba > výkon)
            ax1.fill_between(tx, py, qy, where=(np.array(qy) > np.array(py)), color='red', alpha=0.2, label='Oblast bivalence')
            
            ax1.set_xlabel("Venkovní teplota [°C]")
            ax1.set_ylabel("Výkon [kW]")
            ax1.set_ylim(0, max(qy)*1.2)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            st.pyplot(fig1)

        with tab3:
            df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1
            df_sim['Month'] = df_sim['Month'].clip(1, 12)
            mes_df = df_sim.groupby('Month').agg({'Q_tc': 'sum', 'Q_biv': 'sum', 'El_tc': 'sum', 'El_biv': 'sum'}).reset_index()
            for c in ['Q_tc', 'Q_biv', 'El_tc', 'El_biv']: mes_df[c] /= 1000
            st.dataframe(mes_df.style.format(precision=2), use_container_width=True)

        st.download_button("📥 Stáhnout Excel", io.BytesIO().getvalue(), "simulace.xlsx")
else:
    st.info("Nahrajte soubory TMY a vstupy_TC.csv.")
