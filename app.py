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

        # Příprava TMY a výpočet
        tmy = tmy_raw.copy()
        tmy.columns = tmy.columns.str.strip()
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce').round(0)
        tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()

        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        potreba_ut_teorie = [ztrata * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
        k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000)

        res = []
        for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
            t_voda_req = np.interp(t_sm, [t_design, 15], [t_voda_max, t_voda_min]) if t_sm < 20 else t_voda_min
            delta_t = max(0, t_voda_req - 35.0)
            k_p, k_cop = 1 - (delta_t * 0.01), 1 - (delta_t * 0.025)
            q_need = max(0, (ztrata * (20 - t_sm) / (20 - t_design) * k_oprava)) + q_tuv_avg
            p_real = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc * k_p
            cop_real = np.interp(t_out, df_char['Teplota'], df_char['COP']) * k_cop
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

        # --- ZOBRAZENÍ ---
        st.header(f"📊 Analýza systému: {nazev_projektu}")
        tab1, tab2 = st.tabs(["📉 Výkonová a Energetická bilance", "💰 Ekonomika a přínosy"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("1. Výkonová rovnováha (kW)")
                tx = np.linspace(df_sim['Temp'].min(), 20, 50)
                qy = [max(0, (ztrata * (20 - t) / (20 - t_design) * k_oprava)) + q_tuv_avg for t in tx]
                py = [np.interp(t, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc * (1-(max(0,np.interp(t,[t_design,15],[t_voda_max,t_voda_min])-35)*0.01)) for t in tx]
                fig1, ax1 = plt.subplots(figsize=(10,6))
                ax1.plot(tx, qy, 'r-', label='Potřeba objektu', linewidth=2)
                ax1.plot(tx, py, 'b--', label='Výkon kaskády TČ', linewidth=2)
                ax1.set_xlabel("Venkovní teplota [°C]"); ax1.set_ylabel("Výkon [kW]"); ax1.grid(True, alpha=0.3); ax1.legend()
                st.pyplot(fig1)

            with col_b:
                st.subheader("2. Energetické pokrytí (MWh)")
                df_binned = df_sim.groupby('Temp').agg({'Q_tc':'sum', 'Q_biv':'sum'}).reset_index()
                df_binned[['Q_tc', 'Q_biv']] /= 1000
                fig2, ax2 = plt.subplots(figsize=(10,6))
                ax2.bar(df_binned['Temp'], df_binned['Q_tc'], color='#3498db', label='Energie z TČ')
                ax2.bar(df_binned['Temp'], df_binned['Q_biv'], bottom=df_binned['Q_tc'], color='#e74c3c', label='Energie z bivalence')
                ax2.set_xlabel("Venkovní teplota [°C]"); ax2.set_ylabel("Energie [MWh]"); ax2.legend()
                st.pyplot(fig2)

            st.markdown("---")
            col_c1, col_c2 = st.columns([1, 2])
            with col_c1:
                st.subheader("Podíl bivalence na ELEKTŘINĚ")
                fig_pie, ax_pie = plt.subplots(figsize=(3,3)) # Menší velikost
                ax_pie.pie([el_tc_s, el_biv_s], labels=['TČ', 'Biv'], autopct='%1.1f%%', colors=['#3498db','#e74c3c'], startangle=90)
                st.pyplot(fig_pie)
            with col_c2:
                st.subheader("Sumář roční energie")
                data_sumar = {
                    "Zdroj": ["Tepelná čerpadla", "Bivalence (patrona)", "**CELKEM**"],
                    "Vyrobené teplo [MWh]": [f"{q_tc_s:.1f}", f"{q_biv_s:.1f}", f"**{(q_tc_s+q_biv_s):.1f}**"],
                    "Podíl na teple": [f"{(q_tc_s/(q_tc_s+q_biv_s))*100:.1f} %", f"{(q_biv_s/(q_tc_s+q_biv_s))*100:.1f} %", "100 %"],
                    "Spotřeba el. [MWh]": [f"{el_tc_s:.1f}", f"{el_biv_s:.1f}", f"**{(el_tc_s+el_biv_s):.1f}**"],
                    "Podíl na el.": [f"{(el_tc_s/(el_tc_s+el_biv_s))*100:.1f} %", f"{(el_biv_s/(el_tc_s+el_biv_s))*100:.1f} %", "100 %"]
                }
                st.table(pd.DataFrame(data_sumar))
                st.warning("💡 Všimněte si: Bivalence vyrobí malé % tepla, ale spotřebuje velké % elektřiny (má COP 1).")

        with tab2:
            c1, c2, c3 = st.columns(3)
            c1.metric("Roční úspora", f"{uspora:,.0f} Kč")
            c2.metric("Návratnost", f"{investice/uspora:.1f} let" if uspora > 0 else "N/A")
            c3.metric("SCOP systému", f"{q_tc_s / el_tc_s:.2f}")

            st.markdown("---")
            st.subheader("Srovnání ročních provozních nákladů")
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            labels = ['Stávající stav (CZT)', 'Nový stav (Kaskáda TČ)']
            costs = [naklady_czt, naklady_tc]
            bars = ax3.bar(labels, costs, color=['#95a5a6', '#2ecc71'], width=0.5)
            ax3.set_ylabel("Náklady [Kč/rok]")
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 5000, f'{height:,.0f} Kč', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig3)
            
            st.info(f"Projektovaná roční úspora činí {uspora:,.0f} Kč, což představuje snížení nákladů o {((uspora/naklady_czt)*100):.1f} %.")
            buf = io.BytesIO(); df_sim.to_excel(buf, index=False)
            st.download_button("📥 Stáhnout detailní data (Excel)", buf.getvalue(), "analyza_projektu.xlsx")
