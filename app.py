import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="TČ Simulátor", layout="wide")
st.title("🚀 Energetický simulátor kaskády TČ")
st.markdown("---")

# --- SIDEBAR: VSTUPNÍ PARAMETRY ---
st.sidebar.header("⚙️ Vstupní parametry")
with st.sidebar:
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
    t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
    spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
    spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)
    
    st.markdown("---")
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 3)
    
    st.markdown("---")
    cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800)
    cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284)
    investice = st.number_input("Investice celkem [Kč]", value=3800000)

# --- FUNKCE PRO INTELIGENTNÍ NAČÍTÁNÍ TMY ---
def load_tmy_robust(file):
    content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
    # Hledáme řádek, který obsahuje 'T2m'
    header_idx = -1
    for i, line in enumerate(content):
        if 'T2m' in line:
            header_idx = i
            break
    
    if header_idx == -1:
        return None
    
    # Načteme od nalezeného řádku dál
    df = pd.read_csv(io.StringIO("\n".join(content[header_idx:])))
    return df

# --- FUNKCE PRO NAČÍTÁNÍ CHARAKTERISTIKY ---
def load_char(file):
    content = file.getvalue().decode('utf-8-sig', errors='ignore')
    sep = ';' if ';' in content.split('\n')[0] else ','
    df = pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
    return df

# --- HLAVNÍ ČÁST ---
st.subheader("📁 Nahrání datových podkladů")
col1, col2 = st.columns(2)

with col1:
    tmy_file = st.file_uploader("1. Nahrajte TMY (soubor tmy_...)", type="csv")
with col2:
    char_file = st.file_uploader("2. Nahrajte Charakteristiku TČ (vstupy_TC.csv)", type="csv")

if tmy_file and char_file:
    try:
        # 1. NAČTENÍ TMY
        tmy = load_tmy_robust(tmy_file)
        if tmy is None:
            st.error("V souboru TMY nebyl nalezen sloupec 'T2m'. Zkontrolujte formát souboru.")
            st.stop()
        
        tmy.columns = tmy.columns.str.strip()
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
        tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()

        # 2. NAČTENÍ CHARAKTERISTIKY
        df_char = load_char(char_file)
        df_char.columns = df_char.columns.str.strip()
        t_col, v_col, c_col = 'Teplota', 'Vykon_kW', 'COP'

        # 3. VÝPOČET
        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        potreba_ut_teorie = [ztrata * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
        k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000)

        res = []
        for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
            q_need = max(0, (ztrata * (20 - t_sm) / (20 - t_design) * k_oprava)) + q_tuv_avg
            p_max = np.interp(t_out, df_char[t_col], df_char[v_col]) * pocet_tc
            cop_val = np.interp(t_out, df_char[t_col], df_char[c_col])
            q_tc = min(q_need, p_max)
            q_biv = max(0, q_need - q_tc)
            res.append([t_out, q_need, q_tc, q_biv, q_tc/cop_val if q_tc > 0 else 0, q_biv/0.98])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])

        # --- 4. EKONOMIKA A BILANCE ---
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        el_tc_rok_mwh = df_sim['El_tc_kW'].sum() / 1000
        el_biv_rok_mwh = df_sim['El_biv_kW'].sum() / 1000
        naklady_tc = (el_tc_rok_mwh + el_biv_rok_mwh) * cena_el + 17000
        uspora = naklady_czt - naklady_tc
        navratnost = investice / uspora if uspora > 0 else 0

        # Příprava měsíční tabulky (to co bylo v Colabu)
        # Předpokládáme, že TMY má 8760 řádků (365 dní * 24h)
        df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1 # Jednoduchý odhad měsíce
        df_sim['Month'] = df_sim['Month'].clip(1, 12)
        
        mesicni_df = df_sim.groupby('Month').agg({
            'Temp': 'mean',
            'Q_need_kW': 'sum',
            'Q_tc_kW': 'sum',
            'Q_biv_kW': 'sum',
            'El_tc_kW': 'sum'
        }).reset_index()
        
        # Převod na MWh pro tabulku
        for col in ['Q_need_kW', 'Q_tc_kW', 'Q_biv_kW', 'El_tc_kW']:
            mesicni_df[col] = mesicni_df[col] / 1000
        
        mesicni_df.columns = ['Měsíc', 'Prům. teplota [°C]', 'Potřeba [MWh]', 'Krytí TČ [MWh]', 'Bivalence [MWh]', 'Spotřeba el. [MWh]']

        # --- 5. ZOBRAZENÍ V ZÁLOŽKÁCH ---
        st.header(f"📊 Komplexní analýza: {nazev_projektu}")
        
        tab1, tab2, tab3 = st.tabs(["💰 Ekonomický přehled", "📅 Měsíční bilance", "📈 Detailní grafy"])

        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Finanční přínos")
                st.metric("Roční úspora", f"{uspora:,.0f} Kč")
                st.metric("Návratnost investice", f"{navratnost:.1f} let")
            with col_b:
                st.subheader("Provozní náklady")
                st.write(f"**Původní náklady (CZT):** {naklady_czt:,.0f} Kč")
                st.write(f"**Nové náklady (Elektřina):** {naklady_tc:,.0f} Kč")
                st.write(f"**Investiční náklady:** {investice:,.0f} Kč")

        with tab2:
            st.subheader("Tabulka měsíčních odběrů a krytí")
            st.dataframe(mesicni_df.style.format(precision=2), use_container_width=True)
            
            # Výpočet SCOP
            scop_projekt = df_sim['Q_tc_kW'].sum() / df_sim['El_tc_kW'].sum() if df_sim['El_tc_kW'].sum() > 0 else 0
            st.info(f"**Průměrný sezónní topný faktor (SCOP) kaskády: {scop_projekt:.2f}**")

        with tab3:
            st.subheader("Výkonová křivka a bod bivalence")
            fig, ax = plt.subplots(figsize=(10, 4))
            tx = np.linspace(-15, 20, 100)
            qy = [ztrata * (20 - t) / (20 - t_design) * k_oprava + q_tuv_avg for t in tx]
            py = [np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tx]
            ax.plot(tx, qy, 'r', label='Potřeba tepla objektu')
            ax.plot(tx, py, 'b--', label='Maximální výkon kaskády TČ')
            ax.fill_between(tx, [min(a,b) for a,b in zip(qy,py)], qy, color='red', alpha=0.1, label='Oblast bivalence (E-kotel)')
            ax.set_xlabel("Venkovní teplota [°C]")
            ax.set_ylabel("Výkon [kW]")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # Export (zůstává stejný)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_sim.to_excel(writer, index=False, sheet_name='Hodinova_data')
            mesicni_df.to_excel(writer, index=False, sheet_name='Mesicni_bilance')
        
        st.download_button("📥 Stáhnout kompletní report (Excel)", output.getvalue(), f"analyza_{nazev_projektu}.xlsx")

