import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- 1. KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="Energetický Simulátor TČ", layout="wide")
st.title("🚀 Pokročilý simulátor kaskády TČ")

# --- 2. SIDEBAR: VSTUPNÍ PARAMETRY ---
st.sidebar.header("⚙️ Systémové parametry")
with st.sidebar:
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
    t_design = st.number_input("Návrhová venkovní teplota [°C]", value=-12.0)
    
    st.markdown("---")
    st.subheader("🌡️ Otopná soustava")
    t_voda_max = st.number_input("Teplota vody při návrhové t. [°C]", value=60.0)
    t_voda_min = st.number_input("Teplota vody při +15°C [°C]", value=35.0)
    t_tuv = st.number_input("Požadovaná teplota TUV [°C]", value=55.0)
    
    st.markdown("---")
    spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
    spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 3)
    
    st.markdown("---")
    st.header("💰 Ekonomika")
    cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800.0)
    cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284.0)
    investice = st.number_input("Investice celkem [Kč]", value=3800000.0)

# --- 3. POMOCNÉ FUNKCE ---
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
        df = pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
        df.columns = df.columns.str.strip()
        return df[['Teplota', 'Vykon_kW', 'COP']]
    except: return None

# --- 4. NAHRÁNÍ A EDITACE ---
st.subheader("📁 1. Krok: Data a Charakteristika")
col1, col2 = st.columns(2)
with col1:
    tmy_file = st.file_uploader("Nahrajte TMY", type="csv")
with col2:
    char_file = st.file_uploader("Nahrajte Charakteristiku (vstupy_TC.csv)", type="csv")

if tmy_file and char_file:
    tmy = load_tmy_robust(tmy_file)
    df_char_raw = load_char(char_file)

    if tmy is not None and df_char_raw is not None:
        st.info("💡 Tip: Tabulka níže jsou data výrobce pro jednu teplotu vody (např. 35°C). Model je automaticky přepočítá podle ekvitermy.")
        df_char = st.data_editor(df_char_raw, num_rows="dynamic", hide_index=True)

        # Příprava TMY
        tmy.columns = tmy.columns.str.strip()
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
        tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()

        # --- 5. VÝPOČET S KOREKCÍ NA TEPLOTU VODY ---
        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        potreba_ut_teorie = [ztrata * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
        k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000)

        res = []
        for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
            # Ekvitermní teplota vody (lineární zjednodušení)
            if t_sm < 20:
                t_voda_req = np.interp(t_sm, [t_design, 15], [t_voda_max, t_voda_min])
            else:
                t_voda_req = t_voda_min
            
            # Korekční faktor (Carnotovo přiblížení)
            # Každý stupeň nad 35°C snižuje COP cca o 2.5% a výkon o 1%
            t_ref = 35.0
            korecke_cop = 1 - (max(0, t_voda_req - t_ref) * 0.025)
            korekce_vykon = 1 - (max(0, t_voda_req - t_ref) * 0.01)

            q_need = max(0, (ztrata * (20 - t_sm) / (20 - t_design) * k_oprava)) + q_tuv_avg
            
            # Interpolace základního výkonu a COP z tabulky
            p_base = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc
            cop_base = np.interp(t_out, df_char['Teplota'], df_char['COP'])
            
            p_real = p_base * korekce_vykon
            cop_real = cop_base * korecke_cop
            
            q_tc = min(q_need, p_real)
            q_biv = max(0, q_need - q_tc)
            
            res.append([t_out, q_need, q_tc, q_biv, q_tc/cop_real if q_tc > 0 else 0, q_biv/0.98, t_voda_req])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv', 'T_voda'])

        # --- 6. EKONOMIKA A GRAFY ---
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        el_total_mwh = (df_sim['El_tc'].sum() + df_sim['El_biv'].sum()) / 1000
        naklady_tc = el_total_mwh * cena_el + 17000
        uspora = naklady_czt - naklady_tc
        
        st.header(f"📊 Výsledky: {nazev_projektu}")
        tab1, tab2 = st.tabs(["💰 Přehled a Bilance", "📈 Pokročilé grafy"])

        with tab1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Roční úspora", f"{uspora:,.0f} Kč")
            c2.metric("Návratnost", f"{investice/uspora:.1f} let" if uspora > 0 else "N/A")
            c3.metric("SCOP systému", f"{df_sim['Q_tc'].sum() / df_sim['El_tc'].sum():.2f}")
            c4.metric("Podíl bivalence", f"{(df_sim['Q_biv'].sum()/df_sim['Q_need'].sum())*100:.1f} %")

        with tab2:
            st.subheader("Výkonová rovnováha s vlivem teploty vody")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_sim.sort_values('Temp')['Temp'], df_sim.sort_values('Temp')['Q_need'], 'r', label='Potřeba domu')
            ax.plot(df_sim.sort_values('Temp')['Temp'], df_sim.sort_values('Temp')['Q_tc'], 'b', label='Krytí TČ (korigované)')
            ax.set_xlabel("Venkovní teplota [°C]"); ax.set_ylabel("Výkon [kW]"); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.subheader("Průběh teploty otopné vody (Ekviterm)")
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.plot(df_sim['T_voda'][:24*7], label='Teplota vody (1. týden)')
            ax2.set_ylabel("°C"); ax2.legend(); ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_sim.to_excel(writer, index=False)
        st.download_button("📥 Stáhnout data", buf.getvalue(), "simulace.xlsx")

else:
    st.info("Nahrajte soubory pro spuštění výpočtu.")
