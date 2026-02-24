import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Nastavení stránky
st.set_page_config(page_title="Simulace Kaskády TČ", layout="wide")

st.title("🚀 Energetický simulátor kaskády TČ")
st.markdown("Tento nástroj provádí hodinovou simulaci provozu na základě dat TMY.")

# --- SIDEBAR: VSTUPNÍ PARAMETRY ---
st.sidebar.header("⚙️ Vstupní parametry")

with st.sidebar:
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    
    col1, col2 = st.columns(2)
    with col1:
        ztrata_celkova = st.number_input("Tepelná ztráta [kW]", value=54.0)
        t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
    with col2:
        fakt_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
        f_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)

    st.divider()
    
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 3)
    t_privod = st.slider("Návrhová teplota vody (přívod) [°C]", 35, 75, 60)
    
    st.divider()
    
    cena_el_mwh = st.number_input("Cena elektřiny [Kč/MWh]", value=4800)
    cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284)
    investice = st.number_input("Investice celkem [Kč]", value=3800000)

# --- NAHRÁNÍ DAT (TMY a Charakteristika) ---
# Pro webovou verzi je lepší mít TMY a Char. jako fixní soubory nebo nahrávací pole
tmy_uploaded = st.file_uploader("1. Nahrajte soubor TMY (CSV z PVGIS)", type="csv")
char_uploaded = st.file_uploader("2. Nahrajte charakteristiku TČ (CSV)", type="csv")

if tmy_uploaded and char_uploaded:
    # Načtení dat
    tmy = pd.read_csv(tmy_uploaded, skiprows=17)
    tmy.columns = tmy.columns.str.strip()
    tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
    tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
    tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
    
    df_char = pd.read_csv(char_uploaded)
    
    # --- VÝPOČET ---
    q_tuv_avg = (f_tuv / 8760) * 1000
    potreba_ut_teorie = [ztrata_celkova * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
    k_oprava = fakt_ut / (sum(potreba_ut_teorie) / 1000)
    naklady_czt_rok = (fakt_ut + f_tuv) * (cena_gj_czt * 3.6)

    # Simulace
    res = []
    for t_out, t_smooth in zip(tmy['T2m'], tmy['T_smooth']):
        q_total = max(0, (ztrata_celkova * (20 - t_smooth) / (20 - t_design) * k_oprava)) + q_tuv_avg
        p_max = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc
        cop = np.interp(t_out, df_char['Teplota'], df_char['COP'])
        q_tc = min(q_total, p_max)
        q_biv = max(0, q_total - q_tc)
        res.append([t_out, q_total, q_tc, q_biv, q_tc/cop if q_tc > 0 else 0, q_biv/0.98])

    df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need_kW', 'Q_tc_kW', 'Q_biv_kW', 'El_tc_kW', 'El_biv_kW'])
    
    # Ekonomika
    el_tc_mwh = df_sim['El_tc_kW'].sum() / 1000
    el_biv_mwh = df_sim['El_biv_kW'].sum() / 1000
    naklady_tc = (el_tc_mwh + el_biv_mwh) * cena_el_mwh + 17000
    uspora = naklady_czt_rok - naklady_tc
    
    # --- ZOBRAZENÍ VÝSLEDKŮ ---
    st.header(f"Výsledky analýzy: {nazev_projektu}")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Roční úspora", f"{uspora:,.0f} Kč")
    m2.metric("Návratnost", f"{investice/uspora:.1f} let")
    m3.metric("Spotřeba TČ", f"{el_tc_mwh:.1f} MWh")
    m4.metric("Podíl bivalence", f"{(el_biv_mwh/(el_tc_mwh+el_biv_mwh))*100:.1f} %")

    # Grafy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graf výkonu
    t_r = np.linspace(-15, 18, 100)
    q_d = [(ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg for t in t_r]
    p_k = [np.interp(t, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc for t in t_r]
    ax1.plot(t_r, q_d, 'r', label='Potřeba domu')
    ax1.plot(t_r, p_k, 'b--', alpha=0.3, label='Max výkon kaskády')
    ax1.fill_between(t_r, [min(q,p) for q,p in zip(q_d, p_k)], q_d, color='red', alpha=0.1, label='Bivalence')
    ax1.set_title("Výkonová rovnováha")
    ax1.legend()
    
    # Histogram
    ax2.hist(tmy['T2m'], bins=30, color='skyblue', edgecolor='white')
    ax2.set_title("Rozdělení teplot v roce")
    
    st.pyplot(fig)

    # Export
    st.download_button("Stáhnout hodinovou simulaci (Excel)", 
                       data=df_sim.to_csv().encode('utf-8'), 
                       file_name=f"simulace_{nazev_projektu}.csv")
else:
    st.info("Prosím nahrajte vstupní soubory TMY a Charakteristiku v CSV pro spuštění simulace.")
