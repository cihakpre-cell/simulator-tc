import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# Konfigurace stránky
st.set_page_config(page_title="TČ Simulátor", layout="wide")
st.title("🚀 Energetický simulátor kaskády TČ")

# --- SIDEBAR: VSTUPY ---
st.sidebar.header("⚙️ Vstupní parametry")
with st.sidebar:
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
    t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
    spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
    spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 3)
    cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800)
    cena_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284)
    investice = st.number_input("Investice celkem [Kč]", value=3800000)

# --- NAHRÁNÍ SOUBORŮ ---
st.subheader("📁 Nahrání datových podkladů")
col1, col2 = st.columns(2)
with col1:
    tmy_file = st.file_uploader("1. Nahrajte TMY (CSV z PVGIS)", type="csv")
with col2:
    char_file = st.file_uploader("2. Nahrajte Charakteristiku TČ (CSV nebo XLSX)", type=["csv", "xlsx"])

def load_data(file):
    """Robustní načítání CSV/XLSX pro české prostředí."""
    if file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        # Zkusíme detekovat kódování a oddělovač
        content = file.getvalue()
        for enc in ['utf-8', 'cp1250', 'iso-8859-2']:
            try:
                text = content.decode(enc)
                # Detekce středníku vs čárky
                sep = ';' if ';' in text.split('\n')[0] else ','
                # Načtení s ohledem na českou desetinnou čárku
                df = pd.read_csv(io.StringIO(text), sep=sep, decimal=',')
                return df
            except:
                continue
        return pd.read_csv(file)

if tmy_file and char_file:
    # Načtení TMY (přeskočení hlavičky PVGIS)
    tmy = pd.read_csv(tmy_file, skiprows=17, sep=None, engine='python')
    tmy.columns = tmy.columns.str.strip()
    tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
    tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
    
    # Načtení Charakteristiky
    df_char = load_data(char_file)
    df_char.columns = df_char.columns.str.strip() # Odstranění mezer z názvů
    
    # Mapování sloupců (ignoruje velikost písmen)
    cols = {c.lower(): c for c in df_char.columns}
    t_col = cols.get('teplota')
    v_col = cols.get('vykon_kw')
    c_col = cols.get('cop')

    if not all([t_col, v_col, c_col]):
        st.error(f"V souboru chybí sloupce 'Teplota', 'Vykon_kW' nebo 'COP'. Nalezeno: {list(df_char.columns)}")
        st.stop()

    # --- VÝPOČET ---
    q_tuv_avg = (spotreba_tuv / 8760) * 1000
    potreba_ut_h = [ztrata * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
    k_oprava = spotreba_ut / (sum(potreba_ut_h) / 1000)
    
    res = []
    for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
        q_need = max(0, (ztrata * (20 - t_sm) / (20 - t_design) * k_oprava)) + q_tuv_avg
        p_max = np.interp(t_out, df_char[t_col], df_char[v_col]) * pocet_tc
        cop = np.interp(t_out, df_char[t_col], df_char[c_col])
        q_tc = min(q_need, p_max)
        q_biv = max(0, q_need - q_tc)
        res.append([t_out, q_need, q_tc, q_biv, q_tc/cop if q_tc > 0 else 0, q_biv/0.98])
    
    df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
    
    # Ekonomika
    cost_czt = (spotreba_ut + spotreba_tuv) * (cena_czt * 3.6)
    el_total_mwh = (df_sim['El_tc'].sum() + df_sim['El_biv'].sum()) / 1000
    cost_tc = el_total_mwh * cena_el + 17000
    uspora = cost_czt - cost_tc

    # --- ZOBRAZENÍ ---
    st.header(f"Výsledky: {nazev_projektu}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Roční úspora", f"{uspora:,.0f} Kč")
    c2.metric("Návratnost", f"{investice/uspora:.1f} let")
    c3.metric("Spotřeba el. (TČ+Biv)", f"{el_total_mwh:.1f} MWh")

    # Graf
    fig, ax = plt.subplots(figsize=(10, 4))
    tr = np.linspace(-15, 18, 100)
    qd = [ztrata * (20 - t) / (20 - t_design) * k_oprava + q_tuv_avg for t in tr]
    pk = [np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tr]
    ax.plot(tr, qd, 'r', label='Potřeba domu')
    ax.plot(tr, pk, 'b--', alpha=0.3, label='Max výkon kaskády')
    ax.fill_between(tr, [min(q,p) for q,p in zip(qd, pk)], qd, color='red', alpha=0.1, hatch='//', label='Bivalence')
    ax.set_title("Výkonová bilance")
    ax.set_xlabel("Teplota [°C]"); ax.set_ylabel("Výkon [kW]"); ax.legend()
    st.pyplot(fig)

    # Tabulka
    st.subheader("📊 Souhrn")
    summary = pd.DataFrame({
        "Parametr": ["Projekt", "Bod bivalence", "Úspora vs CZT", "Návratnost"],
        "Hodnota": [nazev_projektu, f"{np.interp(0, [p-q for p,q in zip(pk, qd)], tr):.1f} °C", f"{uspora:,.0f} Kč", f"{investice/uspora:.1f} let"]
    })
    st.table(summary)

    # Export
    output = io.BytesIO()
    df_sim.to_excel(output, index=False)
    st.download_button("📥 Stáhnout simulaci (Excel)", output.getvalue(), "vysledky.xlsx")
else:
    st.info("Nahrajte prosím oba soubory (TMY i Charakteristiku) pro zahájení simulace.")
