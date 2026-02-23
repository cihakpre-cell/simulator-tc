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

# --- FUNKCE PRO NAČÍTÁNÍ ---
def load_charakteristika(file):
    """Načte CSV charakteristiky se středníkem a desetinnou čárkou."""
    content = file.getvalue().decode('utf-8-sig', errors='ignore')
    # Detekce oddělovače (středník je u vás standard)
    sep = ';' if ';' in content.split('\n')[0] else ','
    df = pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
    return df

# --- NAHRÁNÍ SOUBORŮ ---
st.subheader("📁 Nahrání datových podkladů")
col1, col2 = st.columns(2)

with col1:
    tmy_file = st.file_uploader("1. Nahrajte TMY (soubor tmy_50.024...)", type="csv")
with col2:
    char_file = st.file_uploader("2. Nahrajte Charakteristiku TČ (vstupy_TC.csv)", type="csv")

if tmy_file and char_file:
    try:
        # 1. ZPRACOVÁNÍ TMY
        # Data začínají na řádku 17 (index 16), oddělovač je čárka
        tmy = pd.read_csv(tmy_file, skiprows=16)
        tmy.columns = tmy.columns.str.strip()
        
        # Kontrola sloupce T2m
        if 'T2m' not in tmy.columns:
            st.error(f"V TMY chybí sloupec 'T2m'. Nalezeno: {list(tmy.columns)}")
            st.stop()
            
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
        tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()

        # 2. ZPRACOVÁNÍ CHARAKTERISTIKY
        df_char = load_charakteristika(char_file)
        df_char.columns = df_char.columns.str.strip()
        
        # Namapování sloupců (Teplota, Vykon_kW, COP)
        t_col, v_col, c_col = 'Teplota', 'Vykon_kW', 'COP'
        
        # Převod na čísla (pokud by load_data selhal u některých řádků)
        for c in [t_col, v_col, c_col]:
            df_char[c] = pd.to_numeric(df_char[c], errors='coerce')
        df_char = df_char.dropna(subset=[t_col, v_col, c_col])

        # 3. VÝPOČET
        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        # Teoretická potřeba pro výpočet korekčního faktoru
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

        # 4. VÝSLEDKY
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        el_total_mwh = (df_sim['El_tc'].sum() + df_sim['El_biv'].sum()) / 1000
        naklady_tc = el_total_mwh * cena_el + 17000
        uspora = naklady_czt - naklady_tc
        navratnost = investice / uspora if uspora > 0 else 0

        # --- ZOBRAZENÍ ---
        st.success(f"Analýza projektu {nazev_projektu} hotova.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Roční úspora", f"{uspora:,.0f} Kč")
        c2.metric("Návratnost", f"{navratnost:.1f} let")
        c3.metric("Spotřeba elektřiny", f"{el_total_mwh:.1f} MWh")

        # Graf
        fig, ax = plt.subplots(figsize=(10, 4))
        tx = np.linspace(-15, 20, 100)
        qy = [ztrata * (20 - t) / (20 - t_design) * k_oprava + q_tuv_avg for t in tx]
        py = [np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tx]
        ax.plot(tx, qy, 'r', label='Potřeba domu')
        ax.plot(tx, py, 'b--', label='Výkon kaskády')
        ax.fill_between(tx, [min(a,b) for a,b in zip(qy,py)], qy, color='red', alpha=0.1, label='Bivalence')
        ax.set_xlabel("Teplota [°C]"); ax.set_ylabel("Výkon [kW]"); ax.legend(); ax.grid(True)
        st.pyplot(fig)

        # Tabulka
        st.table(pd.DataFrame({
            "Parametr": ["Původní náklady (CZT)", "Nové náklady (TČ)", "Úspora"],
            "Hodnota": [f"{naklady_czt:,.0f} Kč", f"{naklady_tc:,.0f} Kč", f"{uspora:,.0f} Kč"]
        }))

    except Exception as e:
        st.error(f"Chyba při zpracování: {e}")
else:
    st.info("Nahrajte soubory pro spuštění výpočtu.")
