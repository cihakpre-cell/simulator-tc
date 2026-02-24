import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- 1. KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="Energetický Simulátor TČ", layout="wide")
st.title("🚀 Energetický simulátor kaskády TČ")
st.markdown("Propojení meteorologických dat TMY a výkonových charakteristik kaskády.")

# --- 2. SIDEBAR: VSTUPNÍ PARAMETRY ---
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

# --- 3. FUNKCE PRO ROBUSTNÍ NAČÍTÁNÍ ---
def load_tmy_robust(file):
    """Najde v TMY souboru řádek s T2m a načte data."""
    try:
        content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
        header_idx = -1
        for i, line in enumerate(content):
            if 'T2m' in line:
                header_idx = i
                break
        if header_idx == -1: return None
        return pd.read_csv(io.StringIO("\n".join(content[header_idx:])))
    except:
        return None

def load_char(file):
    """Načte charakteristiku s detekcí středníku a des. čárky."""
    try:
        content = file.getvalue().decode('utf-8-sig', errors='ignore')
        sep = ';' if ';' in content.split('\n')[0] else ','
        return pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
    except:
        return None

# --- 4. NAHRÁNÍ SOUBORŮ ---
st.subheader("📁 Nahrání datových podkladů")
col1, col2 = st.columns(2)

with col1:
    tmy_file = st.file_uploader("1. Nahrajte TMY (soubor tmy_...)", type="csv")
with col2:
    char_file = st.file_uploader("2. Nahrajte Charakteristiku TČ (vstupy_TC.csv)", type="csv")

if tmy_file and char_file:
    try:
        # Zpracování TMY
        tmy = load_tmy_robust(tmy_file)
        if tmy is None:
            st.error("Chyba: V souboru TMY nebyl nalezen sloupec 'T2m'.")
            st.stop()
        
        tmy.columns = tmy.columns.str.strip()
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
        tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()

        # Zpracování Charakteristiky
        df_char = load_char(char_file)
        if df_char is None:
            st.error("Chyba: Nepodařilo se načíst soubor charakteristiky.")
            st.stop()
            
        df_char.columns = df_char.columns.str.strip()
        t_col, v_col, c_col = 'Teplota', 'Vykon_kW', 'COP'

        # --- 5. VÝPOČET SIMULACE ---
        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        # Teoretická potřeba pro korekční faktor
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

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need_kW', 'Q_tc_kW', 'Q_biv_kW', 'El_tc_kW', 'El_biv_kW'])

        # --- 6. EKONOMIKA A MĚSÍČNÍ BILANCE ---
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        el_tc_rok_mwh = df_sim['El_tc_kW'].sum() / 1000
        el_biv_rok_mwh = df_sim['El_biv_kW'].sum() / 1000
        naklady_tc = (el_tc_rok_mwh + el_biv_rok_mwh) * cena_el + 17000
        uspora = naklady_czt - naklady_tc
        navratnost = investice / uspora if uspora > 0 else 0

        # Měsíční tabulka
        df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1
        df_sim['Month'] = df_sim['Month'].clip(1, 12)
        mesicni_df = df_sim.groupby('Month').agg({
            'Temp': 'mean',
            'Q_need_kW': 'sum', 'Q_tc_kW': 'sum', 'Q_biv_kW': 'sum', 'El_tc_kW': 'sum'
        }).reset_index()
        
        for col in ['Q_need_kW', 'Q_tc_kW', 'Q_biv_kW', 'El_tc_kW']:
            mesicni_df[col] = mesicni_df[col] / 1000
        mesicni_df.columns = ['Měsíc', 'Prům. teplota [°C]', 'Potřeba [MWh]', 'Krytí TČ [MWh]', 'Bivalence [MWh]', 'Spotřeba el. [MWh]']

        # --- 7. ZOBRAZENÍ VÝSLEDKŮ ---
        st.header(f"📊 Výsledky analýzy: {nazev_projektu}")
        tab1, tab2, tab3 = st.tabs(["💰 Ekonomika", "📅 Měsíční bilance", "📈 Grafy"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Roční úspora", f"{uspora:,.0f} Kč")
            c2.metric("Návratnost", f"{navratnost:.1f} let")
            c3.metric("SCOP kaskády", f"{df_sim['Q_tc_kW'].sum() / df_sim['El_tc_kW'].sum():.2f}")
            
            st.markdown("---")
            st.write(f"**Původní náklady (CZT):** {naklady_czt:,.0f} Kč")
            st.write(f"**Nové náklady (Elektřina):** {naklady_tc:,.0f} Kč")

        with tab2:
            st.dataframe(mesicni_df.style.format(precision=2), use_container_width=True)

        with tab3:
            fig, ax = plt.subplots(figsize=(10, 4))
            tx = np.linspace(-15, 20, 100)
            qy = [ztrata * (20 - t) / (20 - t_design) * k_oprava + q_tuv_avg for t in tx]
            py = [np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tx]
            ax.plot(tx, qy, 'r', label='Potřeba domu')
            ax.plot(tx, py, 'b--', label='Výkon kaskády')
            ax.fill_between(tx, [min(a,b) for a,b in zip(qy,py)], qy, color='red', alpha=0.1, label='Bivalence')
            ax.set_xlabel("Teplota [°C]"); ax.set_ylabel("Výkon [kW]"); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # --- 8. EXPORT ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_sim.to_excel(writer, index=False, sheet_name='Hodinova_data')
            mesicni_df.to_excel(writer, index=False, sheet_name='Mesicni_bilance')
        
        st.download_button("📥 Stáhnout kompletní Excel", output.getvalue(), f"analyza_{nazev_projektu}.xlsx")

    except Exception as e:
        st.error(f"Došlo k chybě při výpočtu: {e}")

else:
    st.info("Nahrajte prosím TMY soubor a charakteristiku TČ.")
