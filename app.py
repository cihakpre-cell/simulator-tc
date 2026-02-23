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
    t_privod = st.slider("Návrhová teplota vody (přívod) [°C]", 35, 75, 60)
    
    st.markdown("---")
    cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800)
    cena_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284)
    investice = st.number_input("Investice celkem [Kč]", value=3800000)

# --- FUNKCE PRO ROBUSTNÍ NAČÍTÁNÍ DAT ---
def load_data(file):
    """Načte CSV nebo XLSX s ohledem na českou lokalizaci (středníky, čárky)."""
    if file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        content = file.getvalue()
        # Zkusíme běžná kódování (UTF-8 pro PVGIS, CP1250 pro český Excel)
        for enc in ['utf-8', 'cp1250', 'iso-8859-2']:
            try:
                text = content.decode(enc)
                # Detekce oddělovače (středník vs čárka)
                first_line = text.split('\n')[0]
                sep = ';' if ';' in first_line else ','
                # Načtení s ohledem na českou desetinnou čárku v CSV
                df = pd.read_csv(io.StringIO(text), sep=sep, decimal=',')
                return df
            except:
                continue
        # Nouzový pád zpět na základní načtení
        return pd.read_csv(file)

# --- NAHRÁNÍ SOUBORŮ ---
st.subheader("📁 Nahrání datových podkladů")
col1, col2 = st.columns(2)

with col1:
    tmy_file = st.file_uploader("1. Nahrajte TMY (CSV z PVGIS)", type="csv")
with col2:
    char_file = st.file_uploader("2. Nahrajte Charakteristiku TČ (CSV nebo XLSX)", type=["csv", "xlsx"])

if tmy_file and char_file:
    try:
        # --- 1. ZPRACOVÁNÍ TMY ---
        # PVGIS soubory mají 16 řádků hlavičky, data začínají na 17. (index 16)
        tmy = pd.read_csv(tmy_file, skiprows=16, sep=None, engine='python')
        tmy.columns = tmy.columns.str.strip()
        
        # Ošetření nečíselných hodnot v teplotě (T2m)
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
        tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
        
        # Výpočet klouzavého průměru pro vyhlazení potřeby tepla
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()

        # --- 2. ZPRACOVÁNÍ CHARAKTERISTIKY ---
        df_char = load_data(char_file)
        df_char.columns = df_char.columns.str.strip()
        
        # Dynamické vyhledání sloupců bez ohledu na velikost písmen
        cols_map = {c.lower(): c for c in df_char.columns}
        t_col = cols_map.get('teplota')
        v_col = cols_map.get('vykon_kw')
        c_col = cols_map.get('cop')

        if not all([t_col, v_col, c_col]):
            st.error(f"V souboru charakteristiky chybí sloupce (Teplota, Vykon_kW, COP). Nalezeno: {list(df_char.columns)}")
            st.stop()

        # --- 3. VÝPOČET SIMULACE ---
        q_tuv_avg = (spotreba_tuv / 8760) * 1000  # Průměrný kW výkon pro TUV
        potreba_ut_h = [ztrata * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
        k_oprava = spotreba_ut / (sum(potreba_ut_h) / 1000) # Korekce na reálnou fakturovanou spotřebu

        res = []
        for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
            # Potřeba objektu v danou hodinu
            q_need = max(0, (ztrata * (20 - t_sm) / (20 - t_design) * k_oprava)) + q_tuv_avg
            # Maximální výkon kaskády při venkovní teplotě
            p_max = np.interp(t_out, df_char[t_col], df_char[v_col]) * pocet_tc
            cop_val = np.interp(t_out, df_char[t_col], df_char[c_col])
            
            q_tc = min(q_need, p_max)      # Výkon dodaný čerpadly
            q_biv = max(0, q_need - q_tc)  # Výkon dodaný bivalencí
            
            res.append([
                t_out, 
                q_need, 
                q_tc, 
                q_biv, 
                q_tc / cop_val if q_tc > 0 else 0, 
                q_biv / 0.98 # Účinnost elektrokotle/bivalence
            ])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need_kW', 'Q_tc_kW', 'Q_biv_kW', 'El_tc_kW', 'El_biv_kW'])

        # --- 4. EKONOMIKA ---
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_czt * 3.6) # Převod GJ na MWh pro výpočet z ceny GJ
        el_tc_rok_mwh = df_sim['El_tc_kW'].sum() / 1000
        el_biv_rok_mwh = df_sim['El_biv_kW'].sum() / 1000
        
        naklady_tc = (el_tc_rok_mwh + el_biv_rok_mwh) * cena_el + 17000 # 17k je paušál za servis/jističe
        uspora = naklady_czt - naklady_tc
        návratnost = investice / uspora if uspora > 0 else 0

        # --- 5. VIZUALIZACE ---
        st.success(f"Simulace pro projekt '{nazev_projektu}' proběhla úspěšně.")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Roční úspora", f"{uspora:,.0f} Kč")
        m2.metric("Návratnost", f"{návratnost:.1f} let")
        m3.metric("Spotřeba TČ", f"{el_tc_rok_mwh:.1f} MWh")
        m4.metric("Podíl bivalence", f"{(el_biv_rok_mwh/(el_tc_rok_mwh+el_biv_rok_mwh + 0.001))*100:.1f} %")

        # Graf výkonové rovnováhy
        fig, ax = plt.subplots(figsize=(10, 4))
        temp_range = np.linspace(-15, 18, 100)
        q_house = [ztrata * (20 - t) / (20 - t_design) * k_oprava + q_tuv_avg for t in temp_range]
        q_pumps = [np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in temp_range]
        
        ax.plot(temp_range, q_house, 'r', label='Potřeba domu (ÚT+TUV)')
        ax.plot(temp_range, q_pumps, 'b--', alpha=0.5, label=f'Max výkon kaskády ({pocet_tc}ks)')
        ax.fill_between(temp_range, [min(h, p) for h, p in zip(q_house, q_pumps)], q_house, color='red', alpha=0.1, label='Oblast bivalence')
        
        ax.set_xlabel("Venkovní teplota [°C]")
        ax.set_ylabel("Výkon [kW]")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Tabulka výsledků
        st.subheader("📊 Souhrnná tabulka")
        summary_df = pd.DataFrame({
            "Parametr": ["Návrhová ztráta objektu", "Celková roční úspora", "Doba návratnosti", "Náklady na CZT (původní)", "Náklady na TČ (nové)"],
            "Hodnota": [f"{ztrata} kW", f"{uspora:,.0f} Kč", f"{návratnost:.1f} let", f"{naklady_czt:,.0f} Kč", f"{naklady_tc:,.0f} Kč"]
        })
        st.table(summary_df)

        # Export do Excelu
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_sim.to_excel(writer, index=False, sheet_name='Hodinova_simulace')
        
        st.download_button(
            label="📥 Stáhnout hodinovou simulaci v Excelu",
            data=output.getvalue(),
            file_name=f"simulace_{nazev_projektu}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Došlo k chybě při zpracování dat: {e}")
        st.info("Zkontrolujte, zda soubor TMY má správný formát a zda charakteristika obsahuje sloupce Teplota;Vykon_kW;COP.")

else:
    st.info("👋 Vítejte! Pro spuštění výpočtu nahrajte vlevo oba potřebné soubory.")
    st.image("https://img.freepik.com/free-vector/energy-efficiency-concept-illustration_114360-10022.jpg", width=400)
