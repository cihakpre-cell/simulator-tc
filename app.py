import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- 1. KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="Energetický Simulátor TČ", layout="wide")
st.title("🚀 Energetický simulátor kaskády TČ")

# --- 2. SIDEBAR: VSTUPNÍ PARAMETRY ---
st.sidebar.header("⚙️ Základní parametry")
with st.sidebar:
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
    t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
    spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
    spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)
    
    st.markdown("---")
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 3)
    
    st.markdown("---")
    st.header("💰 Ekonomika")
    cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800)
    cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284)
    investice = st.number_input("Investice celkem [Kč]", value=3800000)

# --- 3. FUNKCE PRO NAČÍTÁNÍ ---
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
        return df[['Teplota', 'Vykon_kW', 'COP']]
    except: return None

# --- 4. NAHRÁNÍ SOUBORŮ ---
st.subheader("📁 1. Krok: Nahrání dat")
col1, col2 = st.columns(2)
with col1:
    tmy_file = st.file_uploader("Nahrajte TMY (soubor tmy_...)", type="csv")
with col2:
    char_file = st.file_uploader("Nahrajte Charakteristiku (vstupy_TC.csv)", type="csv")

if tmy_file and char_file:
    # Načtení dat
    tmy = load_tmy_robust(tmy_file)
    df_char_raw = load_char(char_file)

    if tmy is not None and df_char_raw is not None:
        # --- EDITOVATELNÁ TABULKA V SIDEBARU ---
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Charakteristika TČ (editovatelná)")
        st.sidebar.info("Zde můžete měnit parametry TČ přímo na webu.")
        df_char = st.sidebar.data_editor(df_char_raw, num_rows="dynamic", hide_index=True)

        # Příprava TMY
        tmy.columns = tmy.columns.str.strip()
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
        tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()

        # --- 5. VÝPOČET SIMULACE ---
        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        potreba_ut_teorie = [ztrata * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
        k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000)

        res = []
        for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
            q_need = max(0, (ztrata * (20 - t_sm) / (20 - t_design) * k_oprava)) + q_tuv_avg
            p_max = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc
            cop_val = np.interp(t_out, df_char['Teplota'], df_char['COP'])
            q_tc = min(q_need, p_max)
            q_biv = max(0, q_need - q_tc)
            res.append([t_out, q_need, q_tc, q_biv, q_tc/cop_val if q_tc > 0 else 0, q_biv/0.98])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])

        # --- 6. EKONOMIKA A BILANCE ---
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        el_total_mwh = (df_sim['El_tc'].sum() + df_sim['El_biv'].sum()) / 1000
        naklady_tc = el_total_mwh * cena_el + 17000
        uspora = naklady_czt - naklady_tc
        
        df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1
        df_sim['Month'] = df_sim['Month'].clip(1, 12)
        mesicni_df = df_sim.groupby('Month').agg({'Q_need': 'sum', 'Q_tc': 'sum', 'Q_biv': 'sum'}).reset_index()
        for c in ['Q_need', 'Q_tc', 'Q_biv']: mesicni_df[c] /= 1000

        # --- 7. ZOBRAZENÍ VÝSLEDKŮ ---
        st.header(f"📊 Projekt: {nazev_projektu}")
        tab1, tab2 = st.tabs(["💰 Ekonomika a Tabulky", "📈 Grafické přehledy"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Roční úspora", f"{uspora:,.0f} Kč")
            c2.metric("Návratnost", f"{investice/uspora:.1f} let" if uspora > 0 else "N/A")
            c3.metric("Podíl TČ", f"{(df_sim['Q_tc'].sum()/df_sim['Q_need'].sum())*100:.1f} %")
            st.subheader("Měsíční bilance energie [MWh]")
            st.dataframe(mesicni_df.style.format(precision=2), use_container_width=True)

        with tab2:
            # GRAF 1: MONOTONICKÝ GRAF
            st.subheader("1. Četnost teplot a bod bivalence")
            df_sorted = df_sim.sort_values('Temp').reset_index(drop=True)
            fig_mon, ax_mon = plt.subplots(figsize=(10, 4))
            ax_mon.plot(df_sorted.index, df_sorted['Q_need'], 'r', label='Potřeba domu')
            ax_mon.plot(df_sorted.index, df_sorted['Q_tc'], 'b', label='Krytí TČ')
            biv_idx = df_sorted[df_sorted['Q_biv'] > 0.1].index
            if len(biv_idx) > 0:
                ax_mon.fill_between(df_sorted.index[:max(biv_idx)], df_sorted['Q_tc'][:max(biv_idx)], 
                                    df_sorted['Q_need'][:max(biv_idx)], color='red', alpha=0.3, label='Bivalence')
            ax_mon.set_ylabel("Výkon [kW]"); ax_mon.legend(); ax_mon.grid(True, alpha=0.2)
            st.pyplot(fig_mon)

            # GRAF 2: VÝKONOVÁ KŘIVKA
            st.subheader("2. Výkonová rovnováha (kW vs °C)")
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            tx = np.linspace(-15, 20, 100)
            qy = [ztrata * (20 - t) / (20 - t_design) * k_oprava + q_tuv_avg for t in tx]
            py = [np.interp(t, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc for t in tx]
            ax1.plot(tx, qy, 'r', label='Potřeba')
            ax1.plot(tx, py, 'b--', label='Výkon kaskády')
            ax1.set_xlabel("Teplota [°C]"); ax1.set_ylabel("Výkon [kW]"); ax1.legend(); ax1.grid(True, alpha=0.2)
            st.pyplot(fig1)

            # GRAF 3: MĚSÍČNÍ ENERGIE
            st.subheader("3. Měsíční energie [MWh]")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.bar(mesicni_df['Month'], mesicni_df['Q_tc'], label='TČ', color='skyblue')
            ax2.bar(mesicni_df['Month'], mesicni_df['Q_biv'], bottom=mesicni_df['Q_tc'], label='Bivalence', color='salmon')
            ax2.legend(); ax2.set_xlabel("Měsíc"); ax2.set_ylabel("MWh")
            st.pyplot(fig2)

        # EXPORT DO EXCELU
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_sim.to_excel(writer, index=False, sheet_name='Simulace')
            mesicni_df.to_excel(writer, index=False, sheet_name='Mesicni_bilance')
            df_char.to_excel(writer, index=False, sheet_name='Pouzita_charakteristika')
        st.download_button("📥 Stáhnout kompletní report", buf.getvalue(), f"analyza_{nazev_projektu}.xlsx")

    except Exception as e:
        st.error(f"Chyba při výpočtu: {e}")
else:
    st.info("Nahrajte soubory pro zobrazení editovatelné charakteristiky a výsledků.")
