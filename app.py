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
    st.subheader("🌡️ Otopná soustava a TUV")
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
        return df[['Teplota', 'Vykon_kW', 'COP']].copy()
    except: return None

# --- 4. NAHRÁNÍ A EDITACE ---
st.subheader("📁 1. Krok: Nahrání dat")
col1, col2 = st.columns(2)
with col1:
    tmy_file = st.file_uploader("Nahrajte TMY", type="csv")
with col2:
    char_file = st.file_uploader("Nahrajte Charakteristiku (vstupy_TC.csv)", type="csv")

if tmy_file and char_file:
    tmy_raw = load_tmy_robust(tmy_file)
    df_char_raw = load_char(char_file)

    if tmy_raw is not None and df_char_raw is not None:
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Charakteristika TČ (pro 35°C)")
        df_char = st.sidebar.data_editor(df_char_raw, num_rows="dynamic", hide_index=True)

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
            if t_sm < 20:
                t_voda_req = np.interp(t_sm, [t_design, 15], [t_voda_max, t_voda_min])
            else:
                t_voda_req = t_voda_min
            
            t_ref = 35.0
            delta_t = max(0, t_voda_req - t_ref)
            korecke_cop = 1 - (delta_t * 0.025)
            korekce_vykon = 1 - (delta_t * 0.01)

            q_need = max(0, (ztrata * (20 - t_sm) / (20 - t_design) * k_oprava)) + q_tuv_avg
            p_base = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc
            cop_base = np.interp(t_out, df_char['Teplota'], df_char['COP'])
            
            p_real = p_base * korekce_vykon
            cop_real = cop_base * korecke_cop
            
            q_tc = min(q_need, p_real)
            q_biv = max(0, q_need - q_tc)
            
            res.append([t_out, q_need, q_tc, q_biv, q_tc/cop_real if q_tc > 0 else 0, q_biv/0.98, t_voda_req])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv', 'T_voda'])

        # --- 6. VÝSLEDKY ---
        q_tc_sum = df_sim['Q_tc'].sum() / 1000
        q_biv_sum = df_sim['Q_biv'].sum() / 1000
        el_tc_sum = df_sim['El_tc'].sum() / 1000
        el_biv_sum = df_sim['El_biv'].sum() / 1000
        
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        naklady_tc = (el_tc_sum + el_biv_sum) * cena_el + 17000
        uspora = naklady_czt - naklady_tc
        
        st.header(f"📊 Analýza: {nazev_projektu}")
        tab1, tab2, tab3 = st.tabs(["💰 Ekonomika a Podíl", "📅 Měsíční bilance", "📈 Grafy"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Roční úspora", f"{uspora:,.0f} Kč")
            c2.metric("Podíl TČ na výrobě", f"{(q_tc_sum/(q_tc_sum+q_biv_sum))*100:.1f} %")
            c3.metric("SCOP systému", f"{q_tc_sum / el_tc_sum:.2f}")

            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("Poměr vyrobeného tepla")
                fig_pie, ax_pie = plt.subplots()
                ax_pie.pie([q_tc_sum, q_biv_sum], labels=['Tepelná čerpadla', 'Bivalence'], 
                           autopct='%1.1f%%', colors=['#5dade2', '#ec7063'], startangle=90)
                st.pyplot(fig_pie)
            
            with col_right:
                st.subheader("Rozpis spotřeby elektřiny")
                st.write(f"**Elektřina pro TČ (kompresor):** {el_tc_sum:.1f} MWh")
                st.write(f"**Elektřina pro bivalenci (patrona):** {el_biv_sum:.1f} MWh")
                st.progress(el_tc_sum / (el_tc_sum + el_biv_sum))

        with tab2:
            df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1
            df_sim['Month'] = df_sim['Month'].clip(1, 12)
            mes_df = df_sim.groupby('Month').agg({'Q_tc': 'sum', 'Q_biv': 'sum', 'El_tc': 'sum', 'El_biv': 'sum'}).reset_index()
            for c in ['Q_tc', 'Q_biv', 'El_tc', 'El_biv']: mes_df[c] /= 1000
            mes_df.columns = ['Měsíc', 'Teplo TČ [MWh]', 'Teplo Biv [MWh]', 'El. TČ [MWh]', 'El. Biv [MWh]']
            st.dataframe(mes_df.style.format(precision=2), use_container_width=True)

        with tab3:
            st.subheader("Hodinové krytí potřeby v roce")
            df_sort = df_sim.sort_values('Temp').reset_index(drop=True)
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.fill_between(df_sort.index, 0, df_sort['Q_tc'], color='#5dade2', label='Krytí TČ')
            ax1.fill_between(df_sort.index, df_sort['Q_tc'], df_sort['Q_need'], color='#ec7063', label='Krytí Bivalence')
            ax1.set_ylabel("Výkon [kW]")
            ax1.legend()
            st.pyplot(fig1)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_sim.to_excel(writer, index=False)
        st.download_button("📥 Exportovat výsledky", buf.getvalue(), f"analyza_{nazev_projektu}.xlsx")

else:
    st.info("Nahrajte soubory pro výpočet.")
