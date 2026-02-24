import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import unicodedata
import tempfile
from fpdf import FPDF

# --- POMOCNÉ FUNKCE ---
def remove_accents(input_str):
    if not input_str: return ""
    nfkd_form = unicodedata.normalize('NFKD', str(input_str))
    return "".join([c for c in nfkd_form if unicode_data.combining(c)]) if 'unicode_data' in globals() else input_str

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

# --- KONFIGURACE ---
st.set_page_config(page_title="Energetický Simulátor TČ", layout="wide")
st.title("🚀 Profesionální simulátor kaskády TČ")

# --- SIDEBAR (Doplněno o TUV) ---
st.sidebar.header("⚙️ Vstupní parametry")
with st.sidebar:
    nazev_projektu = st.text_input("Název projektu", "SVJ Sladkovičova")
    ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
    t_design = st.number_input("Návrhová venkovní teplota [°C]", value=-12.0)
    
    st.markdown("### 🌡️ Otopná soustava")
    t_privod = st.number_input("Návrhová teplota přívodu [°C]", value=60.0)
    t_zpatecka = st.number_input("Návrhová teplota zpátečky [°C]", value=50.0)
    t_min_voda = st.number_input("Teplota vody při +15°C [°C]", value=35.0)
    limit_voda_tc = st.number_input("Max. teplota z TČ [°C]", value=55.0)
    
    st.markdown("### 🚿 Příprava TUV")
    t_tuv_cilova = st.number_input("Požadovaná teplota TUV [°C]", value=55.0)  # Nový parametr
    spotreba_tuv = st.number_input("Roční potřeba pro TUV [MWh/rok]", value=76.0)
    
    st.markdown("### 💰 Ekonomika")
    spotreba_ut = st.number_input("Roční potřeba pro ÚT [MWh/rok]", value=124.0)
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 4)
    cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800.0)
    cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284.0)
    investice = st.number_input("Investice [Kč]", value=3800000.0)

# --- NAHRÁNÍ SOUBORŮ ---
st.subheader("📁 Nahrání datových podkladů")
c_f1, c_f2 = st.columns(2)
with c_f1: tmy_file = st.file_uploader("Meteorologická data (TMY)", type="csv")
with c_f2: char_file = st.file_uploader("Charakteristika TČ (vstupy_TC.csv)", type="csv")

if tmy_file and char_file:
    tmy_raw = load_tmy_robust(tmy_file)
    df_char = load_char(char_file)

    if tmy_raw is not None and df_char is not None:
        tmy = tmy_raw.copy()
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce').fillna(0)
        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        
        sim_data = []
        for _, row in tmy.iterrows():
            t_out = row['T2m']
            month = row['month'] if 'month' in row else 1
            # Ekviterm a limity
            t_voda_req = np.interp(t_out, [t_design, 15], [t_privod, t_min_voda]) if t_out < 20 else t_min_voda
            k_p = 1 - (max(0, t_voda_req - 35.0) * 0.01)
            k_cop = 1 - (max(0, t_voda_req - 35.0) * 0.025)
            
            q_need = max(0, (ztrata * (20 - t_out) / (20 - t_design))) + q_tuv_avg
            p_max = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc * k_p
            
            # Zapojení TČ vs Bivalence
            if t_voda_req > limit_voda_tc:
                q_tc, q_biv = 0, q_need
            else:
                q_tc = min(q_need, p_max)
                q_biv = q_need - q_tc
                
            cop = np.interp(t_out, df_char['Teplota'], df_char['COP']) * k_cop
            sim_data.append([t_out, month, q_need, q_tc, q_biv, q_tc/cop if q_tc > 0 else 0, q_biv/0.98])

        df_sim = pd.DataFrame(sim_data, columns=['Temp', 'Month', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
        
        # Agregace pro grafy
        q_tc_sum = df_sim['Q_tc'].sum()/1000
        q_biv_sum = df_sim['Q_biv'].sum()/1000
        el_tc_sum = df_sim['El_tc'].sum()/1000
        el_biv_sum = df_sim['El_biv'].sum()/1000
        
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        naklady_tc = (el_tc_sum + el_biv_sum) * cena_el + 15000
        uspora = naklady_czt - naklady_tc
        
        # --- ZOBRAZENÍ (TABULKY A GRAFY) ---
        tab1, tab2, tab3 = st.tabs(["📉 Energetická bilance", "📊 Měsíční přehled", "💰 Ekonomika a Export"])
        
        with tab1:
            st.subheader("Krytí potřeby tepla dle venkovní teploty")
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            df_sorted = df_sim.sort_values('Temp')
            ax1.fill_between(df_sorted['Temp'], 0, df_sorted['Q_tc'], color='#3498db', alpha=0.7, label='Krytí TČ')
            ax1.fill_between(df_sorted['Temp'], df_sorted['Q_tc'], df_sorted['Q_need'], color='#e74c3c', alpha=0.7, label='Krytí Bivalence')
            ax1.plot(df_sorted['Temp'], df_sorted['Q_need'], color='black', linewidth=0.8, label='Potřeba objektu')
            ax1.set_xlabel("Venkovní teplota [°C]"); ax1.set_ylabel("Výkon [kW]"); ax1.legend(); st.pyplot(fig1)

        with tab2:
            st.subheader("Měsíční spotřeba energie [MWh]")
            monthly = df_sim.groupby('Month')[['Q_tc', 'Q_biv']].sum() / 1000
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            monthly.plot(kind='bar', stacked=True, ax=ax2, color=['#3498db', '#e74c3c'])
            ax2.set_xticklabels(['Led','Úno','Bře','Dub','Kvě','Čer','Čec','Srp','Zář','Říj','Lis','Pro'], rotation=0)
            st.pyplot(fig2)

        with tab3:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Roční úspora", f"{int(uspora):,} Kč".replace(',',' '))
            col_m2.metric("Návratnost", f"{(investice/uspora):.1f} let" if uspora > 0 else "---")
            col_m3.metric("SCOP systému", f"{(q_tc_sum/el_tc_sum):.2f}")
            col_m4.metric("Podíl bivalence", f"{(q_biv_sum/(q_tc_sum+q_biv_sum)*100):.1f} %")

            c_g1, c_g2 = st.columns([1.5, 1])
            with c_g1:
                fig_econ, ax_econ = plt.subplots(figsize=(8, 5))
                bars = ax_econ.bar(['Původní CZT', 'Nové TČ'], [naklady_czt, naklady_tc], color=['#95a5a6', '#2ecc71'], width=0.5)
                ax_econ.set_title(f"ROČNÍ NÁKLADY (SPÁD {int(t_privod)}/{int(t_zpatecka)} °C)")
                for bar in bars:
                    ax_econ.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10000, f"{int(bar.get_height()):,} Kč".replace(',',' '), ha='center', fontweight='bold')
                st.pyplot(fig_econ)
            with c_g2:
                fig_pie, ax_pie = plt.subplots()
                ax_pie.pie([el_tc_sum, el_biv_sum], labels=['TČ', 'Biv'], autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c'])
                ax_pie.set_title("Podíl spotřebované elektřiny")
                st.pyplot(fig_pie)

            # PDF Export
            if st.button("📄 Generovat kompletní PDF report"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(190, 10, f"EXPERTNÍ ANALÝZA: {nazev_projektu}", ln=True, align='C')
                pdf.set_font("Arial", '', 10)
                pdf.ln(5)
                pdf.cell(190, 7, f"Zadání: Ztráta {ztrata}kW, Spád {t_privod}/{t_zpatecka}C, TUV {t_tuv_cilova}C", ln=True)
                pdf.cell(190, 7, f"Výsledek: Úspora {int(uspora):,} Kč/rok, Návratnost {(investice/uspora):.1f} let", ln=True)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    fig_econ.savefig(tmp.name, dpi=100); pdf.image(tmp.name, x=15, y=pdf.get_y()+5, w=110)
                st.download_button("⬇️ Stáhnout PDF", data=pdf.output(dest='S').encode('latin-1', 'replace'), file_name="Analyza_TC.pdf")

else:
    st.info("Pro spuštění simulace prosím nahrajte soubory TMY a charakteristiku TČ.")
