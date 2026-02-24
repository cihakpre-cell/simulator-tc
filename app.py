import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import unicodedata
import tempfile
from fpdf import FPDF

# --- 1. POMOCNÉ FUNKCE ---
def remove_accents(input_str):
    if not input_str: return ""
    nfkd_form = unicodedata.normalize('NFKD', str(input_str))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def load_tmy_robust(file):
    try:
        content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
        header_idx = -1
        for i, line in enumerate(content):
            if 'T2m' in line: header_idx = i; break
        if header_idx == -1: return None
        df = pd.read_csv(io.StringIO("\n".join(content[header_idx:])))
        df.columns = df.columns.str.strip()
        # Ošetření chybějícího sloupce month
        if 'month' not in df.columns:
            if 'Month' in df.columns:
                df = df.rename(columns={'Month': 'month'})
            else:
                # Fallback: vytvoří měsíce pro 8760 řádků
                df['month'] = np.repeat(np.arange(1, 13), 730)[:len(df)]
        return df
    except: return None

def load_char(file):
    try:
        content = file.getvalue().decode('utf-8-sig', errors='ignore')
        sep = ';' if ';' in content.split('\n')[0] else ','
        df = pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
        df.columns = df.columns.str.strip()
        return df[['Teplota', 'Vykon_kW', 'COP']].copy()
    except: return None

# --- 2. KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="Energetický Simulátor TČ", layout="wide")
st.title("🚀 Profesionální simulátor kaskády TČ")

# --- 3. SIDEBAR (Kompletní vstupy) ---
st.sidebar.header("⚙️ Systémové parametry")
with st.sidebar:
    nazev_projektu = st.text_input("Název projektu", "SVJ Sladkovičova")
    ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
    t_design = st.number_input("Návrhová venkovní teplota [°C]", value=-12.0)
    st.markdown("---")
    st.markdown("### 🌡️ Otopná soustava")
    t_voda_max = st.number_input("TV_Max_Navrh (při -12°C) [°C]", value=60.0)
    t_voda_min = st.number_input("TV_Min_Navrh (při +15°C) [°C]", value=35.0)
    limit_voda_tc = st.number_input("Limit_Voda_TC (Max z TČ) [°C]", value=55.0)
    st.markdown("---")
    st.markdown("### 🚿 Příprava TUV")
    t_tuv_cil = st.number_input("Cílová teplota TUV [°C]", value=55.0)
    spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)
    st.markdown("---")
    spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 4)
    cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800.0)
    cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284.0)
    investice = st.number_input("Investice celkem [Kč]", value=3800000.0)

# --- 4. VÝPOČET ---
st.subheader("📁 1. Krok: Nahrání dat")
col_f1, col_f2 = st.columns(2)
with col_f1: tmy_file = st.file_uploader("Nahrajte TMY", type="csv")
with col_f2: char_file = st.file_uploader("Nahrajte Charakteristiku TČ", type="csv")

if tmy_file and char_file:
    tmy = load_tmy_robust(tmy_file)
    df_char = load_char(char_file)

    if tmy is not None and df_char is not None:
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce').fillna(0)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
        
        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        # Korekční faktor pro ÚT, aby seděla roční MWh
        potreba_ut_teorie = [ztrata * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
        k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000)

        res = []
        for _, row in tmy.iterrows():
            t_out, t_sm, m = row['T2m'], row['T_smooth'], int(row['month'])
            t_v_req = np.interp(t_sm, [t_design, 15], [t_voda_max, t_voda_min]) if t_sm < 20 else t_voda_min
            
            # Fyzikální korekce výkonu a COP
            k_p = 1 - (max(0, t_v_req - 35.0) * 0.01)
            k_cop_ut = 1 - (max(0, t_v_req - 35.0) * 0.025)
            k_cop_tuv = 1 - (max(0, t_tuv_cil - 35.0) * 0.025)
            
            q_need = (max(0, (ztrata * (20 - t_sm) / (20 - t_design))) * k_oprava) + q_tuv_avg
            p_max = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc * k_p
            
            # Rozhodnutí TČ vs Bivalence
            if t_v_req > limit_voda_tc or t_tuv_cil > limit_voda_tc:
                q_tc, q_biv = 0, q_need # TČ vypíná (bezpečnostní limit)
            else:
                q_tc = min(q_need, p_max)
                q_biv = q_need - q_tc
            
            cop_base = np.interp(t_out, df_char['Teplota'], df_char['COP'])
            # Vážený průměr COP pro ÚT a TUV
            el_tc = (q_tc * 0.7 / (cop_base * k_cop_ut)) + (q_tc * 0.3 / (cop_base * k_cop_tuv)) if q_tc > 0 else 0
            res.append([t_out, m, q_need, q_tc, q_biv, el_tc, q_biv/0.98])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Month', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])

        # Sumáře
        q_tc_s, q_biv_s = df_sim['Q_tc'].sum()/1000, df_sim['Q_biv'].sum()/1000
        el_tc_s, el_biv_s = df_sim['El_tc'].sum()/1000, df_sim['El_biv'].sum()/1000
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        naklady_tc = (el_tc_s + el_biv_s) * cena_el + 18000
        uspora = naklady_czt - naklady_tc
        scop = q_tc_s / el_tc_s if el_tc_s > 0 else 0

        # --- GRAFY (Příprava pro PDF i Web) ---
        plt.rcParams.update({'font.size': 8})
        
        # 1. Výkonová rovnováha
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        tx_plot = np.sort(df_sim['Temp'].unique())
        qy_plot = [max(0, (ztrata * (20 - t) / (20 - t_design) * k_oprava)) + q_tuv_avg for t in tx_plot]
        ax1.plot(tx_plot, qy_plot, 'r-', label='Potřeba objektu')
        ax1.set_title("Hodinová výkonová rovnováha"); ax1.legend(); ax1.grid(True, alpha=0.2)

        # 2. Energetické pokrytí
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        df_sort = df_sim.sort_values('Temp')
        ax2.fill_between(df_sort['Temp'], 0, df_sort['Q_tc'], color='#3498db', label='Energie TČ')
        ax2.fill_between(df_sort['Temp'], df_sort['Q_tc'], df_sort['Q_need'], color='#e74c3c', label='Bivalence')
        ax2.set_title("Krytí potřeby tepla"); ax2.legend()

        # 3. Měsíční graf
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        df_sim.groupby('Month')[['Q_tc', 'Q_biv']].sum().plot(kind='bar', stacked=True, ax=ax3, color=['#3498db', '#e74c3c'])
        ax3.set_title("Měsíční výroba tepla [kWh]"); ax3.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'], rotation=0)

        # 4. Ekonomika
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        bars = ax4.bar(['Stávající CZT', 'Nová kaskáda TČ'], [naklady_czt, naklady_tc], color=['#95a5a6', '#2ecc71'])
        for b in bars:
            ax4.text(b.get_x()+b.get_width()/2, b.get_height()+(naklady_czt*0.02), f'{int(b.get_height()):,} Kč'.replace(',',' '), ha='center', fontweight='bold')
        ax4.set_title("Srovnání ročních nákladů")

        # --- ZOBRAZENÍ ---
        t1, t2 = st.tabs(["📉 Energetická bilance", "💰 Ekonomika a Export"])
        with t1:
            st.pyplot(fig1); st.pyplot(fig2); st.pyplot(fig3)
        with t2:
            st.metric("Roční úspora", f"{int(uspora):,} Kč".replace(',',' '))
            st.pyplot(fig4)
            
            # --- PDF EXPORT (Kompletní) ---
            if st.button("📄 Generovat kompletní PDF report"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", 'B', 16)
                pdf.cell(190, 10, f"ANALYZNI REPORT: {remove_accents(nazev_projektu)}", ln=True, align='C')
                
                pdf.ln(10); pdf.set_font("Helvetica", 'B', 12); pdf.cell(190, 10, "1. VSTUPNI PARAMETRY", ln=True)
                pdf.set_font("Helvetica", '', 10)
                pdf.cell(95, 7, f"Tepelna ztrata: {ztrata} kW"); pdf.cell(95, 7, f"Navrhova venkovni teplota: {t_design} C", ln=True)
                pdf.cell(95, 7, f"Teplota vody (TV_Max/Min): {t_voda_max}/{t_voda_min} C"); pdf.cell(95, 7, f"Limit TC: {limit_voda_tc} C", ln=True)
                pdf.cell(95, 7, f"Cilova teplota TUV: {t_tuv_cil} C"); pdf.cell(95, 7, f"Pocet TC v kaskade: {pocet_tc} ks", ln=True)
                pdf.cell(95, 7, f"Rocni potreba UT: {spotreba_ut} MWh"); pdf.cell(95, 7, f"Rocni potreba TUV: {spotreba_tuv} MWh", ln=True)
                
                pdf.ln(5); pdf.set_font("Helvetica", 'B', 12); pdf.cell(190, 10, "2. VYSLEDKY ANALYZY", ln=True)
                pdf.set_font("Helvetica", '', 10)
                pdf.cell(190, 7, f"Rocni uspora nakladu: {int(uspora):,} Kc".replace(',',' '), ln=True)
                pdf.cell(190, 7, f"Prosta navratnost investice: {investice/uspora:.1f} let", ln=True)
                pdf.cell(190, 7, f"Celosezonni ucinnost (SCOP): {scop:.2f}", ln=True)
                pdf.cell(190, 7, f"Podil energie z TC: {(q_tc_s/(q_tc_s+q_biv_s)*100):.1f} %", ln=True)

                # Vložení všech 4 grafů na novou stranu
                pdf.add_page()
                for i, f in enumerate([fig1, fig2, fig3, fig4]):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        f.savefig(tmp.name, dpi=110)
                        pdf.image(tmp.name, x=15, y=None, w=170)
                        pdf.ln(2)
                
                st.download_button("⬇️ Stáhnout PDF", data=pdf.output(dest='S').encode('latin-1', 'replace'), file_name="Analyza_TC.pdf")
