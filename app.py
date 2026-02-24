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
        
        # Ignorování PVGIS hlavičky - hledáme start dat
        data_start_idx = -1
        for i, line in enumerate(content):
            if 'T2m' in line and 'time' in line:
                data_start_idx = i
                break
        
        if data_start_idx == -1: return None
            
        df = pd.read_csv(io.StringIO("\n".join(content[data_start_idx:])))
        df.columns = df.columns.str.strip()

        # Očištění od PVGIS patičky (legendy na konci)
        time_col = df.columns[0]
        df = df[df[time_col].apply(lambda x: str(x)[:4].isdigit() if pd.notnull(x) else False)].copy()

        # Extrakce měsíce z časového razítka PVGIS
        df['month'] = df[time_col].str[4:6].astype(int)
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

# --- 3. SIDEBAR (Všechny parametry) ---
st.sidebar.header("⚙️ Systémové parametry")
with st.sidebar:
    nazev_projektu = st.text_input("Název projektu", "SVJ Sladkovičova")
    ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
    t_design = st.number_input("Návrhová venkovní teplota [°C]", value=-12.0)
    
    st.markdown("### 🌡️ Otopná soustava")
    t_voda_max = st.number_input("TV_Max_Navrh (při -12°C) [°C]", value=60.0)
    t_voda_min = st.number_input("TV_Min_Navrh (při +15°C) [°C]", value=35.0)
    limit_voda_tc = st.number_input("Limit_Voda_TC (Max z TČ) [°C]", value=55.0)
    
    st.markdown("### 🚿 Příprava TUV")
    t_tuv_cil = st.number_input("Cílová teplota TUV [°C]", value=55.0)
    spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)
    
    st.markdown("### 💰 Provoz a Investice")
    spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 4)
    cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800.0)
    cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284.0)
    investice = st.number_input("Investice celkem [Kč]", value=3800000.0)

# --- 4. NAHRÁNÍ DAT ---
st.subheader("📁 1. Krok: Nahrání dat")
col_f1, col_f2 = st.columns(2)
with col_f1: tmy_file = st.file_uploader("Nahrajte TMY (CSV)", type="csv")
with col_f2: char_file = st.file_uploader("Nahrajte Charakteristiku TČ (CSV)", type="csv")

if tmy_file and char_file:
    tmy = load_tmy_robust(tmy_file)
    df_char = load_char(char_file)

    if tmy is not None and df_char is not None:
        # Výpočet bivalentních bodů a energetiky
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce').fillna(0)
        
        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        
        res = []
        for _, row in tmy.iterrows():
            t_out = row['T2m']
            m = int(row['month'])
            # Teplota topné vody dle křivky
            t_v_req = np.interp(t_out, [t_design, 15], [t_voda_max, t_voda_min]) if t_out < 20 else t_voda_min
            
            # Korekce COP
            k_cop_ut = 1 - (max(0, t_v_req - 35.0) * 0.025)
            k_cop_tuv = 1 - (max(0, t_tuv_cil - 35.0) * 0.025)
            k_p = 1 - (max(0, t_v_req - 35.0) * 0.01)

            q_need_ut = max(0, (ztrata * (20 - t_out) / (20 - t_design)))
            p_max = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc * k_p
            
            q_total = q_need_ut + q_tuv_avg
            
            if t_v_req > limit_voda_tc or t_tuv_cil > limit_voda_tc:
                q_tc = 0 
            else:
                q_tc = min(q_total, p_max)
            
            q_biv = max(0, q_total - q_tc)
            
            cop_base = np.interp(t_out, df_char['Teplota'], df_char['COP'])
            el_tc = (q_tc * 0.7 / (cop_base * k_cop_ut)) + (q_tc * 0.3 / (cop_base * k_cop_tuv)) if q_tc > 0 else 0
            
            res.append([t_out, m, q_total, q_tc, q_biv, el_tc, q_biv/0.98])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Month', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
        
        # Ekonomické sumáře
        q_tc_s, q_biv_s = df_sim['Q_tc'].sum()/1000, df_sim['Q_biv'].sum()/1000
        el_tc_s, el_biv_s = df_sim['El_tc'].sum()/1000, df_sim['El_biv'].sum()/1000
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        naklady_tc = (el_tc_s + el_biv_s) * cena_el + 18000
        uspora = naklady_czt - naklady_tc
        scop = q_tc_s / el_tc_s if el_tc_s > 0 else 0

        # --- GRAFY (Původní odsouhlasený vizuál) ---
        # 1. Výkonová rovnováha
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        tx = np.sort(df_sim['Temp'].unique())
        qy = [max(0, (ztrata * (20 - t) / (20 - t_design))) + q_tuv_avg for t in tx]
        ax1.plot(tx, qy, 'r-', label='Potřeba objektu')
        ax1.set_title("Hodinová výkonová rovnováha"); ax1.legend(); ax1.grid(True, alpha=0.3)

        # 2. Energetické pokrytí
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        df_sorted = df_sim.sort_values('Temp')
        ax2.fill_between(df_sorted['Temp'], 0, df_sorted['Q_tc'], color='#3498db', label='TČ')
        ax2.fill_between(df_sorted['Temp'], df_sorted['Q_tc'], df_sorted['Q_need'], color='#e74c3c', label='Biv')
        ax2.set_title("Krytí potřeby dle teploty"); ax2.legend()

        # 3. Měsíční bilance
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        df_sim.groupby('Month')[['Q_tc', 'Q_biv']].sum().plot(kind='bar', stacked=True, ax=ax3, color=['#3498db', '#e74c3c'])
        ax3.set_title("Měsíční výroba tepla [kWh]")

        # 4. Ekonomika
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        bars = ax4.bar(['Stávající CZT', 'Nová kaskáda TČ'], [naklady_czt, naklady_tc], color=['#95a5a6', '#2ecc71'])
        for b in bars:
            ax4.text(b.get_x()+b.get_width()/2, b.get_height()+10000, f'{int(b.get_height()):,} Kč'.replace(',',' '), ha='center', fontweight='bold')
        ax4.set_title("Srovnání ročních nákladů")

        # --- ZOBRAZENÍ V TABS ---
        tab1, tab2 = st.tabs(["📉 Energetická bilance", "💰 Ekonomika a Export"])
        with tab1:
            st.pyplot(fig1); st.pyplot(fig2); st.pyplot(fig3)
        with tab2:
            st.metric("Roční úspora", f"{int(uspora):,} Kč".replace(',',' '))
            st.pyplot(fig4)
            
            # --- KOMPLETNÍ PDF EXPORT (Původní odsouhlasený) ---
            if st.button("📄 Generovat kompletní report"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", 'B', 16)
                pdf.cell(190, 10, f"REPORT: {remove_accents(nazev_projektu)}", ln=True, align='C')
                
                pdf.ln(10); pdf.set_font("Helvetica", 'B', 12); pdf.cell(190, 10, "1. Vstupni parametry", ln=True)
                pdf.set_font("Helvetica", '', 10)
                pdf.cell(90, 7, f"Tepelna ztrata: {ztrata} kW"); pdf.cell(90, 7, f"Navrhova teplota: {t_design} C", ln=True)
                pdf.cell(90, 7, f"Spad (max/min): {t_voda_max}/{t_voda_min} C"); pdf.cell(90, 7, f"Limit TC: {limit_voda_tc} C", ln=True)
                pdf.cell(90, 7, f"Teplota TUV: {t_tuv_cil} C"); pdf.cell(90, 7, f"Pocet TC: {pocet_tc} ks", ln=True)
                
                pdf.ln(5); pdf.set_font("Helvetica", 'B', 12); pdf.cell(190, 10, "2. Vysledky simulace", ln=True)
                pdf.set_font("Helvetica", '', 10)
                pdf.cell(90, 7, f"Rocni uspora: {int(uspora):,} Kc".replace(',',' ')); pdf.cell(90, 7, f"Navratnost: {investice/uspora:.1f} let", ln=True)
                pdf.cell(90, 7, f"SCOP systemu: {scop:.2f}"); pdf.cell(90, 7, f"Podil bivalence: {(q_biv_s/(q_tc_s+q_biv_s)*100):.1f} %", ln=True)

                # Vložení všech 4 grafů
                pdf.add_page()
                for i, f in enumerate([fig1, fig2, fig3, fig4]):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        f.savefig(tmp.name, dpi=120)
                        pdf.image(tmp.name, x=15, y=None, w=160)
                        pdf.ln(5)
                
                st.download_button("⬇️ Stáhnout PDF report", data=pdf.output(dest='S').encode('latin-1', 'replace'), file_name="Report_TC.pdf")
