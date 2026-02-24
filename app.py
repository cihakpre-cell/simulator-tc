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

# --- 2. KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="Energetický Simulátor TČ", layout="wide")
st.title("🚀 Profesionální simulátor kaskády TČ")

# --- 3. SIDEBAR (Parametry dle vašich obrázků) ---
st.sidebar.header("⚙️ Systémové parametry")
with st.sidebar:
    nazev_projektu = st.text_input("Název projektu", "SVJ Sladkovicova")
    ztrata = st.number_input("Tepelná ztráta objektu [kW]", value=54.0)
    t_design = st.number_input("Návrhová venkovní teplota [°C]", value=-12.0)
    
    st.markdown("### 🌡️ Otopná soustava")
    t_privod = st.number_input("Návrhová teplota přívodu (TV_Max_Navrh) [°C]", value=60.0)
    t_zpatecka = st.number_input("Návrhová teplota zpátečky [°C]", value=50.0)
    t_min_voda = st.number_input("Teplota vody při +15°C (TV_Min_Navrh) [°C]", value=35.0)
    limit_voda_tc = st.number_input("Max. teplota z TČ (Limit_Voda_TC) [°C]", value=55.0)
    
    st.markdown("### 🚿 Příprava TUV")
    t_tuv_cilova = st.number_input("Cílová teplota TUV [°C]", value=55.0)
    spotreba_tuv = st.number_input("Roční potřeba pro TUV [MWh/rok]", value=76.0)
    
    st.markdown("### 🏭 Ekonomika a Provoz")
    spotreba_ut = st.number_input("Roční potřeba pro ÚT [MWh/rok]", value=124.0)
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 4)
    cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800.0)
    cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284.0)
    investice = st.number_input("Investice [Kč]", value=3800000.0)

# --- 4. NAHRÁNÍ SOUBORŮ ---
st.subheader("📁 1. Krok: Nahrání datových podkladů")
col_f1, col_f2 = st.columns(2)
with col_f1:
    tmy_file = st.file_uploader("1. Nahrajte TMY (meteorologická data)", type="csv")
with col_f2:
    char_file = st.file_uploader("2. Nahrajte Charakteristiku TČ (vstupy_TC.csv)", type="csv")

# --- 5. VÝPOČET A ZOBRAZENÍ ---
if tmy_file and char_file:
    tmy_raw = load_tmy_robust(tmy_file)
    df_char = load_char(char_file)

    if tmy_raw is not None and df_char is not None:
        tmy = tmy_raw.copy()
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce').fillna(0)
        
        # Výpočet potřeb a simulace
        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        res = []
        for t_out in tmy['T2m']:
            # Ekvitermní křivka
            t_voda_req = np.interp(t_out, [t_design, 15], [t_privod, t_min_voda]) if t_out < 20 else t_min_voda
            
            # Korekce výkonu TČ podle teploty vody
            k_p = 1 - (max(0, t_voda_req - 35.0) * 0.01)
            k_cop = 1 - (max(0, t_voda_req - 35.0) * 0.025)
            
            q_need = max(0, (ztrata * (20 - t_out) / (20 - t_design))) + q_tuv_avg
            p_max = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc * k_p
            
            # Omezení teplotou vody (pokud TČ neumí vyrobit víc než limit_voda_tc)
            if t_voda_req > limit_voda_tc:
                q_tc = 0 # TČ vypíná, vše bere bivalence (zjednodušený model)
            else:
                q_tc = min(q_need, p_max)
            
            q_biv = max(0, q_need - q_tc)
            cop = np.interp(t_out, df_char['Teplota'], df_char['COP']) * k_cop
            
            res.append([q_tc, q_biv, q_tc/cop if q_tc > 0 else 0, q_biv/0.98])

        df_sim = pd.DataFrame(res, columns=['Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
        
        # Ekonomika
        q_tc_s, q_biv_s = df_sim['Q_tc'].sum()/1000, df_sim['Q_biv'].sum()/1000
        el_tc_s, el_biv_s = df_sim['El_tc'].sum()/1000, df_sim['El_biv'].sum()/1000
        
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        naklady_tc = (el_tc_s + el_biv_s) * cena_el + 15000 # + paušál servis
        uspora = naklady_czt - naklady_tc
        navratnost = investice / uspora if uspora > 0 else 0

        # --- GRAFY ---
        # Graf Roční náklady
        fig_econ, ax_econ = plt.subplots(figsize=(10, 6))
        bars = ax_econ.bar(['Puvodni CZT', f'Nove TC ({pocet_tc}ks)'], [naklady_czt, naklady_tc], color=['#95a5a6', '#2ecc71'])
        ax_econ.set_title(f"ROCNI NAKLADY (SPAD {int(t_privod)}/{int(t_zpatecka)} deg C)", fontweight='bold')
        for bar in bars:
            h = bar.get_height()
            ax_econ.text(bar.get_x()+bar.get_width()/2, h + 10000, f'{int(h):,} Kc'.replace(',',' '), ha='center', fontweight='bold')
        
        # Zobrazení výsledků
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Roční úspora", f"{int(uspora):,} Kč".replace(',',' '))
        m2.metric("Návratnost", f"{navratnost:.1f} let")
        m3.metric("SCOP systému", f"{q_tc_s/el_tc_s:.2f}")
        
        st.pyplot(fig_econ)

        # --- PDF GENERÁTOR ---
        def create_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(190, 10, f"ANALYZA: {remove_accents(nazev_projektu)}", ln=True, align='C')
            pdf.set_font("Helvetica", '', 10)
            pdf.ln(5)
            pdf.cell(190, 7, f"Zadani: Ztrata {ztrata}kW, Spad {t_privod}/{t_zpatecka}C, TUV {t_tuv_cilova}C", ln=True)
            pdf.cell(190, 7, f"Vysledek: Uspora {int(uspora):,} Kc/rok, Navratnost {navratnost:.1f} let".replace(',',' '), ln=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                fig_econ.savefig(tmp.name, dpi=120)
                pdf.image(tmp.name, x=30, y=pdf.get_y()+10, w=140)
            return pdf.output()

        if st.sidebar.button("📄 Generovat PDF"):
            pdf_bytes = create_pdf()
            st.sidebar.download_button("⬇️ Stáhnout Report", data=bytes(pdf_bytes), file_name="Report.pdf")

else:
    st.warning("⚠️ Prosím nahrajte oba soubory v kroku 1 pro spuštění výpočtu.")
