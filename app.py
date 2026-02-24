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

# --- 3. SIDEBAR ---
st.sidebar.header("⚙️ Systémové parametry")
with st.sidebar:
    nazev_projektu = st.text_input("Název projektu", "SVJ Sladkovicova")
    ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
    t_design = st.number_input("Návrhová venkovní teplota [°C]", value=-12.0)
    st.markdown("---")
    t_voda_max = st.number_input("Teplota vody při návrhové t. [°C]", value=60.0)
    t_voda_min = st.number_input("Teplota vody při +15°C [°C]", value=35.0)
    st.markdown("---")
    spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
    spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 4)
    st.markdown("---")
    cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800.0)
    cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284.0)
    investice = st.number_input("Investice celkem [Kč]", value=3800000.0)

# --- 4. NAHRÁNÍ DAT A VÝPOČET ---
st.subheader("📁 1. Krok: Nahrání dat")
col_f1, col_f2 = st.columns(2)
with col_f1: tmy_file = st.file_uploader("Nahrajte TMY (teplotní data)", type="csv")
with col_f2: char_file = st.file_uploader("Nahrajte Charakteristiku TČ (vstupy_TC.csv)", type="csv")

if tmy_file and char_file:
    tmy_raw = load_tmy_robust(tmy_file)
    df_char_raw = load_char(char_file)

    if tmy_raw is not None and df_char_raw is not None:
        df_char = st.sidebar.data_editor(df_char_raw, num_rows="dynamic", hide_index=True)

        tmy = tmy_raw.copy()
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce').round(0)
        tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()

        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        potreba_ut_teorie = [ztrata * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
        k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000)

        res = []
        for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
            t_voda_req = np.interp(t_sm, [t_design, 15], [t_voda_max, t_voda_min]) if t_sm < 20 else t_voda_min
            k_p, k_cop = 1 - (max(0, t_voda_req - 35.0) * 0.01), 1 - (max(0, t_voda_req - 35.0) * 0.025)
            q_need = max(0, (ztrata * (20 - t_sm) / (20 - t_design) * k_oprava)) + q_tuv_avg
            p_real = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc * k_p
            cop_real = np.interp(t_out, df_char['Teplota'], df_char['COP']) * k_cop
            q_tc = min(q_need, p_real)
            q_biv = max(0, q_need - q_tc)
            res.append([t_out, q_need, q_tc, q_biv, q_tc/cop_real if q_tc > 0 else 0, q_biv/0.98])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
        
        # Roční sumáře
        q_tc_s, q_biv_s = df_sim['Q_tc'].sum()/1000, df_sim['Q_biv'].sum()/1000
        el_tc_s, el_biv_s = df_sim['El_tc'].sum()/1000, df_sim['El_biv'].sum()/1000
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        naklady_tc = (el_tc_s + el_biv_s) * cena_el + 17000
        uspora = naklady_czt - naklady_tc
        navratnost = investice / uspora if uspora > 0 else 0

        # --- PŘÍPRAVA GRAFŮ ---
        # 1. Výkonová rovnováha
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        tx = np.linspace(df_sim['Temp'].min(), 20, 50)
        qy = [max(0, (ztrata * (20 - t) / (20 - t_design) * k_oprava)) + q_tuv_avg for t in tx]
        py = [np.interp(t, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc * (1-(max(0,np.interp(t,[t_design,15],[t_voda_max,t_voda_min])-35)*0.01)) for t in tx]
        ax1.plot(tx, qy, 'r-', label='Potreba objektu', linewidth=2)
        ax1.plot(tx, py, 'b--', label='Vykon kaskady TC', linewidth=2)
        ax1.set_xlabel("Venkovni teplota [deg C]"); ax1.set_ylabel("Vykon [kW]"); ax1.grid(True, alpha=0.3); ax1.legend()

        # 2. Energetické pokrytí
        df_binned = df_sim.groupby('Temp').agg({'Q_tc':'sum', 'Q_biv':'sum'}).reset_index()
        df_binned[['Q_tc', 'Q_biv']] /= 1000
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.bar(df_binned['Temp'], df_binned['Q_tc'], color='#3498db', label='Energie z TC')
        ax2.bar(df_binned['Temp'], df_binned['Q_biv'], bottom=df_binned['Q_tc'], color='#e74c3c', label='Energie z bivalence')
        ax2.set_xlabel("Venkovni teplota [deg C]"); ax2.set_ylabel("Energie [MWh]"); ax2.legend()

        # 3. Pie chart bivalence
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        ax_pie.pie([el_tc_s, el_biv_s], labels=['TC', 'Biv'], autopct='%1.1f%%', colors=['#3498db','#e74c3c'], startangle=90)

        # 4. ROČNÍ NÁKLADY (Ten s popisky nad sloupci)
        fig_econ, ax_econ = plt.subplots(figsize=(10, 6))
        labels = ['Puvodni CZT', f'Nove TC ({pocet_tc}ks)']
        costs = [naklady_czt, naklady_tc]
        bars = ax_econ.bar(labels, costs, color=['#95a5a6', '#2ecc71'], width=0.6)
        ax_econ.set_ylabel("Naklady [Kc/rok]")
        ax_econ.set_title(f"ROCNI NAKLADY (SPAD {int(t_voda_max)} / {int(t_voda_min)} deg C)", fontweight='bold')
        # Přidání částek nad sloupce
        for bar in bars:
            height = bar.get_height()
            ax_econ.text(bar.get_x() + bar.get_width()/2., height + (max(costs)*0.02),
                        f'{int(height):,} Kc'.replace(',', ' '), ha='center', va='bottom', fontweight='bold', fontsize=12)
        plt.tight_layout()

        # --- ZOBRAZENÍ V APP ---
        tab1, tab2 = st.tabs(["📉 Výkonová a Energetická bilance", "💰 Ekonomika a přínosy"])
        with tab1:
            c1, c2 = st.columns(2)
            c1.pyplot(fig1); c2.pyplot(fig2)
            st.markdown("---")
            c3, c4 = st.columns([1, 2])
            with c3: st.pyplot(fig_pie)
            with c4:
                st.table(pd.DataFrame({
                    "Zdroj": ["Tepelna cerpadla", "Bivalence", "CELKEM"],
                    "Teplo [MWh]": [f"{q_tc_s:.1f}", f"{q_biv_s:.1f}", f"{q_tc_s+q_biv_s:.1f}"],
                    "Podil teplo": [f"{(q_tc_s/(q_tc_s+q_biv_s))*100:.1f}%", f"{(q_biv_s/(q_tc_s+q_biv_s))*100:.1f}%", "100%"],
                    "Elektrina [MWh]": [f"{el_tc_s:.1f}", f"{el_biv_s:.1f}", f"{el_tc_s+el_biv_s:.1f}"],
                    "Podil el.": [f"{(el_tc_s/(el_tc_s+el_biv_s))*100:.1f}%", f"{(el_biv_s/(el_tc_s+el_biv_s))*100:.1f}%", "100%"]
                }))

        with tab2:
            m1, m2, m3 = st.columns(3)
            m1.metric("Roční úspora", f"{int(uspora):,} Kč".replace(',', ' '))
            m2.metric("Návratnost", f"{navratnost:.1f} let")
            m3.metric("SCOP systému", f"{q_tc_s/el_tc_s:.2f}")
            st.pyplot(fig_econ)

        # --- PDF GENEROVÁNÍ (Bezpečné bez diakritiky) ---
        def generate_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(190, 10, f"EXPERTNI ANALYZA: {remove_accents(nazev_projektu)}", ln=True, align='C')
            
            pdf.ln(10)
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(190, 10, "1. VSTUPNI PARAMETRY", ln=True)
            pdf.set_font("Helvetica", '', 10)
            pdf.cell(190, 7, f"Tepelna ztrata: {ztrata} kW (pri {t_design} C)", ln=True)
            pdf.cell(190, 7, f"Teplotni spad: {t_voda_max}/{t_voda_min} C", ln=True)
            pdf.cell(190, 7, f"Pocet TC v kaskade: {pocet_tc} ks", ln=True)
            
            pdf.ln(5)
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(190, 10, "2. EKONOMICKY SUMAR", ln=True)
            pdf.set_font("Helvetica", '', 10)
            pdf.cell(190, 7, f"Rocni uspora: {int(uspora):,} Kc".replace(',', ' '), ln=True)
            pdf.cell(190, 7, f"Prosta navratnost: {navratnost:.1f} let", ln=True)
            pdf.cell(190, 7, f"SCOP systemu: {q_tc_s/el_tc_s:.2f}", ln=True)

            # Vložení grafů do PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t1:
                fig1.savefig(t1.name, format="png", dpi=120); pdf.image(t1.name, x=10, y=pdf.get_y()+5, w=90)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t2:
                fig2.savefig(t2.name, format="png", dpi=120); pdf.image(t2.name, x=105, y=pdf.get_y(), w=90)
            
            pdf.set_y(pdf.get_y() + 65)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t3:
                fig_econ.savefig(t3.name, format="png", dpi=120); pdf.image(t3.name, x=40, y=pdf.get_y(), w=130)
            
            return pdf.output()

        st.sidebar.markdown("---")
        if st.sidebar.button("📄 Připravit PDF Report"):
            try:
                pdf_out = generate_pdf()
                st.sidebar.download_button("⬇️ Stáhnout PDF", data=bytes(pdf_out), 
                                         file_name=f"Analýza_{remove_accents(nazev_projektu)}.pdf")
            except Exception as e:
                st.sidebar.error(f"Chyba PDF: {e}")

        # Excel export
        buf = io.BytesIO(); df_sim.to_excel(buf, index=False)
        st.download_button("📥 Stáhnout Excel data", buf.getvalue(), "vysledky_simulace.xlsx")
else:
    st.info("Nahrajte soubory (TMY a Charakteristiku) pro zahájení výpočtu.")
