import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import tempfile
from fpdf import FPDF

# --- LOGIKA NAČÍTÁNÍ (PVGIS SAFE) ---
def load_tmy_pvgis(file):
    try:
        content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
        start = -1
        for i, line in enumerate(content):
            if 'T2m' in line and 'time' in line:
                start = i
                break
        if start == -1: return None
        df = pd.read_csv(io.StringIO("\n".join(content[start:])))
        df.columns = df.columns.str.strip()
        df = df[df[df.columns[0]].apply(lambda x: str(x)[:4].isdigit() if pd.notnull(x) else False)].copy()
        df['month'] = df[df.columns[0]].str[4:6].astype(int)
        df['T2m'] = pd.to_numeric(df['T2m'], errors='coerce')
        return df
    except: return None

def load_char(file):
    try:
        content = file.getvalue().decode('utf-8-sig', errors='ignore')
        sep = ';' if ';' in content.split('\n')[0] else ','
        df = pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
        return df[['Teplota', 'Vykon_kW', 'COP']].copy()
    except: return None

# --- STREAMLIT UI ---
st.set_page_config(page_title="Simulátor Kaskády TČ", layout="wide")
st.title("📊 Energetická analýza kaskády tepelných čerpadel")

with st.sidebar:
    st.header("Vstupní data")
    nazev = st.text_input("Projekt", "Analýza Sladkovičova")
    ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
    t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
    t_voda_max = st.number_input("Teplota vody max [°C]", value=60.0)
    t_voda_min = st.number_input("Teplota vody min [°C]", value=35.0)
    t_tuv = st.number_input("Teplota TUV [°C]", value=55.0)
    limit_tc = st.number_input("Limit TČ [°C]", value=55.0)
    spotreba_ut = st.number_input("Potřeba ÚT [MWh]", value=124.0)
    spotreba_tuv = st.number_input("Potřeba TUV [MWh]", value=76.0)
    pocet_tc = st.slider("Počet TČ", 1, 10, 4)
    cena_el = st.number_input("Elektřina [Kč/MWh]", value=4800.0)
    cena_czt = st.number_input("CZT [Kč/GJ]", value=1284.0)

tmy_up = st.file_uploader("TMY soubor (PVGIS)", type="csv")
char_up = st.file_uploader("Charakteristika TČ", type="csv")

if tmy_up and char_up:
    tmy = load_tmy_pvgis(tmy_up)
    char = load_char(char_up)

    if tmy is not None and char is not None:
        # VÝPOČET
        q_tuv_h = (spotreba_tuv / 8760) * 1000
        data = []
        for _, row in tmy.iterrows():
            t_out = row['T2m']
            m = row['month']
            t_voda = np.interp(t_out, [t_design, 15], [t_voda_max, t_voda_min]) if t_out < 20 else t_voda_min
            
            q_ut = max(0, (ztrata * (20 - t_out) / (20 - t_design)))
            q_total = q_ut + q_tuv_h
            
            p_tc = np.interp(t_out, char['Teplota'], char['Vykon_kW']) * pocet_tc * (1 - (max(0, t_voda - 35) * 0.01))
            
            if t_voda > limit_tc:
                q_tc = 0
            else:
                q_tc = min(q_total, p_tc)
            
            q_biv = q_total - q_tc
            cop = np.interp(t_out, char['Teplota'], char['COP']) * (1 - (max(0, t_voda - 35) * 0.025))
            el = (q_tc / cop) if q_tc > 0 else 0
            data.append([t_out, m, q_total, q_tc, q_biv, el])
            
        df = pd.DataFrame(data, columns=['T', 'M', 'Q_need', 'Q_tc', 'Q_biv', 'El'])

        # --- GENEROVÁNÍ 5 GRAFŮ ---
        col1, col2 = st.columns(2)
        figs = []

        # 1. Čára trvání teplot
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(range(8760), sorted(df['T'], reverse=True), color='blue')
        ax1.set_title("1. Čára trvání venkovních teplot"); ax1.set_ylabel("Teplota [°C]"); ax1.grid(True, alpha=0.3)
        figs.append(fig1); col1.pyplot(fig1)

        # 2. Potřeba tepla vs Teplota
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        t_rng = np.linspace(t_design, 20, 100)
        q_rng = [max(0, (ztrata * (20 - t) / (20 - t_design)) + q_tuv_h) for t in t_rng]
        ax2.plot(t_rng, q_rng, color='red', label='Potřeba (ÚT+TUV)')
        ax2.set_title("2. Potřeba tepla objektu"); ax2.set_xlabel("Venkovní teplota [°C]"); ax2.legend(); ax2.grid(True, alpha=0.3)
        figs.append(fig2); col2.pyplot(fig2)

        # 3. Pokrytí potřeby (Monotonní)
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        df_sorted = df.sort_values('Q_need', ascending=False).reset_index()
        ax3.fill_between(df_sorted.index, 0, df_sorted['Q_tc'], color='#3498db', label='TČ')
        ax3.fill_between(df_sorted.index, df_sorted['Q_tc'], df_sorted['Q_need'], color='#e74c3c', label='Bivalence')
        ax3.set_title("3. Pokrytí potřeby energie (trvání)"); ax3.legend(); ax3.grid(True, alpha=0.3)
        figs.append(fig3); col1.pyplot(fig3)

        # 4. Měsíční bilance
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        df.groupby('M')[['Q_tc', 'Q_biv']].sum().plot(kind='bar', stacked=True, ax=ax4, color=['#3498db', '#e74c3c'])
        ax4.set_title("4. Měsíční výroba tepla [kWh]"); ax4.set_xticklabels(range(1,13), rotation=0)
        figs.append(fig4); col2.pyplot(fig4)

        # 5. Ekonomika
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        cost_czt = (spotreba_ut + spotreba_tuv) * cena_czt * 3.6
        cost_tc = (df['El'].sum()/1000 + df['Q_biv'].sum()/1000/0.98) * cena_el + 18000
        bars = ax5.bar(['CZT', 'TČ'], [cost_czt, cost_tc], color=['#95a5a6', '#2ecc71'])
        for b in bars:
            ax5.text(b.get_x()+b.get_width()/2, b.get_height(), f'{int(b.get_height()):,} Kč'.replace(',',' '), ha='center', va='bottom', fontweight='bold')
        ax5.set_title("5. Roční provozní náklady"); figs.append(fig5); st.pyplot(fig5)

        # --- PDF EXPORT ---
        if st.button("📄 Exportovat PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16); pdf.cell(190, 10, f"PROJEKT: {nazev}", ln=True, align='C')
            
            # Parametry
            pdf.set_font("Helvetica", '', 10); pdf.ln(5)
            pdf.cell(90, 7, f"Ztrata: {ztrata} kW | Design: {t_design} C")
            pdf.cell(90, 7, f"Uspora: {int(cost_czt-cost_tc):,} Kc", ln=True)
            
            for f in figs:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    f.savefig(tmp.name, dpi=120)
                    if pdf.get_y() > 220: pdf.add_page()
                    pdf.image(tmp.name, x=25, w=160); pdf.ln(5)
            
            st.download_button("⬇️ Stáhnout PDF", data=pdf.output(dest='S').encode('latin-1', 'replace'), file_name="Report.pdf")
