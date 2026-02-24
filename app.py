import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import unicodedata
from fpdf import FPDF
import tempfile

# --- POMOCNÉ FUNKCE ---
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
        return pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
    except: return None

# --- KONFIGURACE ---
st.set_page_config(page_title="Simulator TC v3.6 - PROFESSIONAL", layout="wide")

with st.sidebar:
    st.header("⚙️ Konfigurace")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    
    with st.expander("🏠 Budova a potřeba", expanded=True):
        ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
        t_vnitrni = st.number_input("Žádaná vnitřní teplota [°C]", value=20.0)
        t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
        spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
        spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)

    with st.expander("🔧 Technologie", expanded=True):
        pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 4)
        eta_biv = st.slider("Účinnost bivalence [%]", 80, 100, 98) / 100

    with st.expander("💰 Ekonomika", expanded=True):
        investice = st.number_input("Investice celkem [Kč]", value=4080000)
        cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800)
        cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284)
        servis = st.number_input("Roční servis [Kč]", value=17500)

# --- VÝPOČTY ---
tmy_file = st.file_uploader("1. Nahrajte TMY (CSV)", type="csv")
char_file = st.file_uploader("2. Nahrajte Charakteristiku TČ (CSV)", type="csv")

if tmy_file and char_file:
    tmy = load_tmy_robust(tmy_file)
    df_char = load_char(char_file)

    if tmy is not None and df_char is not None:
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
        tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
        t_col, v_col, c_col = df_char.columns[0], df_char.columns[1], df_char.columns[2]

        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        potreba_ut_teorie = [ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) for t in tmy['T_smooth']]
        k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000)

        res = []
        for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
            q_need = max(0, (ztrata * (t_vnitrni - t_sm) / (t_vnitrni - t_design) * k_oprava)) + q_tuv_avg
            p_max = np.interp(t_out, df_char[t_col], df_char[v_col]) * pocet_tc
            cop_val = np.interp(t_out, df_char[t_col], df_char[c_col])
            q_tc = min(q_need, p_max)
            q_biv = max(0, q_need - q_tc)
            res.append([t_out, q_need, q_tc, q_biv, q_tc/cop_val if q_tc > 0 else 0, q_biv/eta_biv])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
        
        # Bod bivalence
        t_biv_val = -7.0
        for t in np.linspace(15, -15, 500):
            q_req = (ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava) + q_tuv_avg
            if (np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc) < q_req:
                t_biv_val = t
                break

        # Ekonomika
        naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
        mwh_el_total = (df_sim['El_tc'].sum() + df_sim['El_biv'].sum()) / 1000
        naklady_tc = (mwh_el_total * cena_el) + servis
        uspora = naklady_czt - naklady_tc
        navratnost = investice / uspora if uspora > 0 else 0

        # Tabulka
        q_tc_s, q_bv_s = df_sim['Q_tc'].sum()/1000, df_sim['Q_biv'].sum()/1000
        el_tc_s, el_bv_s = df_sim['El_tc'].sum()/1000, df_sim['El_biv'].sum()/1000
        df_biv_res = pd.DataFrame({
            "Metrika": ["Tepelná energie (Výstup)", "Spotřeba elektřiny (Vstup)"],
            "TČ [MWh]": [round(q_tc_s, 2), round(el_tc_s, 2)],
            "Bivalence [MWh]": [round(q_bv_s, 2), round(el_bv_s, 2)],
            "Podíl bivalence [%]": [round(q_bv_s/(q_tc_s+q_bv_s)*100, 1), round(el_bv_s/(el_tc_s+el_bv_s)*100, 1)]
        })

        # --- ZOBRAZENÍ GRAFŮ A POPISŮ ---
        st.header(f"📊 Projekt: {nazev_projektu}")
        
        # 1 a 2
        fig12, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        tr = np.linspace(-15, 18, 100)
        q_p = np.array([(ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava) + q_tuv_avg for t in tr])
        p_p = np.array([np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tr])
        ax1.plot(tr, q_p, 'r-', lw=2, label='Potřeba (ÚT+TUV)')
        ax1.plot(tr, p_p, 'b--', alpha=0.4, label='Max kaskáda TČ')
        ax1.fill_between(tr, p_p, q_p, where=(q_p > p_p), color='red', alpha=0.2, hatch='XXXX')
        ax1.set_title("1. DYNAMIKA PROVOZU"); ax1.set_xlabel("Venkovní teplota [°C]"); ax1.set_ylabel("Výkon [kW]"); ax1.legend()
        
        df_sim['Temp_R'] = df_sim['Temp'].round()
        df_t = df_sim.groupby('Temp_R')[['Q_tc', 'Q_biv']].sum()
        ax2.bar(df_t.index, df_t['Q_tc'], color='#3498db', label='Energie z TČ')
        ax2.bar(df_t.index, df_t['Q_biv'], bottom=df_t['Q_tc'], color='#e74c3c', label='Energie z bivalence')
        ax2.set_title("2. ENERGETICKÝ MIX"); ax2.set_xlabel("Venkovní teplota [°C]"); ax2.set_ylabel("Energie [kWh]"); ax2.legend()
        st.pyplot(fig12)
        st.caption("**Graf 1:** Zobrazuje bod bivalence (průsečík), kde TČ přestává stačit na plné krytí ztrát. **Graf 2:** Ukazuje, že i když bivalence pomáhá při mrazech, celkový objem energie (plocha) je dominantně kryt čerpadlem.")

        # 3 a 4
        fig34, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 7))
        df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1
        m_df = df_sim.groupby('Month').agg({'Q_tc': 'sum', 'Q_biv': 'sum'})
        ax3.bar(m_df.index, m_df['Q_tc']/1000, color='#ADD8E6', label='TČ')
        ax3.bar(m_df.index, m_df['Q_biv']/1000, bottom=m_df['Q_tc']/1000, color='#FF0000', label='Biv')
        ax3.set_title("3. MĚSÍČNÍ BILANCE"); ax3.set_xlabel("Měsíc v roce"); ax3.set_ylabel("Energie [MWh]"); ax3.legend()
        
        q_sort = np.sort(df_sim['Q_need'].values)[::-1]
        p_lim_biv = np.interp(t_biv_val, df_char[t_col], df_char[v_col]) * pocet_tc
        ax4.plot(range(8760), q_sort, 'r-', lw=2)
        ax4.fill_between(range(8760), 0, np.minimum(q_sort, p_lim_biv), color='#ADD8E6', label='Kryto TČ')
        ax4.fill_between(range(8760), p_lim_biv, q_sort, where=(q_sort > p_lim_biv), color='#FF0000', label='Bivalence')
        ax4.set_title("4. MONOTÓNA VÝKONU"); ax4.set_xlabel("Počet hodin [h]"); ax4.set_ylabel("Výkon [kW]"); ax4.legend()
        st.pyplot(fig34)
        st.caption("**Graf 3:** Měsíční potřeba tepla. Bivalence se typicky objevuje jen v prosinci až únoru. **Graf 4:** Plocha pod křivkou odpovídá celkové dodané energii. Červená plocha je minimální oproti bleděmodré.")

        # 5, 6, 7
        st.markdown("---")
        c_l, c_r = st.columns(2)
        with c_l:
            st.table(df_biv_res)
            fig6, ax6 = plt.subplots(figsize=(6, 6))
            ax6.pie([q_tc_s, q_bv_s], labels=['TČ', 'Biv'], autopct='%1.1f%%', colors=['#ADD8E6', '#FF0000'])
            ax6.set_title("PODÍL ENERGIE (VÝSTUP)")
            st.pyplot(fig6)
        with c_r:
            fig7, ax7 = plt.subplots(figsize=(6, 6))
            ax7.bar(['Původní CZT', 'Nové TČ'], [naklady_czt, naklady_tc], color=['#95a5a6', '#2ecc71'])
            ax7.set_ylabel("Provozní náklady [Kč/rok]"); ax7.set_title("EKONOMICKÉ SROVNÁNÍ")
            st.pyplot(fig7)
            st.success(f"Úspora: {uspora:,.0f} Kč/rok | Návratnost: {navratnost:.1f} let")

        # --- PDF GENERÁTOR ---
        def generate_pdf_pro():
            pdf = FPDF()
            pdf.add_page()
            def cz(txt): return txt.encode('cp1250', errors='replace').decode('latin1')
            
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, cz(f"EXPERTNI REPORT: {nazev_projektu.upper()}"), ln=True, align="C")
            
            pdf.set_font("Helvetica", "B", 11)
            pdf.ln(5); pdf.cell(0, 8, cz("1. TECHNICKÉ PARAMETRY A BIVALENCE"), ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, cz(f"- Tepelná ztráta objektu: {ztrata} kW"), ln=True)
            pdf.cell(0, 6, cz(f"- Bod bivalence (vypočtený): {t_biv_val:.1f} °C"), ln=True)
            pdf.cell(0, 6, cz(f"- Podíl bivalence na tepelné energii: {df_biv_res.iloc[0,3]} %"), ln=True)
            pdf.cell(0, 6, cz(f"- Podíl bivalence na spotřebě el.: {df_biv_res.iloc[1,3]} %"), ln=True)
            
            pdf.ln(4); pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 8, cz("2. EKONOMICKÉ HODNOCENÍ"), ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, cz(f"- Investice celkem: {investice:,.0f} Kč"), ln=True)
            pdf.cell(0, 6, cz(f"- Roční úspora nákladů: {uspora:,.0f} Kč"), ln=True)
            pdf.cell(0, 6, cz(f"- Prostá návratnost: {navratnost:.1f} let"), ln=True)

            # Vkládání grafů s popisem
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1:
                fig12.savefig(f1.name, dpi=110); pdf.image(f1.name, x=10, y=90, w=190)
            
            pdf.add_page()
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(0, 10, cz("Graf 1 a 2: Provozní charakteristika v závislosti na venkovní teplotě."), ln=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2:
                fig34.savefig(f2.name, dpi=110); pdf.image(f2.name, x=10, y=25, w=190)
            
            pdf.set_xy(10, 105)
            pdf.cell(0, 10, cz("Graf 3 a 4: Měsíční bilance a trvání výkonu (monotóna)."), ln=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f3:
                fig7.savefig(f3.name, dpi=110); pdf.image(f3.name, x=10, y=120, w=90)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f4:
                fig6.savefig(f4.name, dpi=110); pdf.image(f4.name, x=105, y=120, w=90)

            return bytes(pdf.output())

        st.sidebar.markdown("---")
        if st.sidebar.button("📄 Exportovat PDF s popisy"):
            st.sidebar.download_button("📥 Stáhnout PDF", generate_pdf_pro(), f"Report_{nazev_projektu}.pdf")
