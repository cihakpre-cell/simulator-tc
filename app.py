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
st.set_page_config(page_title="Simulator TC v3.9 - FINAL PROFESSIONAL", layout="wide")

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

        q_tc_s, q_bv_s = df_sim['Q_tc'].sum()/1000, df_sim['Q_biv'].sum()/1000
        el_tc_s, el_bv_s = df_sim['El_tc'].sum()/1000, df_sim['El_biv'].sum()/1000
        df_biv_table = pd.DataFrame({
            "Metrika": ["Tepelná energie (Výstup)", "Spotřeba elektřiny (Vstup)"],
            "TČ [MWh]": [round(q_tc_s, 2), round(el_tc_s, 2)],
            "Bivalence [MWh]": [round(q_bv_s, 2), round(el_bv_s, 2)],
            "Podíl bivalence [%]": [round(q_bv_s/(q_tc_s+q_bv_s)*100, 1), round(el_bv_s/(el_tc_s+el_bv_s)*100, 1)]
        })

        st.header(f"📊 Projekt: {nazev_projektu}")

        # --- GRAFY 1 a 2 ---
        fig12, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        tr = np.linspace(-15, 18, 100)
        q_p = np.array([(ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava) + q_tuv_avg for t in tr])
        p_p = np.array([np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tr])
        ax1.plot(tr, q_p, 'r-', lw=2, label='Potřeba (ÚT+TUV)')
        ax1.plot(tr, p_p, 'b--', alpha=0.4, label='Max kaskáda TČ')
        ax1.fill_between(tr, p_p, q_p, where=(q_p > p_p), color='red', alpha=0.2, hatch='XXXX', label='Oblast bivalence')
        ax1.axvline(t_biv_val, color='black', linestyle=':', lw=2, label=f'Bod bivalence: {t_biv_val:.1f}°C')
        ax1.set_title("1. DYNAMIKA PROVOZU"); ax1.set_xlabel("Venkovní teplota [°C]"); ax1.set_ylabel("Výkon [kW]"); ax1.legend()
        
        df_sim['Temp_R'] = df_sim['Temp'].round()
        df_t = df_sim.groupby('Temp_R')[['Q_tc', 'Q_biv']].sum()
        ax2.bar(df_t.index, df_t['Q_tc'], color='#3498db', label='Energie z TČ')
        ax2.bar(df_t.index, df_t['Q_biv'], bottom=df_t['Q_tc'], color='#e74c3c', label='Energie z bivalence')
        ax2.set_title("2. ENERGETICKÝ MIX DLE TEPLOT"); ax2.set_xlabel("Venkovní teplota [°C]"); ax2.set_ylabel("Energie [kWh]"); ax2.legend()
        st.pyplot(fig12)
        st.info("**Vysvětlení 1 a 2:** Graf vlevo ukazuje, při jaké venkovní teplotě přestává kaskáda TČ stačit (bod bivalence). Graf vpravo ukazuje objem dodané energie pro každou teplotu – i při mrazech TČ stále dodává většinu tepla.")

        # --- GRAFY 3 a 4 ---
        fig34, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 7))
        df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1
        m_df = df_sim.groupby('Month').agg({'Q_tc': 'sum', 'Q_biv': 'sum'})
        ax3.bar(m_df.index, m_df['Q_tc']/1000, color='#ADD8E6', label='TČ')
        ax3.bar(m_df.index, m_df['Q_biv']/1000, bottom=m_df['Q_tc']/1000, color='#FF0000', label='Biv')
        ax3.set_title("3. MĚSÍČNÍ BILANCE ENERGIE"); ax3.set_xlabel("Měsíc"); ax3.set_ylabel("Energie [MWh]"); ax3.legend()
        
        q_sort = np.sort(df_sim['Q_need'].values)[::-1]
        p_lim_biv = np.interp(t_biv_val, df_char[t_col], df_char[v_col]) * pocet_tc
        ax4.plot(range(8760), q_sort, 'r-', lw=2)
        ax4.fill_between(range(8760), 0, np.minimum(q_sort, p_lim_biv), color='#ADD8E6', label='Kryto TČ')
        ax4.fill_between(range(8760), p_lim_biv, q_sort, where=(q_sort > p_lim_biv), color='#FF0000', label='Bivalence')
        ax4.set_title("4. TRVÁNÍ POTŘEBY VÝKONU (MONOTÓNA)"); ax4.set_xlabel("Hodin v roce [h]"); ax4.set_ylabel("Výkon [kW]"); ax4.legend()
        st.pyplot(fig34)
        st.info("**Vysvětlení 3 a 4:** Měsíční graf ukazuje sezónnost – bivalence sepne jen v nejchladnějších měsících. Monotóna (vpravo) znázorňuje všech 8760 hodin roku seřazených dle výkonu; plocha jasně odděluje energii z TČ a bivalence.")

        # --- GRAF 5 ---
        fig5, ax5 = plt.subplots(figsize=(18, 5))
        df_st = df_sim.sort_values('Temp').reset_index(drop=True)
        ax5.plot(df_st.index, df_st['Q_need'], 'r', label='Potřeba ÚT+TUV')
        ax5.plot(df_st.index, df_st['Q_tc'], 'b', label='Krytí kaskádou TČ')
        ax5.set_title("5. ČETNOST TEPLOT V ROCE"); ax5.set_xlabel("Hodiny seřazené od nejmrazivějších"); ax5.set_ylabel("Výkon [kW]"); ax5.legend()
        st.pyplot(fig5)
        st.info("**Vysvětlení 5:** Tento graf zobrazuje potřebu tepla rozprostřenou podle skutečných hodin v roce seřazených od nejnižší teploty. Ukazuje stabilitu pokrytí kaskádou TČ.")

        # --- GRAFY 6 a 7 + TABULKA ---
        st.markdown("---")
        c_l, c_r = st.columns(2)
        with c_l:
            st.subheader("6. Bilance bivalence")
            st.table(df_biv_table)
            fig6, ax6 = plt.subplots(figsize=(6, 6))
            ax6.pie([q_tc_s, q_bv_s], labels=['TČ', 'Biv'], autopct='%1.1f%%', colors=['#ADD8E6', '#FF0000'])
            ax6.set_title("PODÍL DODANÉ ENERGIE (ROČNÍ)"); st.pyplot(fig6)
            st.info("**Vysvětlení 6:** Procentuální vyjádření práce kaskády. Cílem je, aby TČ krylo více než 95 % roční potřeby energie.")
        with c_r:
            st.subheader("7. Ekonomika")
            fig7, ax7 = plt.subplots(figsize=(6, 6))
            labels = ['Původní CZT', 'Nové řešení TČ']
            values = [naklady_czt, naklady_tc]
            bars = ax7.bar(labels, values, color=['#95a5a6', '#2ecc71'])
            ax7.set_title("SROVNÁNÍ PROVOZNÍCH NÁKLADŮ"); ax7.set_ylabel("Kč / rok")
            # Přidání hodnot nad sloupce
            for bar in bars:
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height + 5000, f'{int(height):,} Kč', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig7)
            st.info("**Vysvětlení 7:** Porovnání ročních plateb za teplo (původní CZT) oproti nové variantě (elektřina pro TČ + servis). Rozdíl tvoří roční úsporu.")

        # --- PDF GENERÁTOR ---
        def generate_pdf_v39():
            pdf = FPDF()
            def cz(txt): return txt.encode('cp1250', errors='replace').decode('latin1')
            
            # Strana 1
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, cz(f"TECHNICKÝ REPORT: {nazev_projektu.upper()}"), ln=True, align="C")
            pdf.ln(5); pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 8, cz("1. EKONOMICKÉ A TECHNICKÉ SHRNUTÍ"), ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, cz(f"- Bod bivalence: {t_biv_val:.1f} C"), ln=True)
            pdf.cell(0, 6, cz(f"- Roční úspora: {uspora:,.0f} Kč | Návratnost: {navratnost:.1f} let"), ln=True)
            pdf.ln(2); pdf.set_font("Helvetica", "B", 10); pdf.cell(0, 8, cz("Tabulka bilance bivalence:"), ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 5, cz(f"Energie (MWh): TC {df_biv_table.iloc[0,1]} | Biv {df_biv_table.iloc[0,2]} | Podíl bivalence: {df_biv_table.iloc[0,3]} %"), ln=True)
            pdf.cell(0, 5, cz(f"Elektřina (MWh): TC {df_biv_table.iloc[1,1]} | Biv {df_biv_table.iloc[1,2]} | Podíl bivalence: {df_biv_table.iloc[1,3]} %"), ln=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1:
                fig12.savefig(f1.name, dpi=100); pdf.image(f1.name, x=10, y=pdf.get_y()+5, w=190)
            pdf.set_xy(10, 165); pdf.set_font("Helvetica", "I", 8)
            pdf.cell(0, 5, cz("Graf 1 a 2: Dynamika výkonu a energetický mix v závislosti na venkovní teplotě."), ln=True)

            # Strana 2
            pdf.add_page()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2:
                fig34.savefig(f2.name, dpi=100); pdf.image(f2.name, x=10, y=10, w=190)
            pdf.set_xy(10, 105); pdf.cell(0, 5, cz("Graf 3 a 4: Měsíční bilance energie a monotóna trvání výkonu (rozdělení ploch TČ / bivalence)."), ln=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f5:
                fig5.savefig(f5.name, dpi=100); pdf.image(f5.name, x=10, y=115, w=190)
            pdf.set_xy(10, 165); pdf.cell(0, 5, cz("Graf 5: Četnost teplot v roce a schopnost pokrytí potřeb budovy kaskádou TČ."), ln=True)

            # Strana 3
            pdf.add_page()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f3:
                fig7.savefig(f3.name, dpi=100); pdf.image(f3.name, x=10, y=10, w=90)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f4:
                fig6.savefig(f4.name, dpi=100); pdf.image(f4.name, x=105, y=10, w=90)
            pdf.set_xy(10, 105); pdf.cell(0, 5, cz("Graf 6 a 7: Celkový podíl dodané energie a srovnání provozních nákladů CZT vs. TČ."), ln=True)
                
            return bytes(pdf.output())

        st.sidebar.markdown("---")
        if st.sidebar.button("🚀 Generovat PDF s popisy grafů"):
            st.sidebar.download_button("📥 Stáhnout PDF Report", generate_pdf_v39(), f"Report_{nazev_projektu}.pdf")
