import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import urllib.request
import unicodedata  # FIX CHYBY NameError
from fpdf import FPDF
import tempfile

# --- STAŽENÍ FONTU PRO ČEŠTINU V PDF ---
FONT_REGULAR = "DejaVuSans.ttf"
FONT_BOLD = "DejaVuSans-Bold.ttf"

def download_fonts():
    url_reg = "https://raw.githubusercontent.com/dejavu-fonts/dejavu-fonts/master/ttf/DejaVuSans.ttf"
    url_bold = "https://raw.githubusercontent.com/dejavu-fonts/dejavu-fonts/master/ttf/DejaVuSans-Bold.ttf"
    if not os.path.exists(FONT_REGULAR):
        try: urllib.request.urlretrieve(url_reg, FONT_REGULAR)
        except: pass
    if not os.path.exists(FONT_BOLD):
        try: urllib.request.urlretrieve(url_bold, FONT_BOLD)
        except: pass

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
st.set_page_config(page_title="Simulator TC v4.4 - EKVITERMA & PDF FIX", layout="wide")
download_fonts() # Příprava fontů pro PDF

with st.sidebar:
    st.header("⚙️ Konfigurace")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    nazev_tc = st.text_input("Model tepelného čerpadla", "NIBE S2125-12")
    
    with st.expander("🏠 Budova a potřeba", expanded=True):
        ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
        t_vnitrni = st.number_input("Žádaná vnitřní teplota [°C]", value=20.0)
        t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
        t_spad = st.text_input("Teplotní spád soustavy [°C]", "55/45")
        t_tuv_cil = st.number_input("Teplota TUV [°C]", value=55.0)
        spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
        spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)

    with st.expander("🔧 Technologie & Charakteristika", expanded=True):
        pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 4)
        eta_biv = st.slider("Účinnost bivalence [%]", 80, 100, 98) / 100
        
        # Nahrání CSV pro TČ
        char_file = st.file_uploader("Nahrát CSV charakteristiku TČ", type="csv")
        if char_file:
            df_char_raw = load_char(char_file)
        else:
            df_char_raw = pd.DataFrame({
                "Teplota [°C]": [-15, -7, 2, 7, 15],
                "Výkon [kW]": [7.5, 9.2, 11.5, 12.0, 13.5],
                "COP [-]": [2.1, 2.8, 3.5, 4.2, 5.1]
            })
        
        st.write("Hodnoty charakteristiky (možno editovat):")
        df_char = st.data_editor(df_char_raw, num_rows="dynamic")

    with st.expander("💰 Ekonomika", expanded=True):
        investice = st.number_input("Investice celkem [Kč]", value=4080000)
        cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800)
        cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284)
        servis = st.number_input("Roční servis [Kč]", value=17500)

# --- VÝPOČTY (NOVÁ METODIKA S EKVITERMOU) ---
tmy_file = st.file_uploader("Nahrát TMY data (venkovní teploty)", type="csv")

if tmy_file:
    tmy = load_tmy_robust(tmy_file)
    if tmy is not None and df_char is not None:
        tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
        tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
        # B. Tepelná setrvačnost - klouzavý průměr
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
        
        t_col, v_col, c_col = df_char.columns[0], df_char.columns[1], df_char.columns[2]

        q_tuv_avg = (spotreba_tuv / 8760) * 1000
        potreba_ut_teorie = [max(0, ztrata * (t_vnitrni - t) / (t_vnitrni - t_design)) for t in tmy['T_smooth']]
        # A. Kalibrace budovy
        k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000) if sum(potreba_ut_teorie) > 0 else 1.0

        # C. Ekvitermní příprava (parsování zadaného spádu, např. "55/45")
        try:
            t_water_max = float(t_spad.split('/')[0])
        except:
            t_water_max = 55.0

        res = []
        for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
            # Potřeba ÚT a TUV
            q_ut = max(0, (ztrata * (t_vnitrni - t_sm) / (t_vnitrni - t_design) * k_oprava))
            q_need = q_ut + q_tuv_avg
            
            p_max = np.interp(t_out, df_char[t_col], df_char[v_col]) * pocet_tc
            cop_base = np.interp(t_out, df_char[t_col], df_char[c_col])
            
            # Výpočet dynamické teploty topné vody (ekvitermní křivka, min 25 °C)
            if t_out <= t_design:
                t_water_actual = t_water_max
            elif t_out >= t_vnitrni:
                t_water_actual = 25.0
            else:
                t_water_actual = 25.0 + (t_water_max - 25.0) * ((t_vnitrni - t_out) / (t_vnitrni - t_design))
            
            # C. Zvýšení COP (cca +2.5% za každý 1 °C snížení teploty topné vody)
            cop_ut = cop_base * (1 + 0.025 * max(0, t_water_max - t_water_actual))
            cop_tuv = cop_base # TUV se ohřívá vždy naplno
            
            q_tc = min(q_need, p_max)
            q_biv = max(0, q_need - q_tc)
            
            # Rozdělení výkonu a el. práce (Priorita TUV)
            q_tc_tuv = min(q_tc, q_tuv_avg)
            q_tc_ut = q_tc - q_tc_tuv
            
            el_tc = 0
            if cop_tuv > 0: el_tc += q_tc_tuv / cop_tuv
            if cop_ut > 0:  el_tc += q_tc_ut / cop_ut
            el_biv = q_biv / eta_biv if eta_biv > 0 else 0
            
            res.append([t_out, q_need, q_tc, q_biv, el_tc, el_biv])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
        
        # Bod bivalence (D.)
        t_biv_val = -12.0
        for t in np.linspace(15, -15, 500):
            q_req = max(0, (ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava)) + q_tuv_avg
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
            "Podíl bivalence [%]": [round(q_bv_s/(q_tc_s+q_bv_s)*100, 1) if (q_tc_s+q_bv_s)>0 else 0, 
                                    round(el_bv_s/(el_tc_s+el_bv_s)*100, 1) if (el_tc_s+el_bv_s)>0 else 0]
        })

        expl_12 = "Graf 1 a 2: Bod bivalence určuje venkovní teplotu, pod kterou musí kaskádě TČ pomáhat bivalentní zdroj. Energetický mix ukazuje, že i v mrazech TČ kryje drtivou většinu energie."
        expl_34 = "Graf 3 a 4: Měsíční bilance ukazuje sezónní využití zdrojů. Monotóna výkonu (vpravo) vizualizuje časové rozložení potřeby tepla a jasně odděluje práci kaskády od bivalence."
        expl_5 = "Graf 5: Četnost teplot v roce seřazená od nejnižších. Znázorňuje stabilitu a schopnost kaskády TČ pokrývat potřeby budovy v reálném čase."
        expl_67 = "Graf 6 a 7: Roční podíl energie potvrzuje efektivitu kaskády. Ekonomické srovnání ukazuje přímou úsporu v provozních nákladech oproti původnímu CZT."

        st.header(f"📊 Projekt: {nazev_projektu} ({nazev_tc})")

        # --- GENERÁTOR GRAFŮ ---
        fig12, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        tr = np.linspace(-15, 18, 100)
        q_p = np.array([max(0, (ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava)) + q_tuv_avg for t in tr])
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
        st.pyplot(fig12); st.info(expl_12)

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
        st.pyplot(fig34); st.info(expl_34)

        fig5, ax5 = plt.subplots(figsize=(18, 5))
        df_st = df_sim.sort_values('Temp').reset_index(drop=True)
        ax5.plot(df_st.index, df_st['Q_need'], 'r', label='Potřeba ÚT+TUV')
        ax5.plot(df_st.index, df_st['Q_tc'], 'b', label='Krytí kaskádou TČ')
        ax5.set_title("5. ČETNOST TEPLOT V ROCE"); ax5.set_xlabel("Hodiny seřazené od nejmrazivějších"); ax5.set_ylabel("Výkon [kW]"); ax5.legend()
        st.pyplot(fig5); st.info(expl_5)

        c_l, c_r = st.columns(2)
        with c_l:
            st.subheader("6. Bilance bivalence"); st.table(df_biv_table)
            fig6, ax6 = plt.subplots(figsize=(6, 6))
            ax6.pie([q_tc_s, q_bv_s], labels=['TČ', 'Biv'], autopct='%1.1f%%', colors=['#ADD8E6', '#FF0000'])
            ax6.set_title("PODÍL DODANÉ ENERGIE (ROČNÍ)"); st.pyplot(fig6)
        with c_r:
            st.subheader("7. Ekonomika")
            fig7, ax7 = plt.subplots(figsize=(6, 6))
            labels = ['Původní CZT', 'Nové řešení TČ']
            values = [naklady_czt, naklady_tc]
            bars = ax7.bar(labels, values, color=['#95a5a6', '#2ecc71'])
            ax7.set_title("SROVNÁNÍ PROVOZNÍCH NÁKLADŮ"); ax7.set_ylabel("Kč / rok")
            for bar in bars:
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height + 5000, f'{int(height):,} Kč', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig7); st.info(expl_67)

        # --- PDF GENERÁTOR (FIX CHYBY) ---
        def generate_pdf_v44():
            pdf = FPDF()
            
            # Bezpečné nakládání s kódováním
            has_unicode_font = os.path.exists(FONT_REGULAR) and os.path.exists(FONT_BOLD)
            if has_unicode_font:
                try:
                    pdf.add_font("DejaVu", "", FONT_REGULAR, uni=True)
                    pdf.add_font("DejaVu", "B", FONT_BOLD, uni=True)
                except:
                    has_unicode_font = False
            
            def cz(txt):
                if has_unicode_font: return str(txt)
                # Ošetření přes unicodedata - nyní bezpečné, protože import je nahoře
                return "".join([c for c in unicodedata.normalize('NFKD', str(txt)) if not unicodedata.combining(c)])

            pdf.add_page()
            
            if has_unicode_font: pdf.set_font("DejaVu", "B", 16)
            else: pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, cz(f"TECHNICKÝ REPORT: {nazev_projektu.upper()}"), ln=True, align="C")
            
            if has_unicode_font: pdf.set_font("DejaVu", "B", 12)
            else: pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, cz(f"Model TČ: {nazev_tc}"), ln=True, align="C")
            
            pdf.ln(5); 
            if has_unicode_font: pdf.set_font("DejaVu", "B", 11)
            else: pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, cz("1. VSTUPNÍ PARAMETRY ZADÁNÍ"), ln=True)
            
            if has_unicode_font: pdf.set_font("DejaVu", "", 10)
            else: pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, cz(f"- Tepelná ztráta objektu: {ztrata} kW"), ln=True)
            pdf.cell(0, 6, cz(f"- Návrhová venkovní teplota: {t_design} °C (Žádaná vnitřní: {t_vnitrni} °C)"), ln=True)
            pdf.cell(0, 6, cz(f"- Teplotní spád otopné soustavy: {t_spad} °C"), ln=True)
            pdf.cell(0, 6, cz(f"- Cílová teplota TUV: {t_tuv_cil} °C"), ln=True)
            pdf.cell(0, 6, cz(f"- Roční spotřeba: ÚT {spotreba_ut} MWh | TUV {spotreba_tuv} MWh"), ln=True)
            pdf.cell(0, 6, cz(f"- Technologie: Kaskáda {pocet_tc} ks TČ"), ln=True)
            pdf.cell(0, 6, cz(f"- Ekonomika: Cena CZT {cena_gj_czt} Kč/GJ | El. {cena_el} Kč/MWh"), ln=True)

            pdf.ln(4)
            if has_unicode_font: pdf.set_font("DejaVu", "B", 11)
            else: pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, cz("2. VÝSLEDKY A EKONOMIKA"), ln=True)
            
            if has_unicode_font: pdf.set_font("DejaVu", "", 10)
            else: pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, cz(f"- Bod bivalence (vypočtený): {t_biv_val:.1f} °C"), ln=True)
            pdf.cell(0, 6, cz(f"- Roční úspora nákladů: {uspora:,.0f} Kč | Návratnost: {navratnost:.1f} let"), ln=True)
            
            pdf.ln(2)
            if has_unicode_font: pdf.set_font("DejaVu", "B", 10)
            else: pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 8, cz("Tabulka bilance bivalence:"), ln=True)
            
            if has_unicode_font: pdf.set_font("DejaVu", "", 9)
            else: pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 5, cz(f"Energie (MWh): TČ {df_biv_table.iloc[0,1]} | Biv {df_biv_table.iloc[0,2]} | Podíl: {df_biv_table.iloc[0,3]} %"), ln=True)
            pdf.cell(0, 5, cz(f"Elektřina (MWh): TČ {df_biv_table.iloc[1,1]} | Biv {df_biv_table.iloc[1,2]} | Podíl: {df_biv_table.iloc[1,3]} %"), ln=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1:
                fig12.savefig(f1.name, dpi=100); pdf.image(f1.name, x=10, y=pdf.get_y()+5, w=190)
            
            pdf.set_xy(10, 185) 
            if has_unicode_font: pdf.set_font("DejaVu", "", 8)
            else: pdf.set_font("Helvetica", "I", 8)
            pdf.multi_cell(0, 5, cz(expl_12))

            # Strana 2
            pdf.add_page()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2:
                fig34.savefig(f2.name, dpi=100); pdf.image(f2.name, x=10, y=10, w=190)
            pdf.set_xy(10, 85); pdf.multi_cell(0, 5, cz(expl_34))
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f5:
                fig5.savefig(f5.name, dpi=100); pdf.image(f5.name, x=10, y=100, w=190)
            pdf.set_xy(10, 155); pdf.multi_cell(0, 5, cz(expl_5))

            # Strana 3
            pdf.add_page()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f3:
                fig7.savefig(f3.name, dpi=100); pdf.image(f3.name, x=10, y=10, w=90)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f4:
                fig6.savefig(f4.name, dpi=100); pdf.image(f4.name, x=105, y=10, w=90)
            pdf.set_xy(10, 105); pdf.multi_cell(0, 5, cz(expl_67))
                
            return bytes(pdf.output())

        st.sidebar.markdown("---")
        if st.sidebar.button("🚀 GENEROVAT PDF (v4.4)"):
            st.sidebar.download_button("📥 Stáhnout PDF Report", generate_pdf_v44(), f"Report_{nazev_projektu}.pdf")
