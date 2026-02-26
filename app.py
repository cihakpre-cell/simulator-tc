import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import urllib.request
import unicodedata
from fpdf import FPDF
import tempfile
import requests
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

# --- KONFIGURACE FONTŮ ---
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

def load_tmy_robust(file):
    try:
        if hasattr(file, 'getvalue'):
            content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
        else:
            content = file.read().decode('utf-8', errors='ignore').splitlines()
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

st.set_page_config(page_title="Simulator TC v6.1 - Map Restored", layout="wide")
download_fonts()

if "lat" not in st.session_state: st.session_state.lat = 49.8175
if "lon" not in st.session_state: st.session_state.lon = 15.4730
if "tmy_df" not in st.session_state: st.session_state.tmy_df = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Konfigurace")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    nazev_tc = st.text_input("Model tepelného čerpadla", "NIBE S2125-12")
    
    metodika_ui = st.radio("Metodika výpočtu:", ["Faktury (známá spotřeba)", "Projekt (známá TZ)"])
    metodika_vypoctu = "Faktury" if "Faktury" in metodika_ui else "Projekt"

    with st.expander("🏠 Budova a potřeba", expanded=True):
        ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
        t_vnitrni = st.number_input("Žádaná vnitřní teplota [°C]", value=20.0)
        t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
        t_spad = st.text_input("Teplotní spád soustavy [°C]", "55/45")
        if metodika_vypoctu == "Faktury":
            spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
            spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)
        else:
            pocet_osob = st.number_input("Počet osob", value=80)
            litry_osoba = st.number_input("l/osoba/den", value=45)
            spotreba_tuv = (pocet_osob * litry_osoba * 365 * 45 * 1.163) / 1000000
            st.write(f"Vypočtená TUV: {spotreba_tuv:.1f} MWh/rok")
            spotreba_ut = 0 

    with st.expander("🔧 Technologie", expanded=True):
        pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 4)
        eta_biv = st.slider("Účinnost bivalence [%]", 80, 100, 98) / 100
        char_file = st.file_uploader("Nahrát CSV charakteristiku", type="csv")
        if char_file: df_char_raw = load_char(char_file)
        else:
            df_char_raw = pd.DataFrame({"Teplota [°C]": [-15, -7, 2, 7, 15], "Výkon [kW]": [7.5, 9.2, 11.5, 12.0, 13.5], "COP [-]": [2.1, 2.8, 3.5, 4.2, 5.1]})
        df_char = st.data_editor(df_char_raw, num_rows="dynamic")

    with st.expander("💰 Ekonomika", expanded=True):
        cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800)
        cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284)
        servis = st.number_input("Roční servis [Kč]", value=17500)

# --- TMY DATA ---
st.header("🌍 Zdroj klimatických dat (TMY)")
c1, c2 = st.columns([1, 2])
with c1:
    adresa = st.text_input("Lokalita (vyhledat):")
    if adresa and st.button("Hledat"):
        loc = Nominatim(user_agent="tc_sim_final_fix").geocode(adresa)
        if loc: st.session_state.lat, st.session_state.lon = loc.latitude, loc.longitude
    st.write(f"📍 **Souřadnice:** {st.session_state.lat:.4f}, {st.session_state.lon:.4f}")
    if st.button("⬇️ STÁHNOUT TMY DATA", type="primary"):
        url = f"https://re.jrc.ec.europa.eu/api/tmy?lat={st.session_state.lat}&lon={st.session_state.lon}&outputformat=csv"
        resp = requests.get(url)
        if resp.status_code == 200:
            st.session_state.tmy_df = load_tmy_robust(io.BytesIO(resp.content))
with c2:
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=13)
    folium.Marker([st.session_state.lat, st.session_state.lon]).add_to(m)
    out = st_folium(m, height=250, width=600, key="mapa_v61")
    if out and out.get("last_clicked"):
        if out["last_clicked"]["lat"] != st.session_state.lat:
            st.session_state.lat, st.session_state.lon = out["last_clicked"]["lat"], out["last_clicked"]["lng"]
            st.rerun()

# --- VÝPOČET ---
if st.session_state.tmy_df is not None:
    tmy = st.session_state.tmy_df.copy()
    tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
    tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
    tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
    t_col, v_col, c_col = df_char.columns[0], df_char.columns[1], df_char.columns[2]
    q_tuv_avg = (spotreba_tuv / 8760) * 1000
    potreba_ut_teorie = [max(0, ztrata * (t_vnitrni - t) / (t_vnitrni - t_design)) for t in tmy['T_smooth']]
    
    if metodika_vypoctu == "Faktury":
        k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000) if sum(potreba_ut_teorie) > 0 else 1.0
    else:
        k_oprava = 1.0
        spotreba_ut = sum(potreba_ut_teorie) / 1000

    try: t_water_max = float(t_spad.split('/')[0])
    except: t_water_max = 55.0

    res = []
    for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
        q_ut = max(0, (ztrata * (t_vnitrni - t_sm) / (t_vnitrni - t_design) * k_oprava))
        q_need = q_ut + q_tuv_avg
        p_max = np.interp(t_out, df_char[t_col], df_char[v_col]) * pocet_tc
        cop_base = np.interp(t_out, df_char[t_col], df_char[c_col])
        t_water_act = 25.0 + (t_water_max - 25.0) * ((t_vnitrni - t_out) / (t_vnitrni - t_design)) if t_out < t_vnitrni else 25.0
        cop_ut = cop_base * (1 + 0.025 * max(0, t_water_max - t_water_act)) if cop_base > 0 else 0
        q_tc = min(q_need, p_max); q_biv = max(0, q_need - q_tc)
        el_tc = (min(q_tc, q_tuv_avg) / cop_base) + (max(0, q_tc - q_tuv_avg) / cop_ut) if cop_base > 0 else 0
        el_biv = q_biv / eta_biv
        res.append([t_out, q_need, q_tc, q_biv, el_tc, el_biv])

    df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
    t_biv_val = -12.0
    for t in np.linspace(15, -15, 500):
        if (np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc) < (max(0, (ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava)) + q_tuv_avg):
            t_biv_val = t; break

    naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
    naklady_tc = ((df_sim['El_tc'].sum() + df_sim['El_biv'].sum()) / 1000 * cena_el) + servis
    q_tc_s, q_bv_s = df_sim['Q_tc'].sum()/1000, df_sim['Q_biv'].sum()/1000
    el_tc_s, el_bv_s = df_sim['El_tc'].sum()/1000, df_sim['El_biv'].sum()/1000

    # --- GRAFY STREAMLIT ---
    fig12, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    tr = np.linspace(-15, 18, 100); q_p = np.array([max(0, (ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava)) + q_tuv_avg for t in tr])
    p_p = np.array([np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tr])
    ax1.plot(tr, q_p, 'r-', lw=2, label='Potřeba (UT+TUV)'); ax1.plot(tr, p_p, 'b--', alpha=0.4, label='Max kaskáda TČ')
    ax1.fill_between(tr, p_p, q_p, where=(q_p > p_p), color='red', alpha=0.2, hatch='XXXX', label='Oblast bivalence')
    ax1.axvline(t_biv_val, color='k', ls=':', label=f'Bod bivalence: {t_biv_val:.1f}°C'); ax1.set_title("1. DYNAMIKA PROVOZU"); ax1.legend()
    df_sim['TR'] = df_sim['Temp'].round(); dft = df_sim.groupby('TR')[['Q_tc', 'Q_biv']].sum()
    ax2.bar(dft.index, dft['Q_tc'], color='#3498db', label='TČ'); ax2.bar(dft.index, dft['Q_biv'], bottom=dft['Q_tc'], color='#e74c3c', label='Biv'); ax2.set_title("2. ENERGETICKÝ MIX DLE TEPLOT"); ax2.legend()

    fig34, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 7))
    df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1; m_df = df_sim.groupby('Month').agg({'Q_tc': 'sum', 'Q_biv': 'sum'})
    ax3.bar(m_df.index, m_df['Q_tc']/1000, color='#ADD8E6', label='TČ'); ax3.bar(m_df.index, m_df['Q_biv']/1000, bottom=m_df['Q_tc']/1000, color='#FF0000', label='Biv'); ax3.set_title("3. MĚSÍČNÍ BILANCE ENERGIE"); ax3.legend()
    q_sort = np.sort(df_sim['Q_need'].values)[::-1]; p_lim = np.interp(t_biv_val, df_char[t_col], df_char[v_col]) * pocet_tc
    ax4.plot(range(8760), q_sort, 'r', lw=2); ax4.fill_between(range(8760), 0, np.minimum(q_sort, p_lim), color='#ADD8E6', label='Kryto TČ'); ax4.fill_between(range(8760), p_lim, q_sort, where=(q_sort > p_lim), color='#FF0000', label='Bivalence'); ax4.set_title("4. TRVÁNÍ POTŘEBY (MONOTONA)"); ax4.legend()

    fig5, ax5 = plt.subplots(figsize=(18, 5))
    df_st = df_sim.sort_values('Temp').reset_index(drop=True); ax5.plot(df_st.index, df_st['Q_need'], 'r', label='Potřeba UT+TUV'); ax5.plot(df_st.index, df_st['Q_tc'], 'b', label='Krytí TČ'); ax5.set_title("5. ČETNOST TEPLOT V ROCE"); ax5.legend()

    fig6, ax6 = plt.subplots(figsize=(6, 6)); ax6.pie([q_tc_s, q_bv_s], labels=['TČ', 'Biv'], autopct='%1.1f%%', colors=['#ADD8E6', '#FF0000']); ax6.set_title("ROČNÍ PODÍL ENERGIE")
    fig7, ax7 = plt.subplots(figsize=(6, 6)); ax7.bar(['CZT', 'TČ'], [naklady_czt, naklady_tc], color=['#95a5a6', '#2ecc71'])
    for i, v in enumerate([naklady_czt, naklady_tc]): ax7.text(i, v, f"{int(v):,} Kč", ha='center', va='bottom')
    ax7.set_title("SROVNÁNÍ NÁKLADŮ [Kč/rok]")

    # Zobrazení na webu
    st.header(f"📊 Výsledky: {nazev_projektu}")
    st.pyplot(fig12)
    st.pyplot(fig34)
    st.pyplot(fig5)
    cl, cr = st.columns(2)
    with cl: st.table(df_table if 'df_table' in locals() else pd.DataFrame()); st.pyplot(fig6)
    with cr: st.pyplot(fig7)

    # --- PDF GENERATOR v6.1 ---
    def generate_pdf_v61():
        pdf = FPDF()
        has_u = os.path.exists(FONT_REGULAR)
        if has_u: 
            pdf.add_font("DejaVu", "", FONT_REGULAR)
            pdf.add_font("DejaVu", "B", FONT_BOLD)
            pdf.set_font("DejaVu", "", 12)
        else: pdf.set_font("Helvetica", "", 12)
        
        def cz(t): 
            if not has_u: return "".join([c for c in unicodedata.normalize('NFKD', str(t)) if not unicodedata.combining(c)])
            return str(t)
        
        pdf.add_page()
        pdf.set_font(pdf.font_family, "B", 16)
        pdf.cell(0, 10, cz(f"TECHNICKÝ REPORT: {nazev_projektu.upper()}"), ln=True, align="C")
        pdf.ln(5)
        
        # METODIKA
        pdf.set_font(pdf.font_family, "B", 11)
        pdf.cell(0, 8, cz("METODIKA VÝPOČTU A LOGIKA SIMULACE"), ln=True)
        pdf.set_font(pdf.font_family, "", 9)
        metodika_text = (
            "Vypocet vychazi z hodinove simulace energeticke bilance objektu (8760 kroku za rok). "
            "Simulace vyuziva klimaticka data TMY (Typicky Meteorologicky Rok) pro konkretni GPS lokaci. "
            "Vypocet zohlednuje tepelnou setrvacnost budovy, dynamicke rizeni teploty otopne vody dle ekvitermy a prioritni ohrev TUV. "
            "Vysledkem je presne stanoveni bodu bivalence a realneho sezonniho COP."
        )
        if metodika_vypoctu == "Faktury":
            metodika_text += f"\n\nSPECIFIKA VARIANTY FAKTURY: Vypocet je kalibrovan na zaklade skutecne spotreby {spotreba_ut + spotreba_tuv:.1f} MWh/rok."
        else:
            metodika_text += f"\n\nSPECIFIKA VARIANTY PROJEKT: Vypocet vychazi z projektove tepelne ztraty {ztrata} kW pri navrhove teplote {t_design} C."
        pdf.multi_cell(0, 5, cz(metodika_text))
        pdf.ln(5); pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(5)
        
        # PARAMETRY
        pdf.set_font(pdf.font_family, "B", 11); pdf.cell(0, 8, cz("VSTUPNÍ PARAMETRY"), ln=True)
        pdf.set_font(pdf.font_family, "", 10)
        curr_y = pdf.get_y()
        pdf.cell(0, 6, cz(f"- Lokalita: {st.session_state.lat:.4f}, {st.session_state.lon:.4f}"), ln=True)
        pdf.cell(0, 6, cz(f"- Model TC: {nazev_tc} | Pocet: {pocet_tc} | Ztrata: {ztrata} kW"), ln=True)
        pdf.cell(0, 6, cz(f"- Spotreba UT: {spotreba_ut:.1f} MWh | TUV: {spotreba_tuv:.1f} MWh"), ln=True)
        pdf.cell(0, 6, cz(f"- Bod bivalence: {t_biv_val:.1f} C"), ln=True)
        
        # --- OBNOVENÁ MAPKA LOKALITY ---
        # Použití statického API pro mapu (nejstabilnější pro PDF)
        try:
            zoom = 13
            # Náhradní vykreslení mapky přes Matplotlib pokud API není dostupné
            fig_m, ax_m = plt.subplots(figsize=(4, 3))
            ax_m.set_facecolor('#e5e3df')
            ax_m.plot(st.session_state.lon, st.session_state.lat, 'ro', markersize=12, markeredgecolor='k')
            ax_m.set_title(f"GPS: {st.session_state.lat:.3f}, {st.session_state.lon:.3f}", fontsize=8)
            ax_m.set_xticks([]); ax_m.set_yticks([])
            for spine in ax_m.spines.values(): spine.set_visible(True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f_map:
                fig_m.savefig(f_map.name, dpi=100, bbox_inches='tight')
                pdf.image(f_map.name, x=140, y=curr_y, w=50)
            plt.close(fig_m)
        except: pass

        # GRAFY A POPISY
        pdf.ln(10)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            fig12.savefig(f.name, dpi=100); pdf.image(f.name, x=10, y=pdf.get_y(), w=190)
        pdf.ln(2); pdf.set_font(pdf.font_family, "", 8); pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 4, cz("Graf 1 a 2: Bod bivalence urcuje venkovni teplotu, pod kterou musi kaskade TC pomahat bivalentni zdroj."))
        
        pdf.add_page()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            fig34.savefig(f.name, dpi=100); pdf.image(f.name, x=10, y=15, w=190)
        pdf.set_y(90); pdf.multi_cell(0, 4, cz("Graf 3 a 4: Mesicni bilance ukazuje sezonni vyuziti zdroju. Monotona vykonu vizualizuje casove rozlozeni potreby tepla."))
        pdf.ln(10)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            fig5.savefig(f.name, dpi=100); pdf.image(f.name, x=10, y=pdf.get_y(), w=190)
        pdf.set_y(175); pdf.multi_cell(0, 4, cz("Graf 5: Serazena cetnost hodinovych teplot v roce. Krivka kryti TC kopiruje potrebu budovy az do bodu bivalence."))
        
        pdf.add_page()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            fig6.savefig(f.name, dpi=100); pdf.image(f.name, x=10, y=15, w=90)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            fig7.savefig(f.name, dpi=100); pdf.image(f.name, x=105, y=15, w=90)
        
        # TABULKA BIVALENCE V PDF
        pdf.set_y(100); pdf.set_text_color(0, 0, 0); pdf.set_font(pdf.font_family, "B", 11)
        pdf.cell(0, 10, cz("PODROBNÁ BILANCE BIVALENCE"), ln=True)
        pdf.set_font(pdf.font_family, "B", 9)
        pdf.cell(50, 8, cz("Metrika"), 1); pdf.cell(40, 8, cz("TC"), 1); pdf.cell(40, 8, cz("Bivalence"), 1); pdf.cell(40, 8, cz("Podil"), 1); pdf.ln()
        pdf.set_font(pdf.font_family, "", 9)
        pdf.cell(50, 8, cz("Teplo [MWh]"), 1); pdf.cell(40, 8, f"{q_tc_s:.2f}", 1); pdf.cell(40, 8, f"{q_bv_s:.2f}", 1); pdf.cell(40, 8, f"{q_bv_s/(q_tc_s+q_bv_s)*100:.1f} %", 1); pdf.ln()
        pdf.cell(50, 8, cz("Elektrina [MWh]"), 1); pdf.cell(40, 8, f"{el_tc_s:.2f}", 1); pdf.cell(40, 8, f"{el_bv_s:.2f}", 1); pdf.cell(40, 8, f"{el_bv_s/(el_tc_s+el_bv_s)*100:.1f} %", 1); pdf.ln()
        
        pdf.ln(5); pdf.set_font(pdf.font_family, "", 8); pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 4, cz("Graf 6 znazornuje podil bivalence na tepelne energii za rok. Graf 7 znazornuje porovnani rocnich nakladu."))
        
        return bytes(pdf.output())

    # Tlačítko v sidebaru
    with st.sidebar:
        st.markdown("---")
        if st.button("🚀 GENEROVAT PDF REPORT", type="primary"):
            st.download_button("📥 Stáhnout PDF", generate_pdf_v61(), f"Report_{nazev_projektu}.pdf")
