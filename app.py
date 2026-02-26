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

st.set_page_config(page_title="Simulator TC v5.7 - Stable", layout="wide")
download_fonts()

if "lat" not in st.session_state: st.session_state.lat = 49.8175
if "lon" not in st.session_state: st.session_state.lon = 15.4730
if "tmy_df" not in st.session_state: st.session_state.tmy_df = None
if "tmy_source_label" not in st.session_state: st.session_state.tmy_source_label = "Nenahráno"

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Konfigurace")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    nazev_tc = st.text_input("Model tepelného čerpadla", "NIBE S2125-12")
    
    # OČIŠTĚNÝ ROZCESTNÍK (Emoji jsou jen popisky v Streamlitu)
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
        char_file = st.file_uploader("Nahrát CSV", type="csv")
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
        loc = Nominatim(user_agent="tc_sim_final").geocode(adresa)
        if loc: st.session_state.lat, st.session_state.lon = loc.latitude, loc.longitude
    
    st.write(f"📍 **Souřadnice:** {st.session_state.lat:.4f}, {st.session_state.lon:.4f}")
    if st.button("⬇️ STÁHNOUT TMY DATA", type="primary"):
        url = f"https://re.jrc.ec.europa.eu/api/tmy?lat={st.session_state.lat}&lon={st.session_state.lon}&outputformat=csv"
        resp = requests.get(url)
        if resp.status_code == 200:
            st.session_state.tmy_df = load_tmy_robust(io.BytesIO(resp.content))
            st.session_state.tmy_source_label = f"PVGIS ({st.session_state.lat:.2f})"
with c2:
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=8)
    folium.Marker([st.session_state.lat, st.session_state.lon]).add_to(m)
    out = st_folium(m, height=300, width=600, key="mapa_final")
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

    # --- GRAFY ---
    st.header(f"📊 Výsledky: {nazev_projektu}")
    fig12, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    tr = np.linspace(-15, 18, 100); q_p = np.array([max(0, (ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava)) + q_tuv_avg for t in tr])
    p_p = np.array([np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tr])
    ax1.plot(tr, q_p, 'r-', lw=2, label='Potřeba'); ax1.plot(tr, p_p, 'b--', alpha=0.4, label='TČ')
    ax1.fill_between(tr, p_p, q_p, where=(q_p > p_p), color='red', alpha=0.2, hatch='XXXX', label='Bivalence')
    ax1.axvline(t_biv_val, color='k', ls=':', label=f'Bod bivalence: {t_biv_val:.1f}°C'); ax1.set_title("1. Dynamika"); ax1.legend()
    df_sim['TR'] = df_sim['Temp'].round(); dft = df_sim.groupby('TR')[['Q_tc', 'Q_biv']].sum()
    ax2.bar(dft.index, dft['Q_tc'], color='#3498db', label='TČ'); ax2.bar(dft.index, dft['Q_biv'], bottom=dft['Q_tc'], color='#e74c3c', label='Biv'); ax2.legend()
    st.pyplot(fig12)

    fig34, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 7))
    df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1; m_df = df_sim.groupby('Month').agg({'Q_tc': 'sum', 'Q_biv': 'sum'})
    ax3.bar(m_df.index, m_df['Q_tc']/1000, color='#ADD8E6'); ax3.bar(m_df.index, m_df['Q_biv']/1000, bottom=m_df['Q_tc']/1000, color='#FF0000'); ax3.set_title("3. Měsíční bilance")
    q_sort = np.sort(df_sim['Q_need'].values)[::-1]; p_lim = np.interp(t_biv_val, df_char[t_col], df_char[v_col]) * pocet_tc
    ax4.plot(range(8760), q_sort, 'r'); ax4.fill_between(range(8760), 0, np.minimum(q_sort, p_lim), color='#ADD8E6'); ax4.fill_between(range(8760), p_lim, q_sort, where=(q_sort > p_lim), color='#FF0000'); ax4.set_title("4. Monotóna")
    st.pyplot(fig34)

    fig5, ax5 = plt.subplots(figsize=(18, 5))
    df_st = df_sim.sort_values('Temp').reset_index(drop=True); ax5.plot(df_st.index, df_st['Q_need'], 'r'); ax5.plot(df_st.index, df_st['Q_tc'], 'b'); ax5.set_title("5. Četnost teplot")
    st.pyplot(fig5)

    c_l, c_r = st.columns(2)
    with c_l:
        df_table = pd.DataFrame({"Metrika": ["Teplo [MWh]"], "TČ": [round(q_tc_s, 2)], "Biv": [round(q_bv_s, 2)], "Podíl [%]": [round(q_bv_s/(q_tc_s+q_bv_s)*100, 1)]})
        st.table(df_table); fig6, ax6 = plt.subplots(figsize=(6, 6)); ax6.pie([q_tc_s, q_bv_s], labels=['TČ', 'Biv'], autopct='%1.1f%%', colors=['#ADD8E6', '#FF0000']); st.pyplot(fig6)
    with c_r:
        fig7, ax7 = plt.subplots(figsize=(6, 6)); ax7.bar(['CZT', 'TČ'], [naklady_czt, naklady_tc], color=['grey', 'green'])
        ax7.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax7.set_title("7. Srovnání nákladů [Kč/rok]"); st.pyplot(fig7)

    # --- PDF ---
    def generate_pdf_final():
        pdf = FPDF()
        has_u = os.path.exists(FONT_REGULAR)
        if has_u: pdf.add_font("DejaVu", "", FONT_REGULAR); pdf.add_font("DejaVu", "B", FONT_BOLD); pdf.set_font("DejaVu", "", 12)
        else: pdf.set_font("Helvetica", "", 12)
        
        def cz(t): 
            if not has_u: return "".join([c for c in unicodedata.normalize('NFKD', str(t)) if not unicodedata.combining(c)])
            return str(t)
        
        pdf.add_page()
        pdf.set_font(pdf.font_family, "B", 16)
        pdf.cell(0, 10, cz(f"REPORT: {nazev_projektu.upper()}"), ln=True, align="C")
        pdf.ln(10); pdf.set_font(pdf.font_family, "B", 11); pdf.cell(0, 8, cz("PARAMETRY"), ln=True); pdf.set_font(pdf.font_family, "", 10)
        pdf.cell(0, 6, cz(f"- Lokalita: {st.session_state.lat:.2f}, {st.session_state.lon:.2f} | Metodika: {metodika_vypoctu}"), ln=True)
        pdf.cell(0, 6, cz(f"- TZ: {ztrata} kW | UT: {spotreba_ut:.1f} MWh | TUV: {spotreba_tuv:.1f} MWh"), ln=True)
        pdf.cell(0, 6, cz(f"- TC: {nazev_tc} | Biv: {t_biv_val:.1f} C"), ln=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f: fig12.savefig(f.name, dpi=100); pdf.image(f.name, x=10, y=pdf.get_y()+5, w=190)
        pdf.add_page()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f: fig34.savefig(f.name, dpi=100); pdf.image(f.name, x=10, y=10, w=190)
        return bytes(pdf.output())

    if st.sidebar.button("🚀 GENEROVAT PDF"):
        st.sidebar.download_button("📥 Stáhnout", generate_pdf_final(), f"Report_{nazev_projektu}.pdf")
