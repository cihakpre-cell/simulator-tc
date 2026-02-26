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

st.set_page_config(page_title="Simulator TC v5.1", layout="wide")
download_fonts()

# Inicializace session state
if "lat" not in st.session_state: st.session_state.lat = 49.8
if "lon" not in st.session_state: st.session_state.lon = 15.5
if "tmy_df" not in st.session_state: st.session_state.tmy_df = None
if "tmy_source_label" not in st.session_state: st.session_state.tmy_source_label = "Nenahráno"

# --- SIDEBAR KONFIGURACE ---
with st.sidebar:
    st.header("⚙️ Konfigurace")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sládkovičova")
    nazev_tc = st.text_input("Model tepelného čerpadla", "NIBE S2125-12")
    
    ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
    t_vnitrni = st.number_input("Žádaná vnitřní teplota [°C]", value=20.0)
    t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
    t_spad = st.text_input("Teplotní spád soustavy [°C]", "55/45")
    spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
    spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)

    st.subheader("🔧 Technologie")
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 4)
    eta_biv = st.slider("Účinnost bivalence [%]", 80, 100, 98) / 100
    df_char = st.data_editor(pd.DataFrame({
        "Teplota [°C]": [-15, -7, 2, 7, 15],
        "Výkon [kW]": [7.5, 9.2, 11.5, 12.0, 13.5],
        "COP [-]": [2.1, 2.8, 3.5, 4.2, 5.1]
    }), num_rows="dynamic")

    st.subheader("💰 Ekonomika")
    cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800)
    cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284)
    servis = st.number_input("Roční servis [Kč]", value=17500)

# --- MAPA A DATA ---
st.header("🌍 Lokalizace a klimatická data")
c1, c2 = st.columns([1, 2])
with c1:
    adresa = st.text_input("Hledat adresu:")
    if st.button("Lokalizovat"):
        loc = Nominatim(user_agent="tc_sim").geocode(adresa)
        if loc:
            st.session_state.lat, st.session_state.lon = loc.latitude, loc.longitude
            st.rerun()
    st.write(f"Souřadnice: {st.session_state.lat:.4f}, {st.session_state.lon:.4f}")
    if st.button("⬇️ STÁHNOUT TMY DATA", type="primary"):
        url = f"https://re.jrc.ec.europa.eu/api/tmy?lat={st.session_state.lat}&lon={st.session_state.lon}&outputformat=csv"
        resp = requests.get(url)
        if resp.status_code == 200:
            st.session_state.tmy_df = load_tmy_robust(io.BytesIO(resp.content))
            st.session_state.tmy_source_label = f"PVGIS (Lat:{st.session_state.lat:.2f})"
            st.success("Data připravena.")
        else: st.error("Chyba při stahování.")

with c2:
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=15)
    folium.Marker([st.session_state.lat, st.session_state.lon]).add_to(m)
    map_data = st_folium(m, height=300, width=700)
    if map_data and map_data.get("last_clicked"):
        st.session_state.lat, st.session_state.lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        st.rerun()

# --- VÝPOČET A ZOBRAZENÍ ---
if st.session_state.tmy_df is not None:
    tmy = st.session_state.tmy_df.copy()
    tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
    tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
    tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
    
    t_col, v_col, c_col = df_char.columns[0], df_char.columns[1], df_char.columns[2]
    q_tuv_avg = (spotreba_tuv / 8760) * 1000
    potreba_ut_teorie = [max(0, ztrata * (t_vnitrni - t) / (t_vnitrni - t_design)) for t in tmy['T_smooth']]
    k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000) if sum(potreba_ut_teorie) > 0 else 1.0
    t_water_max = float(t_spad.split('/')[0]) if '/' in t_spad else 55.0

    res = []
    for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
        q_ut = max(0, (ztrata * (t_vnitrni - t_sm) / (t_vnitrni - t_design) * k_oprava))
        q_need = q_ut + q_tuv_avg
        p_max = np.interp(t_out, df_char[t_col], df_char[v_col]) * pocet_tc
        cop_base = np.interp(t_out, df_char[t_col], df_char[c_col])
        t_water_act = 25.0 + (t_water_max - 25.0) * ((t_vnitrni - t_out) / (t_vnitrni - t_design)) if t_out < t_vnitrni else 25.0
        cop_ut = cop_base * (1 + 0.025 * max(0, t_water_max - t_water_act))
        q_tc = min(q_need, p_max); q_biv = max(0, q_need - q_tc)
        el_tc = (min(q_tc, q_tuv_avg) / cop_base) + (max(0, q_tc - q_tuv_avg) / cop_ut) if cop_base > 0 else 0
        el_biv = q_biv / eta_biv
        res.append([t_out, q_need, q_tc, q_biv, el_tc, el_biv])

    df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
    
    # --- VYKRESLENÍ GRAFŮ V APLIKACI ---
    st.markdown("---")
    st.header(f"📊 Výsledky: {nazev_projektu}")
    
    fig12, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    tr = np.linspace(-15, 18, 100)
    q_p = np.array([max(0, (ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava)) + q_tuv_avg for t in tr])
    p_p = np.array([np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tr])
    ax1.plot(tr, q_p, 'r-', lw=2, label='Potřeba'); ax1.plot(tr, p_p, 'b--', alpha=0.4, label='TČ')
    ax1.set_title("Dynamika bodu bivalence"); ax1.legend()
    df_sim['TR'] = df_sim['Temp'].round(); dft = df_sim.groupby('TR')[['Q_tc', 'Q_biv']].sum()
    ax2.bar(dft.index, dft['Q_tc'], label='TČ'); ax2.bar(dft.index, dft['Q_biv'], bottom=dft['Q_tc'], label='Biv')
    ax2.set_title("Energie dle teplot"); ax2.legend()
    st.pyplot(fig12)

    # --- PDF EXPORT ---
    def generate_pdf_v51():
        pdf = FPDF()
        if os.path.exists(FONT_REGULAR):
            pdf.add_font("DejaVu", "", FONT_REGULAR); pdf.add_font("DejaVu", "B", FONT_BOLD); pdf.set_font("DejaVu", "B", 16)
        def cz(t): return "".join([c for c in unicodedata.normalize('NFKD', str(t)) if not unicodedata.combining(c)]) if not os.path.exists(FONT_REGULAR) else str(t)
        
        pdf.add_page()
        pdf.cell(0, 10, cz(f"TECHNICKÝ REPORT: {nazev_projektu.upper()}"), ln=True, align="C")
        
        pdf.ln(12) # Větší mezera
        pdf.set_font(pdf.font_family, "B", 12); pdf.cell(0, 8, cz("METODIKA VÝPOČTU A LOGIKA SIMULACE"), ln=True)
        pdf.set_font(pdf.font_family, "", 10)
        metodika = (
            "Analyza vychazi z hodinove simulace energeticke bilance (8760 kroku) s vyuzitim meteorologickych dat TMY. "
            "Simulace zohlednuje tepelnou setrvacnost budovy (6h prumer) a prioritu TUV. COP je dynamicky dopocitavan "
            "dle ekvitermni krivky otopne soustavy."
        )
        pdf.multi_cell(0, 5, cz(metodika)); pdf.ln(5); pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(5)

        pdf.set_font(pdf.font_family, "B", 11); pdf.cell(0, 8, cz("VSTUPNÍ PARAMETRY"), ln=True); pdf.set_font(pdf.font_family, "", 10)
        curr_y = pdf.get_y()
        pdf.cell(0, 6, cz(f"- Lokalita: {st.session_state.lat:.4f}, {st.session_state.lon:.4f}"), ln=True)
        pdf.cell(0, 6, cz(f"- Model: {nazev_tc} ({pocet_tc} ks)"), ln=True)
        pdf.cell(0, 6, cz(f"- Ztrata: {ztrata} kW | Spotreba UT+TUV: {spotreba_ut+spotreba_tuv} MWh"), ln=True)

        try:
            url = f"https://static-maps.yandex.ru/1.x/?ll={st.session_state.lon},{st.session_state.lat}&z=16&l=map&size=450,450&pt={st.session_state.lon},{st.session_state.lat},pm2rdm"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f: f.write(r.content); pdf.image(f.name, x=145, y=curr_y, w=50)
        except: pass

        pdf.ln(10)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f: fig12.savefig(f.name, dpi=100); pdf.image(f.name, x=10, y=pdf.get_y(), w=190)
        return bytes(pdf.output())

    if st.sidebar.button("🚀 GENEROVAT PDF REPORT"):
        pdf_bytes = generate_pdf_v51()
        st.sidebar.download_button("📥 Stáhnout PDF", pdf_bytes, f"Report_{nazev_projektu}.pdf")
