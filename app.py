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

st.set_page_config(page_title="Simulator TC v5.2 - Expert Mode", layout="wide")
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
    
    # --- ROZCESTNÍK METODIKY ---
    metodika_vypoctu = st.radio("Metodika výpočtu potřeby:", 
                               ["📊 Podle roční spotřeby (Faktury)", "🏗️ Podle tepelné ztráty (Projekt)"])

    with st.expander("🏠 Budova a potřeba", expanded=True):
        ztrata = st.number_input("Tepelná ztráta [kW]", value=54.0)
        t_vnitrni = st.number_input("Žádaná vnitřní teplota [°C]", value=20.0)
        t_design = st.number_input("Návrhová teplota [°C]", value=-12.0)
        t_spad = st.text_input("Teplotní spád soustavy [°C]", "55/45")
        
        if "Faktury" in metodika_vypoctu:
            st.info("💡 Ztráta zde slouží pouze pro určení tvaru ekvitermní křivky.")
            spotreba_ut = st.number_input("Spotřeba ÚT [MWh/rok]", value=124.0)
            spotreba_tuv = st.number_input("Spotřeba TUV [MWh/rok]", value=76.0)
        else:
            st.info("💡 Spotřeba bude dopočtena z TZ a parametrů TUV.")
            pocet_osob = st.number_input("Počet osob v objektu", value=80)
            litry_osoba = st.slider("Spotřeba TUV [l/osoba/den]", 20, 100, 45)
            t_tuv_vstup = 10.0
            t_tuv_vystup = 55.0
            # Výpočet MWh TUV: osoby * litry * 365 dní * deltaT * konstanta / 1000
            spotreba_tuv = (pocet_osob * litry_osoba * 365 * (t_tuv_vystup - t_tuv_vstup) * 1.163) / 1000000
            st.write(f"Vypočtená spotřeba TUV: **{spotreba_tuv:.2f} MWh/rok**")
            # ÚT se dopočte v hlavním cyklu bez k_oprava

    with st.expander("🔧 Technologie & Charakteristika", expanded=True):
        pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 4)
        eta_biv = st.slider("Účinnost bivalence [%]", 80, 100, 98) / 100
        char_file = st.file_uploader("Nahrát CSV charakteristiku TČ", type="csv")
        if char_file:
            df_char_raw = load_char(char_file)
        else:
            df_char_raw = pd.DataFrame({
                "Teplota [°C]": [-15, -7, 2, 7, 15],
                "Výkon [kW]": [7.5, 9.2, 11.5, 12.0, 13.5],
                "COP [-]": [2.1, 2.8, 3.5, 4.2, 5.1]
            })
        df_char = st.data_editor(df_char_raw, num_rows="dynamic")

    with st.expander("💰 Ekonomika", expanded=True):
        cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800)
        cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284)
        servis = st.number_input("Roční servis [Kč]", value=17500)

# --- VÝBĚR TMY DAT ---
st.header("🌍 Zdroj klimatických dat (TMY)")
c1, c2 = st.columns([1, 2])
with c1:
    adresa = st.text_input("Vyhledat město/adresu:")
    if st.button("Lokalizovat"):
        geolocator = Nominatim(user_agent="tc_simulator_app")
        loc = geolocator.geocode(adresa)
        if loc:
            st.session_state.lat, st.session_state.lon = loc.latitude, loc.longitude
            st.success(f"Nalezeno: {loc.address}")
    
    st.write(f"**Souřadnice:** {st.session_state.lat:.4f}, {st.session_state.lon:.4f}")
    if st.button("⬇️ STÁHNOUT DATA PRO TUTO LOKACI", type="primary"):
        url = f"https://re.jrc.ec.europa.eu/api/tmy?lat={st.session_state.lat}&lon={st.session_state.lon}&outputformat=csv"
        resp = requests.get(url)
        if resp.status_code == 200:
            st.session_state.tmy_df = load_tmy_robust(io.BytesIO(resp.content))
            st.session_state.tmy_source_label = f"PVGIS (Lat: {st.session_state.lat:.4f}, Lon: {st.session_state.lon:.4f})"
            st.success("Data stažena!")

with c2:
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=15)
    folium.Marker([st.session_state.lat, st.session_state.lon]).add_to(m)
    st_folium(m, height=300, width=700)

# --- HLAVNÍ VÝPOČET ---
if st.session_state.tmy_df is not None and df_char is not None:
    tmy = st.session_state.tmy_df.copy()
    tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
    tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
    tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
    
    t_col, v_col, c_col = df_char.columns[0], df_char.columns[1], df_char.columns[2]
    q_tuv_avg = (spotreba_tuv / 8760) * 1000
    
    # Metodika výpočtu ÚT
    potreba_ut_teorie = [max(0, ztrata * (t_vnitrni - t) / (t_vnitrni - t_design)) for t in tmy['T_smooth']]
    
    if "Faktury" in metodika_vypoctu:
        k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000) if sum(potreba_ut_teorie) > 0 else 1.0
    else:
        k_oprava = 1.0 # V projektové metodice jedeme striktně podle TZ
        spotreba_ut = sum(potreba_ut_teorie) / 1000

    try: t_water_max = float(t_spad.split('/')[0])
    except: t_water_max = 55.0

    res = []
    for t_out, t_sm in zip(tmy['T2m'], tmy['T_smooth']):
        q_ut = max(0, (ztrata * (t_vnitrni - t_sm) / (t_vnitrni - t_design) * k_oprava))
        q_need = q_ut + q_tuv_avg
        p_max = np.interp(t_out, df_char[t_col], df_char[v_col]) * pocet_tc
        cop_base = np.interp(t_out, df_char[t_col], df_char[c_col])
        t_water_actual = 25.0 + (t_water_max - 25.0) * ((t_vnitrni - t_out) / (t_vnitrni - t_design)) if t_out < t_vnitrni else 25.0
        cop_ut = cop_base * (1 + 0.025 * max(0, t_water_max - t_water_actual))
        q_tc = min(q_need, p_max); q_biv = max(0, q_need - q_tc)
        el_tc = (min(q_tc, q_tuv_avg) / cop_base) + (max(0, q_tc - q_tuv_avg) / cop_ut) if cop_base > 0 else 0
        el_biv = q_biv / eta_biv
        res.append([t_out, q_need, q_tc, q_biv, el_tc, el_biv])

    df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
    
    # Výpočet bivalence pro vizualizaci
    t_biv_val = -12.0
    for t in np.linspace(15, -15, 500):
        if (np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc) < (max(0, (ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava)) + q_tuv_avg):
            t_biv_val = t; break

    # --- ZOBRAZENÍ VÝSLEDKŮ ---
    st.markdown("---")
    st.header(f"📊 Výsledky projektu: {nazev_projektu}")
    
    fig12, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    tr = np.linspace(-15, 18, 100)
    q_p = np.array([max(0, (ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava)) + q_tuv_avg for t in tr])
    p_p = np.array([np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tr])
    ax1.plot(tr, q_p, 'r-', lw=2, label='Potřeba (ÚT+TUV)')
    ax1.plot(tr, p_p, 'b--', alpha=0.4, label='Max kaskáda TČ')
    ax1.axvline(t_biv_val, color='black', linestyle=':', label=f'Bod bivalence: {t_biv_val:.1f}°C')
    ax1.set_title("1. DYNAMIKA PROVOZU"); ax1.legend()
    df_sim['Temp_R'] = df_sim['Temp'].round(); df_t = df_sim.groupby('Temp_R')[['Q_tc', 'Q_biv']].sum()
    ax2.bar(df_t.index, df_t['Q_tc'], color='#3498db', label='TČ'); ax2.bar(df_t.index, df_t['Q_biv'], bottom=df_t['Q_tc'], color='#e74c3c', label='Biv')
    ax2.set_title("2. ENERGETICKÝ MIX DLE TEPLOT"); ax2.legend()
    st.pyplot(fig12)

    # (Následují další grafy 3-7 stejné jako v v4.9...)
    # [Zkráceno pro přehlednost, v ostrém kódu ponechat kompletní vykreslování jako v 4.9]

    # --- PDF GENERÁTOR v5.2 ---
    def generate_pdf_v52():
        pdf = FPDF()
        has_u = os.path.exists(FONT_REGULAR)
        if has_u: pdf.add_font("DejaVu", "", FONT_REGULAR); pdf.add_font("DejaVu", "B", FONT_BOLD); pdf.set_font("DejaVu", "B", 16)
        def cz(txt): return str(txt) if has_u else "".join([c for c in unicodedata.normalize('NFKD', str(txt)) if not unicodedata.combining(c)])

        pdf.add_page()
        pdf.cell(0, 10, cz(f"TECHNICKÝ REPORT: {nazev_projektu.upper()}"), ln=True, align="C")
        
        pdf.ln(10)
        pdf.set_font(pdf.font_family, "B", 11); pdf.cell(0, 8, cz("METODIKA VÝPOČTU A LOGIKA SIMULACE"), ln=True)
        pdf.set_font(pdf.font_family, "", 9)
        metodika_v_pdf = f"Výpočet proveden metodou: {metodika_vypoctu}. " + (
            "Bilance vychází z hodinové simulace TMY dat pro danou GPS lokalitu. "
            "Uvažována je tepelná setrvačnost objektu, priorita ohřevu TUV a ekvitermní řízení výkonu."
        )
        pdf.multi_cell(0, 5, cz(metodika_v_pdf))
        pdf.ln(5); pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(5)

        pdf.set_font(pdf.font_family, "B", 11); pdf.cell(0, 8, cz("VSTUPNÍ PARAMETRY"), ln=True)
        pdf.set_font(pdf.font_family, "", 10)
        pdf.cell(0, 6, cz(f"- Lokalita: {st.session_state.lat:.4f}, {st.session_state.lon:.4f}"), ln=True)
        pdf.cell(0, 6, cz(f"- Tepelná ztráta objektu: {ztrata} kW (při {t_design} °C)"), ln=True)
        pdf.cell(0, 6, cz(f"- Roční potřeba ÚT: {spotreba_ut:.2f} MWh | TUV: {spotreba_tuv:.2f} MWh"), ln=True)
        pdf.cell(0, 6, cz(f"- Model TČ: {nazev_tc} | Počet: {pocet_tc} ks | Bod bivalence: {t_biv_val:.1f} °C"), ln=True)

        return bytes(pdf.output())

    if st.sidebar.button("🚀 GENEROVAT PDF REPORT"):
        pdf_data = generate_pdf_v52()
        st.sidebar.download_button("📥 Stáhnout PDF", pdf_data, f"Report_{nazev_projektu}.pdf")
