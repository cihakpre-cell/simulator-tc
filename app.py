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

st.set_page_config(page_title="Simulator TC v4.9 - Fixed Map Report", layout="wide")
download_fonts()

# Inicializace session state
if "lat" not in st.session_state: st.session_state.lat = 49.8
if "lon" not in st.session_state: st.session_state.lon = 15.5
if "tmy_df" not in st.session_state: st.session_state.tmy_df = None
if "tmy_source_label" not in st.session_state: st.session_state.tmy_source_label = "Nenahráno"

# --- INFORMAČNÍ SEKCE ---
with st.sidebar:
    st.header("⚙️ Konfigurace")
    with st.expander("ℹ️ Metodika a logika výpočtu"):
        st.markdown("""
        **1. Klimatická data (TMY):** Výpočet probíhá hodinu po hodině.
        **2. Tepelná setrvačnost:** Simulace využívá plovoucí průměr venkovních teplot za 6 hodin.
        **3. Priorita TUV:** Potřeba TUV je uvažována jako prioritní odběr.
        **4. Ekvitermní COP:** Účinnost (COP) je dynamicky přepočítávána.
        **5. Bod bivalence:** Výsledný bod, kde potřeba budovy převýší výkon kaskády.
        """)
    
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
        investice = st.number_input("Investice celkem [Kč]", value=4080000)
        cena_el = st.number_input("Cena elektřiny [Kč/MWh]", value=4800)
        cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284)
        servis = st.number_input("Roční servis [Kč]", value=17500)

# --- VÝBĚR TMY DAT ---
st.header("🌍 Zdroj klimatických dat (TMY)")
tmy_source = st.radio("Zvolte způsob nahrání dat:", ["🌍 Stáhnout automaticky z mapy (PVGIS)", "📂 Nahrát vlastní CSV soubor"], horizontal=True)

if tmy_source == "🌍 Stáhnout automaticky z mapy (PVGIS)":
    c1, c2 = st.columns([1, 2])
    with c1:
        adresa = st.text_input("Vyhledat město/adresu:")
        if st.button("Lokalizovat"):
            geolocator = Nominatim(user_agent="tc_simulator_app")
            loc = geolocator.geocode(adresa)
            if loc:
                st.session_state.lat, st.session_state.lon = loc.latitude, loc.longitude
                st.success(f"Nalezeno: {loc.address}")
            else: st.error("Adresa nenalezena.")
        
        st.write(f"**Souřadnice:** {st.session_state.lat:.4f}, {st.session_state.lon:.4f}")
        if st.button("⬇️ STÁHNOUT DATA PRO TUTO LOKACI", type="primary"):
            with st.spinner("Stahuji data z PVGIS..."):
                url = f"https://re.jrc.ec.europa.eu/api/tmy?lat={st.session_state.lat}&lon={st.session_state.lon}&outputformat=csv"
                resp = requests.get(url)
                if resp.status_code == 200:
                    st.session_state.tmy_df = load_tmy_robust(io.BytesIO(resp.content))
                    st.session_state.tmy_source_label = f"PVGIS (Lat: {st.session_state.lat:.4f}, Lon: {st.session_state.lon:.4f})"
                    st.success("Data stažena!")
                else: st.error("Chyba při stahování dat z PVGIS.")
    with c2:
        m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=15)
        folium.Marker([st.session_state.lat, st.session_state.lon]).add_to(m)
        map_data = st_folium(m, height=300, width=700)
        if map_data and map_data.get("last_clicked"):
            st.session_state.lat, st.session_state.lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
            st.rerun()
else:
    tmy_file = st.file_uploader("Nahrát TMY data", type="csv")
    if tmy_file:
        st.session_state.tmy_df = load_tmy_robust(tmy_file)
        st.session_state.tmy_source_label = "Vlastní CSV soubor"

# --- HLAVNÍ VÝPOČET ---
if st.session_state.tmy_df is not None and df_char is not None:
    tmy = st.session_state.tmy_df.copy()
    tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
    tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
    tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
    
    t_col, v_col, c_col = df_char.columns[0], df_char.columns[1], df_char.columns[2]
    q_tuv_avg = (spotreba_tuv / 8760) * 1000
    potreba_ut_teorie = [max(0, ztrata * (t_vnitrni - t) / (t_vnitrni - t_design)) for t in tmy['T_smooth']]
    k_oprava = spotreba_ut / (sum(potreba_ut_teorie) / 1000) if sum(potreba_ut_teorie) > 0 else 1.0
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
    t_biv_val = -12.0
    for t in np.linspace(15, -15, 500):
        if (np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc) < (max(0, (ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava)) + q_tuv_avg):
            t_biv_val = t; break

    naklady_czt = (spotreba_ut + spotreba_tuv) * (cena_gj_czt * 3.6)
    mwh_el_total = (df_sim['El_tc'].sum() + df_sim['El_biv'].sum()) / 1000
    naklady_tc = (mwh_el_total * cena_el) + servis
    q_tc_s, q_bv_s = df_sim['Q_tc'].sum()/1000, df_sim['Q_biv'].sum()/1000
    el_tc_s, el_bv_s = df_sim['El_tc'].sum()/1000, df_sim['El_biv'].sum()/1000
    df_biv_table = pd.DataFrame({
        "Metrika": ["Tepelná energie (Výstup)", "Spotřeba elektřiny (Vstup)"],
        "TČ [MWh]": [round(q_tc_s, 2), round(el_tc_s, 2)],
        "Bivalence [MWh]": [round(q_bv_s, 2), round(el_bv_s, 2)],
        "Podíl bivalence [%]": [round(q_bv_s/(q_tc_s+q_bv_s)*100, 1), round(el_bv_s/(el_tc_s+el_bv_s)*100, 1)]
    })

    expl_12 = "Graf 1 a 2: Bod bivalence určuje venkovní teplotu, pod kterou musí kaskádě TČ pomáhat bivalentní zdroj."
    expl_34 = "Graf 3 a 4: Měsíční bilance ukazuje sezónní využití zdrojů. Monotóna výkonu vizualizuje časové rozložení potřeby tepla."
    expl_5 = "Graf 5: Seřazená četnost hodinových teplot v roce. Křivka krytí TČ kopíruje potřebu budovy až do bodu bivalence."
    expl_67 = "Graf 6 znázorňuje podíl bivalence na tepelné energii za rok, podíl na spotřebované energii je v tabulce pod grafem. Graf 7 znázorňuje porovnání ročních nákladů."

    st.markdown("---")
    st.header(f"📊 Výsledky projektu: {nazev_projektu}")

    fig12, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    tr = np.linspace(-15, 18, 100)
    q_p = np.array([max(0, (ztrata * (t_vnitrni - t) / (t_vnitrni - t_design) * k_oprava)) + q_tuv_avg for t in tr])
    p_p = np.array([np.interp(t, df_char[t_col], df_char[v_col]) * pocet_tc for t in tr])
    ax1.plot(tr, q_p, 'r-', lw=2, label='Potřeba (ÚT+TUV)')
    ax1.plot(tr, p_p, 'b--', alpha=0.4, label='Max kaskáda TČ')
    ax1.fill_between(tr, p_p, q_p, where=(q_p > p_p), color='red', alpha=0.2, hatch='XXXX', label='Oblast bivalence')
    ax1.axvline(t_biv_val, color='black', linestyle=':', lw=2, label=f'Bod bivalence: {t_biv_val:.1f}°C')
    ax1.set_title("1. DYNAMIKA PROVOZU"); ax1.legend()
    df_sim['Temp_R'] = df_sim['Temp'].round(); df_t = df_sim.groupby('Temp_R')[['Q_tc', 'Q_biv']].sum()
    ax2.bar(df_t.index, df_t['Q_tc'], color='#3498db', label='TČ'); ax2.bar(df_t.index, df_t['Q_biv'], bottom=df_t['Q_tc'], color='#e74c3c', label='Biv')
    ax2.set_title("2. ENERGETICKÝ MIX DLE TEPLOT"); ax2.legend()
    st.pyplot(fig12); st.info(expl_12)

    fig34, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 7))
    df_sim['Month'] = (df_sim.index // (24 * 30.5)).astype(int) + 1; m_df = df_sim.groupby('Month').agg({'Q_tc': 'sum', 'Q_biv': 'sum'})
    ax3.bar(m_df.index, m_df['Q_tc']/1000, color='#ADD8E6', label='TČ'); ax3.bar(m_df.index, m_df['Q_biv']/1000, bottom=m_df['Q_tc']/1000, color='#FF0000', label='Biv')
    ax3.set_title("3. MĚSÍČNÍ BILANCE ENERGIE"); ax3.legend()
    q_sort = np.sort(df_sim['Q_need'].values)[::-1]; p_lim_biv = np.interp(t_biv_val, df_char[t_col], df_char[v_col]) * pocet_tc
    ax4.plot(range(8760), q_sort, 'r-', lw=2); ax4.fill_between(range(8760), 0, np.minimum(q_sort, p_lim_biv), color='#ADD8E6', label='Kryto TČ')
    ax4.fill_between(range(8760), p_lim_biv, q_sort, where=(q_sort > p_lim_biv), color='#FF0000', label='Bivalence')
    ax4.set_title("4. TRVÁNÍ POTŘEBY (MONOTÓNA)"); ax4.legend()
    st.pyplot(fig34); st.info(expl_34)

    fig5, ax5 = plt.subplots(figsize=(18, 5))
    df_st = df_sim.sort_values('Temp').reset_index(drop=True)
    ax5.plot(df_st.index, df_st['Q_need'], 'r', label='Potřeba ÚT+TUV'); ax5.plot(df_st.index, df_st['Q_tc'], 'b', label='Krytí TČ')
    ax5.set_title("5. ČETNOST TEPLOT V ROCE"); ax5.legend()
    st.pyplot(fig5); st.info(expl_5)

    c_l, c_r = st.columns(2)
    with c_l:
        st.subheader("6. Bilance bivalence"); st.table(df_biv_table)
        fig6, ax6 = plt.subplots(figsize=(6, 6)); ax6.pie([q_tc_s, q_bv_s], labels=['TČ', 'Biv'], autopct='%1.1f%%', colors=['#ADD8E6', '#FF0000'])
        ax6.set_title("ROČNÍ PODÍL ENERGIE"); st.pyplot(fig6)
    with c_r:
        st.subheader("7. Ekonomika"); fig7, ax7 = plt.subplots(figsize=(6, 6)); bars = ax7.bar(['CZT', 'TČ'], [naklady_czt, naklady_tc], color=['#95a5a6', '#2ecc71'])
        for bar in bars: ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height()):,} Kč', ha='center', va='bottom')
        ax7.set_title("SROVNÁNÍ NÁKLADŮ [Kč/rok]"); st.pyplot(fig7); st.info(expl_67)

    def generate_pdf_v49():
        pdf = FPDF()
        has_u = os.path.exists(FONT_REGULAR)
        if has_u: pdf.add_font("DejaVu", "", FONT_REGULAR); pdf.add_font("DejaVu", "B", FONT_BOLD); pdf.set_font("DejaVu", "B", 16)
        else: pdf.set_font("Helvetica", "B", 16)
        def cz(txt): return str(txt) if has_u else "".join([c for c in unicodedata.normalize('NFKD', str(txt)) if not unicodedata.combining(c)])

        # Strana 1
        pdf.add_page()
        pdf.cell(0, 10, cz(f"TECHNICKÝ REPORT: {nazev_projektu.upper()}"), ln=True, align="C")
        pdf.set_font(pdf.font_family, "B", 11); pdf.cell(0, 8, cz("VSTUPNÍ PARAMETRY"), ln=True); pdf.set_font(pdf.font_family, "", 10)
        pdf.cell(0, 6, cz(f"- Lokalita: {st.session_state.lat:.4f}, {st.session_state.lon:.4f}"), ln=True)
        pdf.cell(0, 6, cz(f"- Zdroj klimatických dat: {st.session_state.tmy_source_label}"), ln=True)
        pdf.cell(0, 6, cz(f"- Model TČ: {nazev_tc} | Počet v kaskádě: {pocet_tc} | Tepelná ztráta budovy: {ztrata} kW"), ln=True)
        pdf.cell(0, 6, cz(f"- Roční spotřeba ÚT: {spotreba_ut} MWh | TUV: {spotreba_tuv} MWh"), ln=True)
        pdf.cell(0, 6, cz(f"- Teplotní spád: {t_spad} | Bod bivalence: {t_biv_val:.1f} °C"), ln=True)
        
        # FIX: Opravený náhled mapy s viditelným markerem
        try:
            lat, lon = st.session_state.lat, st.session_state.lon
            # Použití jiného stabilnějšího rendereru pro mapy (OpenStreetMap Static)
            map_url = f"https://static-maps.yandex.ru/1.x/?ll={lon},{lat}&z=16&l=map&size=450,450&pt={lon},{lat},pm2rdm"
            m_resp = requests.get(map_url, timeout=10)
            if m_resp.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f_map:
                    f_map.write(m_resp.content)
                    f_map_name = f_map.name
                pdf.image(f_map_name, x=145, y=25, w=50) # Umístění vpravo nahoře
        except:
            pass # Pokud mapa selže, report se vygeneruje bez ní

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1: fig12.savefig(f1.name, dpi=100); pdf.image(f1.name, x=10, y=pdf.get_y()+5, w=190)
        pdf.set_xy(10, pdf.get_y()+85); pdf.set_font(pdf.font_family, "", 8); pdf.multi_cell(0, 4, cz(expl_12))

        # Strana 2 a 3
        pdf.add_page(); 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2: fig34.savefig(f2.name, dpi=100); pdf.image(f2.name, x=10, y=15, w=190)
        pdf.set_xy(10, 90); pdf.multi_cell(0, 5, cz(expl_34))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f5img: fig5.savefig(f5img.name, dpi=100); pdf.image(f5img.name, x=10, y=110, w=190)
        pdf.set_xy(10, 165); pdf.multi_cell(0, 5, cz(expl_5))

        pdf.add_page(); 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f6: fig6.savefig(f6.name, dpi=100); pdf.image(f6.name, x=10, y=15, w=90)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f7: fig7.savefig(f7.name, dpi=100); pdf.image(f7.name, x=105, y=15, w=90)
        pdf.set_xy(10, 105); pdf.set_font(pdf.font_family, "B", 10); pdf.cell(0, 8, cz("TABULKA BILANCE BIVALENCE"), ln=True); pdf.set_font(pdf.font_family, "", 9)
        pdf.cell(0, 5, cz(f"Energie (MWh): TČ {df_biv_table.iloc[0,1]} | Biv {df_biv_table.iloc[0,2]} | Podíl: {df_biv_table.iloc[0,3]}%"), ln=True)
        pdf.cell(0, 5, cz(f"Elektřina (MWh): TČ {df_biv_table.iloc[1,1]} | Biv {df_biv_table.iloc[1,2]} | Podíl: {df_biv_table.iloc[1,3]}%"), ln=True)
        pdf.ln(5); pdf.set_font(pdf.font_family, "", 8); pdf.multi_cell(0, 5, cz(expl_67))
        return bytes(pdf.output())

    if st.sidebar.button("🚀 GENEROVAT PDF REPORT"):
        pdf_data = generate_pdf_v49()
        st.sidebar.download_button("📥 Stáhnout PDF", pdf_data, f"Report_{nazev_projektu}.pdf", "application/pdf")

