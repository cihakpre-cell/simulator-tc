import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import unicodedata
import tempfile
from fpdf import FPDF

# --- ROBUSTNÍ NAČÍTÁNÍ TMY (Fixuje chybu DataError) ---
def load_tmy_robust(file):
    try:
        content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
        # Najdeme řádek, kde začínají data (přeskočíme hlavičku PVGIS)
        start_idx = -1
        for i, line in enumerate(content):
            if 'time(UTC)' in line or 'T2m' in line:
                start_idx = i
                break
        
        if start_idx == -1: return None

        df = pd.read_csv(io.StringIO("\n".join(content[start_idx:])))
        df.columns = df.columns.str.strip()
        # Vynucení číselného formátu pro teplotu
        df['T2m'] = pd.to_numeric(df['T2m'], errors='coerce')
        return df.dropna(subset=['T2m']).reset_index(drop=True)
    except:
        return None

def load_char(file):
    try:
        content = file.getvalue().decode('utf-8-sig', errors='ignore')
        sep = ';' if ';' in content.split('\n')[0] else ','
        df = pd.read_csv(io.StringIO(content), sep=sep, decimal=',')
        return df[['Teplota', 'Vykon_kW', 'COP']].apply(pd.to_numeric, errors='coerce').dropna()
    except:
        return None

def remove_accents(input_str):
    if not input_str: return ""
    nfkd_form = unicodedata.normalize('NFKD', str(input_str))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="Expertní simulátor TČ", layout="wide")
st.title("📊 Profesionální analýza kaskády TČ")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Vstupní parametry")
    nazev_projektu = st.text_input("Název projektu", "SVJ Sladkovičova")
    ztrata_celkova = st.number_input("Tepelná ztráta [kW]", value=54.0)
    t_design = st.number_input("Návrhová venkovní teplota [°C]", value=-12.0)
    fakt_ut = st.number_input("Reálná spotřeba ÚT [MWh/rok]", value=124.0)
    f_tuv = st.number_input("Reálná spotřeba TUV [MWh/rok]", value=76.0)
    t_privod = st.number_input("Teplota přívod [°C]", value=60)
    t_zpatecka = st.number_input("Teplota zpátečka [°C]", value=50)
    spad_text = f"{int(t_privod)} / {int(t_zpatecka)} °C"
    pocet_tc = st.slider("Počet TČ v kaskádě", 1, 10, 3)
    cena_el_mwh = st.number_input("Cena elektřiny [Kč/MWh]", value=4800.0)
    cena_gj_czt = st.number_input("Cena CZT [Kč/GJ]", value=1284.0)
    investice = st.number_input("Investice [Kč]", value=3800000.0)
    servis = st.number_input("Roční servis [Kč]", value=17000.0)

col_u1, col_u2 = st.columns(2)
with col_u1: tmy_up = st.file_uploader("Nahrajte TMY (CSV)", type="csv")
with col_u2: char_up = st.file_uploader("Nahrajte Charakteristiku (CSV)", type="csv")

if tmy_up and char_up:
    tmy = load_tmy_robust(tmy_up)
    char = load_char(char_up)

    if tmy is not None and char is not None:
        tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()
        q_tuv_avg = (f_tuv / 8760) * 1000
        
        # Kalibrace
        potreba_ut_teorie = [ztrata_celkova * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
        k_oprava = fakt_ut / (sum(potreba_ut_teorie) / 1000)
        naklady_czt_rok = (fakt_ut + f_tuv) * (cena_gj_czt * 3.6)

        # Bod bivalence
        t_biv = -15.0
        for t in np.linspace(15, -15, 500):
            q_need = (ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg
            p_tc = np.interp(t, char['Teplota'], char['Vykon_kW']) * pocet_tc
            if p_tc < q_need:
                t_biv = t
                break

        # Simulace
        res = []
        for t_out, t_smooth in zip(tmy['T2m'], tmy['T_smooth']):
            q_total = max(0, (ztrata_celkova * (20 - t_smooth) / (20 - t_design) * k_oprava)) + q_tuv_avg
            p_max = np.interp(t_out, char['Teplota'], char['Vykon_kW']) * pocet_tc
            cop = np.interp(t_out, char['Teplota'], char['COP'])
            q_tc = min(q_total, p_max)
            q_biv = max(0, q_total - q_tc)
            res.append([round(t_out), q_total, q_tc, q_biv, q_tc/cop if q_tc > 0 else 0, q_biv/0.98])

        df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
        
        # Výpočty pro podíly
        sum_tc = df_sim['Q_tc'].sum() / 1000
        sum_biv = df_sim['Q_biv'].sum() / 1000
        total_e = sum_tc + sum_biv
        p_tc = (sum_tc / total_e) * 100
        p_biv = (sum_biv / total_e) * 100

        naklady_tc = ((df_sim['El_tc'].sum() + df_sim['El_biv'].sum()) / 1000) * cena_el_mwh + servis
        uspora = naklady_czt_rok - naklady_tc

        # --- GRAFICKÝ REPORT ---
        fig = plt.figure(figsize=(16, 12))
        plt.suptitle(f"EXPERTNÍ ANALÝZA: {nazev_projektu}", fontsize=18, fontweight='bold')

        # 1. GRAF (FIX) - Dynamika
        ax1 = plt.subplot(2, 2, 1)
        tr = np.linspace(-15, 18, 100)
        q_p = [(ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg for t in tr]
        p_p = [np.interp(t, char['Teplota'], char['Vykon_kW']) * pocet_tc for t in tr]
        ax1.plot(tr, q_p, color='red', lw=1.5, label='Potřeba domu')
        ax1.plot(tr, p_p, color='blue', lw=1, ls='--', alpha=0.3, label='Max limit')
        ax1.plot(tr, [min(q,p) for q,p in zip(q_p, p_p)], color='green', lw=5, alpha=0.5, label='Modulace TČ')
        ax1.axvline(t_biv, color='black', ls=':', label=f'Bod biv. {t_biv:.1f}°C')
        ax1.set_title("DYNAMIKA PROVOZU A MODULACE", fontweight='bold')
        ax1.legend(loc='lower right', fontsize=8); ax1.grid(alpha=0.2)

        # 2. GRAF (FIX) - Sloupce
        ax2 = plt.subplot(2, 2, 2)
        df_t = df_sim.groupby('Temp')[['Q_tc', 'Q_biv']].sum().sort_index()
        ax2.bar(df_t.index, df_t['Q_tc'], color='#3498db', label='TČ')
        ax2.bar(df_t.index, df_t['Q_biv'], bottom=df_t['Q_tc'], color='#e74c3c', label='Biv.')
        ax2.set_title("DODANÁ ENERGIE DLE TEPLOT", fontweight='bold')
        ax2.legend(fontsize=8); ax2.grid(alpha=0.1, axis='y')

        # 3. GRAF (PŘIDÁN) - Donut podíl bivalence
        ax3 = plt.subplot(2, 2, 3)
        ax3.pie([sum_tc, sum_biv], labels=['TČ', 'Bivalence'], autopct='%1.1f%%', startangle=90, 
                colors=['#3498db', '#e74c3c'], wedgeprops=dict(width=0.45, edgecolor='w'))
        ax3.set_title("ROČNÍ ENERGETICKÉ KRYTÍ", fontweight='bold')
        ax3.text(0, 0, f"{int(total_e)}\nMWh", ha='center', va='center', fontweight='bold')

        # 4. TABULKA (FIX)
        ax4 = plt.subplot(2, 2, 4); ax4.axis('off')
        summ = [["Projekt", nazev_projektu], ["Bivalence", f"{t_biv:.1f} °C"], 
                ["Podíl TČ", f"{p_tc:.1f} %"], ["Podíl Biv.", f"{p_biv:.1f} %"],
                ["Úspora", f"{uspora:,.0f} Kč"]]
        tbl = ax4.table(cellText=summ, loc='center', cellLoc='left', colWidths=[0.4, 0.6])
        tbl.scale(1, 3.2); tbl.set_fontsize(11)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)

        # --- PDF EXPORT ---
        if st.button("📄 Exportovat PDF"):
            pdf = FPDF(); pdf.add_page(); pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(190, 10, f"REPORT: {remove_accents(nazev_projektu)}", ln=True, align='C')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                fig.savefig(tmp.name, dpi=150); pdf.image(tmp.name, x=10, y=35, w=190)
            st.download_button("⬇️ Stáhnout PDF", data=pdf.output(dest='S').encode('latin-1', 'replace'), file_name="Report.pdf")
