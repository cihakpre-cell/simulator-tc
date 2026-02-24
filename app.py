import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import os

# --- 1. NAČTENÍ VSTUPŮ ---
file_name = 'vstupy_TC.xlsx'
try:
    df_zadani = pd.read_excel(file_name, sheet_name='Zadani').set_index('Parametr')
    df_char = pd.read_excel(file_name, sheet_name='Charakteristika')
except:
    df_zadani = pd.read_csv('vstupy_TC.xlsx - Zadani.csv').set_index('Parametr')
    df_char = pd.read_csv('vstupy_TC.xlsx - Charakteristika.csv')

def get_v(key, default=0):
    return df_zadani.loc[key, 'Hodnota'] if key in df_zadani.index else default

# Parametry
nazev = get_v('Nazev_Projektu', 'Projekt SVJ')
ztrata_celkova = float(get_v('Tepelna_Ztrata', 54))
t_design = float(get_v('Navrhova_Teplota', -12))
fakt_ut = float(get_v('Spotreba_UT_CZT', 124))
f_tuv = float(get_v('Spotreba_TUV_CZT', 76))
cena_el_mwh = float(get_v('Cena_Elektrina_MWh', 4800))
cena_gj_czt = float(get_v('Cena_CZT_GJ', 1284))
investice = float(get_v('Investice_CAPEX', 3800000))
servis = float(get_v('Servisni_Naklady_Rok', 17000))
pocet_tc = int(get_v('Pocet_TC_v_Kaskade', 3))
t_privod = get_v('Teplota_Privod_Design', 60)
t_zpatecka = get_v('Teplota_Zpatecka_Design', 50)
spad_text = f"{int(t_privod)} / {int(t_zpatecka)} °C"

# Načtení TMY
tmy_file = [f for f in os.listdir('.') if f.startswith('tmy') and f.endswith('.csv')][0]
tmy = pd.read_csv(tmy_file, skiprows=17)
tmy.columns = tmy.columns.str.strip()
tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()

# Kalibrace
q_tuv_avg = (f_tuv / 8760) * 1000
potreba_ut_teorie = [ztrata_celkova * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
k_oprava = fakt_ut / (sum(potreba_ut_teorie) / 1000)
naklady_czt_rok = (fakt_ut + f_tuv) * (cena_gj_czt * 3.6)

# --- 2. VÝPOČET BIVALENCE ---
t_biv = -99
for t in np.linspace(15, -15, 500):
    q_need = (ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg
    p_tc = np.interp(t, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc
    if p_tc < q_need:
        t_biv = t
        break

# --- 3. SIMULACE ---
res = []
for t_out, t_smooth in zip(tmy['T2m'], tmy['T_smooth']):
    q_total = max(0, (ztrata_celkova * (20 - t_smooth) / (20 - t_design) * k_oprava)) + q_tuv_avg
    p_max = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc
    cop = np.interp(t_out, df_char['Teplota'], df_char['COP'])
    q_tc = min(q_total, p_max)
    q_biv = q_total - q_tc
    res.append([t_out, q_total, q_tc, q_biv, q_tc/cop if q_tc > 0 else 0, q_biv/0.98])

df_sim = pd.DataFrame(res, columns=['Temp', 'Q_need_kW', 'Q_tc_kW', 'Q_biv_kW', 'El_tc_kW', 'El_biv_kW'])
naklady_tc = (df_sim['El_tc_kW'].sum() + df_sim['El_biv_kW'].sum()) / 1000 * cena_el_mwh + servis
uspora = naklady_czt_rok - naklady_tc

# --- 4. REPORT S ROZŠÍŘENÝM GRAFEM ---
plt.figure(figsize=(16, 12))
plt.suptitle(f"EXPERTNÍ ANALÝZA: {nazev}", fontsize=18, fontweight='bold')

# A. Výkonová charakteristika s modulací a bivalencí
ax1 = plt.subplot(2, 2, 1)
t_r = np.linspace(-15, 18, 100)
q_domu = [(ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg for t in t_r]
p_kaskady = [np.interp(t, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc for t in t_r]
# Provozní výkon TČ (vlevo od bivalence = max výkon, vpravo = potřeba domu)
p_provoz_tc = [min(q, p) for q, p in zip(q_domu, p_kaskady)]

ax1.plot(t_r, q_domu, 'r-', lw=2, label='Potřeba domu (UT + TUV)')
ax1.plot(t_r, p_kaskady, 'b--', alpha=0.5, label='Max. potenciál kaskády')
ax1.plot(t_r, p_provoz_tc, 'g-', lw=3, label='Reálný dodávaný výkon TČ')

# Zvýraznění bivalentního dohřevu (šrafování mezi potřebou a výkonem TČ v mrazu)
t_mraz = np.linspace(-15, t_biv, 50)
q_mraz = [(ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg for t in t_mraz]
p_mraz = [np.interp(t, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc for t in t_mraz]
ax1.fill_between(t_mraz, p_mraz, q_mraz, color='red', alpha=0.3, hatch='//', label='Bivalentní dohřev')

ax1.axvline(t_biv, color='k', ls=':', label=f'Bod bivalence {t_biv:.1f}°C')
ax1.set_title("VÝKONOVÁ ROVNOVÁHA A MODULACE", fontweight='bold'); ax1.legend(); ax1.grid(alpha=0.3)
ax1.set_xlabel("Venkovní teplota [°C]"); ax1.set_ylabel("Výkon [kW]")

# B. Histogram
ax2 = plt.subplot(2, 2, 2)
n, bins, patches = ax2.hist(tmy['T2m'], bins=np.arange(-15, 20, 1), color='skyblue', edgecolor='white')
for i in range(len(patches)):
    if bins[i] < t_biv: patches[i].set_facecolor('#ff4444')
ax2.annotate('Zóna bivalentního\ndohřevu', xy=(t_biv-1, 50), xytext=(t_biv-8, 400),
             arrowprops=dict(facecolor='red', shrink=0.05), color='red', fontweight='bold')
ax2.set_title("ČETNOST TEPLOT V ROCE", fontweight='bold')

# C. Ekonomika
ax3 = plt.subplot(2, 2, 3)
ax3.bar(['CZT', f'TČ ({pocet_tc}ks)'], [naklady_czt_rok, naklady_tc], color=['gray', '#2ecc71'])
ax3.set_title(f"ROČNÍ NÁKLADY (SPÁD {spad_text})", fontweight='bold')
for i, v in enumerate([naklady_czt_rok, naklady_tc]): ax3.text(i, v+10000, f"{v:,.0f} Kč", ha='center', fontweight='bold')

# D. Tabulka
ax4 = plt.subplot(2, 2, 4); ax4.axis('off')
summary = [
    ["Teplotní spád otopné soustavy", spad_text],
    ["Vypočtený bod bivalence", f"{t_biv:.1f} °C"],
    ["Energie z bivalence (dohřev)", f"{(df_sim['Q_biv_kW'].sum()/df_sim['Q_need_kW'].sum())*100:.2f} %"],
    ["Roční úspora", f"{uspora:,.0f} Kč"],
    ["Návratnost (prostá)", f"{investice/uspora:.1f} let"]
]
tbl = ax4.table(cellText=summary, loc='center', cellLoc='left', colWidths=[0.6, 0.3])
tbl.auto_set_font_size(False); tbl.set_fontsize(13); tbl.scale(1, 3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'Report_{pocet_tc}ks.png')
plt.show()

# --- 5. EXPORT ---
legenda_data = [['Temp', 'Venkovní teplota [°C]'], ['Q_need_kW', 'Potřeba UT+TUV [kW]'], ['Q_tc_kW', 'Reálný výkon TČ (modulovaný) [kW]'], ['Q_biv_kW', 'Výkon dohřevu [kW]'], ['El_tc_kW', 'Příkon TČ [kW]'], ['El_biv_kW', 'Příkon dohřevu [kW]']]
with pd.ExcelWriter(f'Analyza_{pocet_tc}ks.xlsx') as writer:
    pd.DataFrame(summary, columns=['Parametr', 'Hodnota']).to_excel(writer, sheet_name='Souhrn', index=False)
    df_sim.to_excel(writer, sheet_name='Hodinova_Simulace', index=False)
    pd.DataFrame(legenda_data, columns=['Zkratka', 'Význam']).to_excel(writer, sheet_name='Legenda', index=False)

files.download(f'Report_{pocet_tc}ks.png')
files.download(f'Analyza_{pocet_tc}ks.xlsx')
