import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. NAČTENÍ A PŘÍPRAVA DAT ---
xls = pd.ExcelFile('vstupy_TC.xlsx')
df_zadani = pd.read_excel(xls, 'Zadani').set_index('Parametr')
df_char = pd.read_excel(xls, 'Charakteristika')

# Parametry
nazev = df_zadani.loc['Nazev_Projektu', 'Hodnota']
ztrata_celkova = float(df_zadani.loc['Tepelna_Ztrata', 'Hodnota'])
t_design = float(df_zadani.loc['Navrhova_Teplota', 'Hodnota'])
pocet_tc = int(df_zadani.loc['Pocet_TC_v_Kaskade', 'Hodnota'])
investice = float(df_zadani.loc['Investice_CAPEX', 'Hodnota'])
cena_el_mwh = float(df_zadani.loc['Cena_Elektrina_MWh', 'Hodnota'])
cena_gj_czt = float(df_zadani.loc['Cena_CZT_GJ', 'Hodnota'])
fakt_ut = float(df_zadani.loc['Spotreba_UT_CZT', 'Hodnota'])
f_tuv = float(df_zadani.loc['Spotreba_TUV_CZT', 'Hodnota'])
servis = float(df_zadani.loc['Servisni_Naklady_Rok', 'Hodnota'])

# TMY data
tmy = pd.read_csv('tmy_50.024_14.455_2005_2023.csv', skiprows=17)
tmy.columns = tmy.columns.str.strip()
tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()

# Kalibrace (k_oprava)
q_tuv_avg = (f_tuv / 8760) * 1000
potreba_ut_teorie = [ztrata_celkova * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
k_oprava = fakt_ut / (sum(potreba_ut_teorie) / 1000)

# --- 2. VÝPOČET BODU BIVALENCE ---
t_biv = -99
for t in np.linspace(15, -15, 500):
    q_need = (ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg
    p_tc = np.interp(t, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc
    if p_tc < q_need:
        t_biv = t
        break

# --- 3. SIMULACE ROKU ---
results = []
for t_out, t_smooth in zip(tmy['T2m'], tmy['T_smooth']):
    q_total = max(0, ztrata_celkova * (20 - t_smooth) / (20 - t_design) * k_oprava) + q_tuv_avg
    p_max = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc
    cop = np.interp(t_out, df_char['Teplota'], df_char['COP'])
    q_tc = min(q_total, p_max)
    q_biv = q_total - q_tc
    results.append([q_total, q_tc, q_biv, q_tc/cop if q_tc > 0 else 0, q_biv/0.98])

df_res = pd.DataFrame(results, columns=['Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])

# --- 4. EKONOMIKA ---
naklady_czt = (fakt_ut + f_tuv) * (cena_gj_czt * 3.6)
naklady_tc = (df_res['El_tc'].sum() + df_res['El_biv'].sum()) / 1000 * cena_el_mwh + servis
uspora = naklady_czt - naklady_tc
podil_biv = (df_res['Q_biv'].sum() / df_res['Q_need'].sum()) * 100

# --- 5. VIZUALIZACE ---
plt.figure(figsize=(16, 12))

# GRAF A: Výkonová rovnováha
ax1 = plt.subplot(2, 2, 1)
t_range = np.linspace(-15, 16, 100)
q_range = [(ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg for t in t_range]
p_range = [np.interp(t, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc for t in t_range]
ax1.plot(t_range, q_range, 'r-', lw=2, label='Tepelné zatížení budovy')
ax1.plot(t_range, p_range, 'b--', lw=2, label=f'Max. výkon TČ ({pocet_tc}ks)')
ax1.fill_between(t_range, q_range, p_range, where=[(q > p) for q, p in zip(q_range, p_range)], color='red', alpha=0.2, label='Oblast deficitu (Bivalence)')
ax1.axvline(t_biv, color='black', ls=':')
ax1.set_title("VÝKONOVÁ CHARAKTERISTIKA", fontweight='bold')
ax1.set_xlabel("Teplota [°C]"); ax1.set_ylabel("Výkon [kW]"); ax1.legend(); ax1.grid(alpha=0.3)

# GRAF B: Histogram četnosti s vyznačením mrazů
ax2 = plt.subplot(2, 2, 2)
t_min_real = tmy['T2m'].min()
bins = np.arange(np.floor(t_min_real), 20, 1)
n, bins_h, patches = ax2.hist(tmy['T2m'], bins=bins, color='skyblue', edgecolor='white', alpha=0.8)
for i in range(len(patches)):
    if bins_h[i] < t_biv:
        patches[i].set_facecolor('#ff4444') # Červená pro mrazy pod bivalencí
ax2.set_xlim(-15, 20)
ax2.axvline(t_biv, color='red', ls='--')
ax2.set_title("ROZLOŽENÍ TEPLOT V ROCE (TMY)", fontweight='bold')
ax2.set_xlabel("Venkovní teplota [°C]"); ax2.set_ylabel("Počet hodin"); ax2.grid(axis='y', alpha=0.2)
ax2.annotate('Zóna bivalence\n(červená)', xy=(t_biv-2, 50), xytext=(t_biv-10, 400),
             arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.5), color='red', fontweight='bold')

# TABULKA VÝSLEDKŮ
ax3 = plt.subplot(2, 2, (3, 4))
ax3.axis('off')
table_data = [
    ["Projekt", nazev],
    ["Počet TČ v kaskádě", f"{pocet_tc} ks"],
    ["Celková investice", f"{investice:,.0f} Kč"],
    ["Skutečný bod bivalence", f"{t_biv:.1f} °C"],
    ["PODÍL BIVALENCE NA ENERGII (kWh)", f"{podil_biv:.2f} %"],
    ["Roční úspora oproti CZT", f"{uspora:,.0f} Kč/rok"],
    ["Návratnost investice", f"{investice/uspora:.1f} let"]
]
tbl = ax3.table(cellText=table_data, loc='center', cellLoc='left', colWidths=[0.4, 0.4])
tbl.set_fontsize(14); tbl.scale(1, 2.5)

plt.tight_layout()
plt.show()

print(f"Hotovo. Bod bivalence {t_biv:.1f}°C. Skutečné minimum v datech: {t_min_real:.1f}°C.")
