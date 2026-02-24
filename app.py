import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# 1. NAČTENÍ VSTUPŮ (Vše v jednom, aby nedošlo k NameError)
try:
    xls = pd.ExcelFile('vstupy_TC.xlsx')
    df_zadani = pd.read_excel(xls, 'Zadani').set_index('Parametr')
    df_char = pd.read_excel(xls, 'Charakteristika')
    
    # Základní parametry
    nazev = df_zadani.loc['Nazev_Projektu', 'Hodnota']
    ztrata_celkova = float(df_zadani.loc['Tepelna_Ztrata', 'Hodnota'])
    t_design = float(df_zadani.loc['Navrhova_Teplota', 'Hodnota'])
    fakt_ut = float(df_zadani.loc['Spotreba_UT_CZT', 'Hodnota'])
    f_tuv = float(df_zadani.loc['Spotreba_TUV_CZT', 'Hodnota'])
    cena_el_mwh = float(df_zadani.loc['Cena_Elektrina_MWh', 'Hodnota'])
    cena_gj_czt = float(df_zadani.loc['Cena_CZT_GJ', 'Hodnota'])
    servis = float(df_zadani.loc['Servisni_Naklady_Rok', 'Hodnota'])

    # Načtení TMY
    tmy = pd.read_csv('tmy_50.024_14.455_2005_2023.csv', skiprows=17)
    tmy.columns = tmy.columns.str.strip()
    tmy['T2m'] = pd.to_numeric(tmy['T2m'], errors='coerce')
    tmy = tmy.dropna(subset=['T2m']).reset_index(drop=True)
    tmy['T_smooth'] = tmy['T2m'].rolling(window=6, min_periods=1).mean()

    # Kalibrace k_oprava a TUV
    q_tuv_avg = (f_tuv / 8760) * 1000
    potreba_ut_teorie = [ztrata_celkova * (20 - t) / (20 - t_design) if t < 20 else 0 for t in tmy['T_smooth']]
    k_oprava = fakt_ut / (sum(potreba_ut_teorie) / 1000)
    naklady_czt = (fakt_ut + f_tuv) * (cena_gj_czt * 3.6)

except Exception as e:
    print(f"Chyba při inicializaci dat: {e}")

# 2. FUNKCE PRO SIMULACI VARIANTY
def analyzuj_variantu(pocet_tc, investice):
    # Výpočet bodu bivalence
    t_biv = -99
    for t in np.linspace(15, -15, 500):
        q_need = (ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg
        p_tc = np.interp(t, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc
        if p_tc < q_need:
            t_biv = t
            break
            
    # Hodinová simulace
    res = []
    for t_out, t_smooth in zip(tmy['T2m'], tmy['T_smooth']):
        q_total = max(0, ztrata_celkova * (20 - t_smooth) / (20 - t_design) * k_oprava) + q_tuv_avg
        p_max = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * pocet_tc
        cop = np.interp(t_out, df_char['Teplota'], df_char['COP'])
        q_tc = min(q_total, p_max)
        q_biv = q_total - q_tc
        res.append([q_total, q_tc, q_biv, q_tc/cop if q_tc > 0 else 0, q_biv/0.98])
    
    df_v = pd.DataFrame(res, columns=['Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
    
    # Výsledky
    el_mwh = (df_v['El_tc'].sum() + df_v['El_biv'].sum()) / 1000
    naklady_tc = el_mwh * cena_el_mwh + servis
    uspora = naklady_czt - naklady_tc
    podil_biv = (df_v['Q_biv'].sum() / df_v['Q_need'].sum()) * 100
    
    return {
        "Varianta": f"{pocet_tc}ks TČ",
        "Investice [Kč]": investice,
        "Bod bivalence [°C]": round(t_biv, 1),
        "Podíl bivalence [%]": round(podil_biv, 2),
        "Úspora [Kč/rok]": round(uspora, 0),
        "Návratnost [let]": round(investice / uspora, 1)
    }, df_v

# 3. VÝPOČET VARIANT (Zadejte reálné odhadované ceny)
v3_data, df_res_3 = analyzuj_variantu(3, 1800000)
v5_data, df_res_5 = analyzuj_variantu(5, 2600000)

df_srovnani = pd.DataFrame([v3_data, v5_data])

# 4. ZOBRAZENÍ VÝSLEDKŮ A GRAFU
print("\n--- SROVNÁNÍ VARIANT NÁVRHU ---")
print(df_srovnani.to_string(index=False))

plt.figure(figsize=(14, 6))

# Histogram četnosti s vyznačením bivalence pro 3ks (horší případ)
ax = plt.subplot(1, 1, 1)
t_min_real = tmy['T2m'].min()
bins = np.arange(np.floor(t_min_real), 20, 1)
n, bins_h, patches = ax.hist(tmy['T2m'], bins=bins, color='skyblue', edgecolor='white', alpha=0.7, label='Četnost teplot (hodin v roce)')

t_biv_3 = v3_data["Bod bivalence [°C]"]
for i in range(len(patches)):
    if bins_h[i] < t_biv_3:
        patches[i].set_facecolor('#ff4444')

ax.axvline(t_biv_3, color='red', ls='--', lw=2, label=f'Bod bivalence 3ks ({t_biv_3}°C)')
ax.set_title(f"STATISTIKA TEPLOT A ROZSAH BIVALENCE (Varianta 3ks TČ)", fontsize=14)
ax.set_xlabel("Teplota [°C]"); ax.set_ylabel("Hodin v roce")
ax.legend()
plt.show()

# 5. EXPORT DO EXCELU
with pd.ExcelWriter('Kompletni_Analyza_SVJ.xlsx') as writer:
    df_srovnani.to_excel(writer, sheet_name='Srovnani_Variant', index=False)
    df_res_3.to_excel(writer, sheet_name='Hodinova_Data_3ks', index=False)
    df_res_5.to_excel(writer, sheet_name='Hodinova_Data_5ks', index=False)

print("\nExport dokončen. Soubor 'Kompletni_Analyza_SVJ.xlsx' je připraven ke stažení.")
files.download('Kompletni_Analyza_SVJ.xlsx')
