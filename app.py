import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files # Pro stažení souboru v Colabu

# --- FUNKCE PRO VÝPOČET VARIANTY ---
def simuluj_variantu(n_strojů, investice_v):
    # Parametry z globálního nastavení
    q_tuv_avg = (f_tuv / 8760) * 1000
    
    # 1. Bod bivalence pro daný počet strojů
    t_biv_v = -99
    for t in np.linspace(15, -15, 500):
        q_need = (ztrata_celkova * (20 - t) / (20 - t_design) * k_oprava) + q_tuv_avg
        p_tc = np.interp(t, df_char['Teplota'], df_char['Vykon_kW']) * n_strojů
        if p_tc < q_need:
            t_biv_v = t
            break

    # 2. Roční simulace
    res = []
    for t_out, t_smooth in zip(tmy['T2m'], tmy['T_smooth']):
        q_total = max(0, ztrata_celkova * (20 - t_smooth) / (20 - t_design) * k_oprava) + q_tuv_avg
        p_max = np.interp(t_out, df_char['Teplota'], df_char['Vykon_kW']) * n_strojů
        cop = np.interp(t_out, df_char['Teplota'], df_char['COP'])
        q_tc = min(q_total, p_max)
        q_biv = q_total - q_tc
        res.append([q_total, q_tc, q_biv, q_tc/cop if q_tc > 0 else 0, q_biv/0.98])
    
    df_v = pd.DataFrame(res, columns=['Q_need', 'Q_tc', 'Q_biv', 'El_tc', 'El_biv'])
    
    # 3. Ekonomika
    el_mwh = (df_v['El_tc'].sum() + df_res['El_biv'].sum()) / 1000
    naklady_v = el_mwh * cena_el_mwh + servis
    uspora_v = naklady_czt - naklady_v
    podil_biv_v = (df_v['Q_biv'].sum() / df_v['Q_need'].sum()) * 100
    
    return {
        "Stroje": n_strojů,
        "Investice": investice_v,
        "Bivalence_T": round(t_biv_v, 1),
        "Podil_Biv_%": round(podil_biv_v, 2),
        "Uspora_rok": round(uspora_v, 0),
        "Navratnost": round(investice_v / uspora_v, 1) if uspora_v > 0 else 0
    }

# --- HLAVNÍ VÝPOČET ---
# (Předpokládáme načtená data z předchozích kroků)
naklady_czt = (fakt_ut + f_tuv) * (cena_gj_czt * 3.6)

# Definujeme varianty pro porovnání (Počet TČ, Odhadovaná investice)
v3 = simuluj_variantu(3, 1800000) # Příklad ceny pro 3ks
v5 = simuluj_variantu(5, 2600000) # Příklad ceny pro 5ks

df_srovnani = pd.DataFrame([v3, v5])

# --- EXPORT DO EXCELU ---
with pd.ExcelWriter('Report_TC_Sladkovicova.xlsx') as writer:
    df_srovnani.to_excel(writer, sheet_name='Srovnani_Variant', index=False)
    # Zde můžete přidat i hodinová data pro jednu z variant
    df_res.to_excel(writer, sheet_name='Hodinova_Data_3ks')

print("SROVNÁVACÍ TABULKA:")
print(df_srovnani.to_string(index=False))

# --- VIZUALIZACE SROVNÁNÍ ---
plt.figure(figsize=(10, 5))
plt.bar(['3 TČ', '5 TČ'], df_srovnani['Navratnost'], color=['green', 'orange'])
plt.ylabel("Doba návratnosti (let)")
plt.title("Srovnání ekonomické efektivity variant")
for i, v in enumerate(df_srovnani['Navratnost']):
    plt.text(i, v + 0.2, f"{v} let", ha='center', fontweight='bold')
plt.show()

# Stažení souboru (pouze v Google Colab)
files.download('Report_TC_Sladkovicova.xlsx')
