import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Smart_Bin.csv")

df.head()
df.info()
df.describe()
df.isnull().sum()

# kategorik değişkenler
df['Container Type'].unique()
df['Recyclable fraction'].unique()
df['Class'].unique()

df['Container Type'].value_counts()
df['Recyclable fraction'].value_counts()
df['Class'].value_counts()

# pivot tablolar
pivot_fl_b = pd.pivot_table(df, values='FL_B', index='Container Type', 
                            columns='Recyclable fraction', aggfunc='mean')
pivot_fl_b = pivot_fl_b.round(2)
print("FL_B Ortalaması:")
print(pivot_fl_b)

pivot_fl_a = pd.pivot_table(df, values='FL_A', index='Container Type', 
                            columns='Recyclable fraction', aggfunc='mean')
pivot_fl_a = pivot_fl_a.round(2)
print("\nFL_A Ortalaması:")
print(pivot_fl_a)

pivot_degisim = pivot_fl_b - pivot_fl_a
pivot_degisim = pivot_degisim.round(2)
print("\nDoluluk Değişimi:")
print(pivot_degisim)

konteyner_ort = df.groupby('Container Type')['FL_B'].mean().sort_values(ascending=False)
print("\nKonteyner Türü Bazında FL_B Ortalaması:")
print(konteyner_ort.round(2))

atik_ort = df.groupby('Recyclable fraction')['FL_B'].mean().sort_values(ascending=False)
print("\nAtık Türü Bazında FL_B Ortalaması:")
print(atik_ort.round(2))

sinif_analiz = df.groupby('Class').agg({
    'FL_B': ['mean', 'std', 'count'],
    'FL_A': ['mean', 'std']
}).round(2)
print("\nSınıf Bazında Analiz:")
print(sinif_analiz)

sayisal_kolonlar = ['FL_B', 'FL_A', 'VS', 'FL_B_3', 'FL_A_3', 'FL_B_12', 'FL_A_12']
korelasyon = df[sayisal_kolonlar].corr().round(3)
print("\nKorelasyon Matrisi:")
print(korelasyon)

print("\nEn Yüksek FL_B Değerleri:")
for konteyner in pivot_fl_b.index:
    for atik in pivot_fl_b.columns:
        deger = pivot_fl_b.loc[konteyner, atik]
        if deger > 71:
            print(f"  {konteyner} + {atik}: {deger}")

print("\n--- SONUÇLAR ---")
print(f"Toplam kayıt sayısı: {len(df)}")
print(f"Konteyner türü sayısı: {df['Container Type'].nunique()}")
print(f"En yüksek FL_B: {pivot_fl_b.values.max()}")
print(f"En düşük FL_B: {pivot_fl_b.values.min()}")

# grafikler
pivot1 = df.pivot_table(index='Container Type', values='FL_B', aggfunc='mean').reset_index()
pivot1 = pivot1.sort_values(by='FL_B', ascending=True)

plt.figure(figsize=(12, 7))
renkler = plt.cm.Greens(np.linspace(0.3, 0.9, len(pivot1)))
plt.barh(pivot1['Container Type'], pivot1['FL_B'], color=renkler)
plt.title('Konteyner Türlerine Göre Ortalama Doluluk Oranı', fontsize=14)
plt.xlabel('Ortalama FL_B Değeri')
plt.ylabel('Konteyner Türü')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('1_konteyner_doluluk.png', dpi=150)
plt.show()

pivot2 = df.pivot_table(index='Recyclable fraction', values='FL_B', aggfunc='mean').reset_index()

plt.figure(figsize=(8, 8))
renkler2 = ['#3498db', '#e74c3c', '#2ecc71']
plt.pie(pivot2['FL_B'], labels=pivot2['Recyclable fraction'], autopct='%1.1f%%', 
        colors=renkler2, explode=[0.05, 0, 0], shadow=True, startangle=90)
plt.title('Atık Türlerinin Doluluk Oranı Dağılımı', fontsize=14)
plt.tight_layout()
plt.savefig('2_atik_doluluk.png', dpi=150)
plt.show()

pivot3 = df.pivot_table(index='Class', values=['FL_B', 'FL_A'], aggfunc='mean').reset_index()

x = np.arange(len(pivot3))
genislik = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
bar1 = ax.bar(x - genislik/2, pivot3['FL_B'], genislik, label='FL_B (Sonra)', color='#9b59b6')
bar2 = ax.bar(x + genislik/2, pivot3['FL_A'], genislik, label='FL_A (Önce)', color='#f39c12')

ax.set_xlabel('Sınıf')
ax.set_ylabel('Doluluk Oranı')
ax.set_title('Sınıf Bazında Doluluk Karşılaştırması (FL_B vs FL_A)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(pivot3['Class'])
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('3_sinif_doluluk.png', dpi=150)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_fl_b, annot=True, fmt='.1f', cmap='BuPu', linewidths=0.5, linecolor='white')
plt.title('Konteyner ve Atık Türü Kombinasyonlarının Doluluk Analizi', fontsize=14)
plt.xlabel('Geri Dönüşüm Türü')
plt.ylabel('Konteyner Tipi')
plt.tight_layout()
plt.savefig('4_pivot_heatmap.png', dpi=150)
plt.show()

print("\nGrafikler kaydedildi.")
