# Akilli Cop Kutusu Veri Analizi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans

# Veriyi yukle
df = pd.read_csv("Smart_Bin.csv")

# Veriye bakalim
print("Veri Seti:")
print(df.head())
print(f"\nToplam kayit: {len(df)}")

# Eksik veri kontrolu
print("\nEksik veriler:")
print(df.isnull().sum())

# Kategorik degiskenlere bakalim
print("\nKonteyner turu sayisi:", df['Container Type'].nunique())
print("Atik turleri:", df['Recyclable fraction'].unique())

# Pivot tablo
pivot = pd.pivot_table(df, values='FL_B', index='Container Type', 
                       columns='Recyclable fraction', aggfunc='mean')
pivot = pivot.round(2)
print("\nDoluluk Ortalamasi:")
print(pivot)

# Korelasyon
kor = df[['FL_B', 'FL_A', 'VS']].corr().round(2)
print("\nKorelasyon:")
print(kor)

# Grafik 1 - Konteyner doluluk
plt.figure(figsize=(10, 6))
ort = df.groupby('Container Type')['FL_B'].mean().sort_values()
plt.barh(ort.index, ort.values, color='green')
plt.xlabel('Doluluk Orani')
plt.title('Konteyner Turlerine Gore Doluluk')
plt.tight_layout()
plt.savefig('1_konteyner_doluluk.png')
plt.show()

# Grafik 2 - Sinif karsilastirma
plt.figure(figsize=(8, 5))
sinif_ort = df.groupby('Class')['FL_B'].mean()
plt.bar(sinif_ort.index, sinif_ort.values, color=['purple', 'orange'])
plt.ylabel('Doluluk Orani')
plt.title('Sinif Bazinda Doluluk')
plt.tight_layout()
plt.savefig('3_sinif_doluluk.png')
plt.show()

# MAKINE OGRENMESI
print("\n" + "="*40)
print("RANDOM FOREST")
print("="*40)

# Label Encoding
le = LabelEncoder()
df['Container_Enc'] = le.fit_transform(df['Container Type'])
df['Recyclable_Enc'] = le.fit_transform(df['Recyclable fraction'])
df['Class_Enc'] = le.fit_transform(df['Class'])

# X ve y
X = df[['FL_B', 'FL_A', 'VS', 'Container_Enc', 'Recyclable_Enc']]
y = df['Class_Enc']

# Egitim test ayirma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Egitim: {len(X_train)}, Test: {len(X_test)}")

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahmin
tahmin = model.predict(X_test)

# Model Performans Metrikleri
print("\n--- MODEL PERFORMANSI ---")
basari = accuracy_score(y_test, tahmin)
precision = precision_score(y_test, tahmin)
recall = recall_score(y_test, tahmin)
f1 = f1_score(y_test, tahmin)

print(f"Accuracy (Dogruluk): %{basari*100:.1f}")
print(f"Precision (Kesinlik): %{precision*100:.1f}")
print(f"Recall (Duyarlilik): %{recall*100:.1f}")
print(f"F1-Score: %{f1*100:.1f}")

# KNN (K-Nearest Neighbors)
print("\n" + "="*40)
print("KNN (K-EN YAKIN KOMSU)")
print("="*40)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_tahmin = knn.predict(X_test)

print("\n--- KNN PERFORMANSI ---")
knn_basari = accuracy_score(y_test, knn_tahmin)
knn_precision = precision_score(y_test, knn_tahmin)
knn_recall = recall_score(y_test, knn_tahmin)
knn_f1 = f1_score(y_test, knn_tahmin)

print(f"Accuracy (Dogruluk): %{knn_basari*100:.1f}")
print(f"Precision (Kesinlik): %{knn_precision*100:.1f}")
print(f"Recall (Duyarlilik): %{knn_recall*100:.1f}")
print(f"F1-Score: %{knn_f1*100:.1f}")

# Model Karsilastirma
print("\n" + "="*40)
print("MODEL KARSILASTIRMA")
print("="*40)
print(f"Random Forest Accuracy: %{basari*100:.1f}")
print(f"KNN Accuracy: %{knn_basari*100:.1f}")
if basari > knn_basari:
    print("Random Forest daha iyi!")
else:
    print("KNN daha iyi!")

# K-MEANS
print("\n" + "="*40)
print("K-MEANS KUMELEME")
print("="*40)

X_kume = df[['FL_B', 'FL_A', 'VS']].dropna()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kumeler = kmeans.fit_predict(X_kume)

print("Kume Merkezleri:")
print(pd.DataFrame(kmeans.cluster_centers_, columns=['FL_B', 'FL_A', 'VS']).round(2))

print("\nKume Dagilimi:")
print(pd.Series(kumeler).value_counts().sort_index())

print("\nANALIZ TAMAMLANDI")
