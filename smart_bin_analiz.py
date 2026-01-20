import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans

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

# ============================================
# MAKİNE ÖĞRENMESİ - RANDOM FOREST
# ============================================
print("\n" + "=" * 50)
print("MAKİNE ÖĞRENMESİ - RANDOM FOREST")
print("=" * 50)

# Label Encoding - Kategorik değişkenleri sayısala çevirme
le_container = LabelEncoder()
le_recyclable = LabelEncoder()
le_class = LabelEncoder()

df['Container_Encoded'] = le_container.fit_transform(df['Container Type'])
df['Recyclable_Encoded'] = le_recyclable.fit_transform(df['Recyclable fraction'])
df['Class_Encoded'] = le_class.fit_transform(df['Class'])

print("\nLabel Encoding Sonuçları:")
print(f"Container Type: {dict(zip(le_container.classes_, range(len(le_container.classes_))))}")
print(f"Recyclable fraction: {dict(zip(le_recyclable.classes_, range(len(le_recyclable.classes_))))}")
print(f"Class: {dict(zip(le_class.classes_, range(len(le_class.classes_))))}")

# Özellikler ve hedef değişken
X = df[['FL_B', 'FL_A', 'VS', 'Container_Encoded', 'Recyclable_Encoded']]
y = df['Class_Encoded']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nEğitim seti boyutu: {len(X_train)}")
print(f"Test seti boyutu: {len(X_test)}")

# Random Forest Modeli
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

# Tahmin
y_pred = rf_model.predict(X_test)

# Model Performansı
print("\n--- RANDOM FOREST SONUÇLARI ---")
print(f"Doğruluk (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=le_class.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_class.classes_, yticklabels=le_class.classes_)
plt.title('Random Forest - Confusion Matrix', fontsize=14)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.tight_layout()
plt.savefig('5_rf_confusion_matrix.png', dpi=150)
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    'Özellik': X.columns,
    'Önem': rf_model.feature_importances_
}).sort_values('Önem', ascending=False)

print("\nÖzellik Önem Sıralaması:")
print(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Özellik'], feature_importance['Önem'], color='steelblue')
plt.xlabel('Önem Derecesi')
plt.ylabel('Özellik')
plt.title('Random Forest - Feature Importance', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('6_rf_feature_importance.png', dpi=150)
plt.show()

# ============================================
# K-MEANS KÜMELEME
# ============================================
print("\n" + "=" * 50)
print("K-MEANS KÜMELEME ANALİZİ")
print("=" * 50)

# Kümeleme için özellikler - NaN değerleri temizle
X_cluster = df[['FL_B', 'FL_A', 'VS']].dropna().copy()
print(f"\nKümeleme için kullanılan kayıt sayısı: {len(X_cluster)} (NaN'lar çıkarıldı)")

# K-Means Modeli (K=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster)

# Küme etiketlerini orijinal indekslerle eşleştir
df_cluster = df.loc[X_cluster.index].copy()
df_cluster['Cluster'] = cluster_labels

print("\nKüme Merkezleri:")
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, 
                               columns=['FL_B', 'FL_A', 'VS'],
                               index=['Küme 0', 'Küme 1', 'Küme 2'])
print(cluster_centers.round(2))

print("\nKüme Dağılımı:")
print(df_cluster['Cluster'].value_counts().sort_index())

# Küme Bazında İstatistikler
print("\nKüme Bazında Ortalamalar:")
cluster_stats = df_cluster.groupby('Cluster')[['FL_B', 'FL_A', 'VS']].mean().round(2)
print(cluster_stats)

# Küme ve Sınıf İlişkisi
print("\nKüme ve Class İlişkisi:")
cluster_class = pd.crosstab(df_cluster['Cluster'], df_cluster['Class'])
print(cluster_class)

print("\n" + "=" * 50)
print("TÜM ANALİZLER TAMAMLANDI")
print("=" * 50)
