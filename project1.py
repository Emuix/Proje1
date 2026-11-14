from statistics import kde
import sklearn  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
df = pd.read_csv("HeartDiseaseCSV.csv")


# Sütunlar: id, age, sex, dataset, cp, trestbps, chol, fbs, restecg, thalch, exang ,oldpeak , slope, ca, thal, num
# id -> Her hasta için benzersiz bir kimlik numarası, age -> hastanın yaşı, sex-> hastanın cinsiyeti, dataset -> veri seti türü, cp -> göğüs ağrısı tipi, trestbps -> dinlenme kan basıncı, chol -> kolesterol seviyesi, fbs -> açlık kan şekeri, restecg -> dinlenme elektrokardiyogramı, thalch -> maksimum kalp atış hızı, exang -> egzersiz sırasında anjina, oldpeak -> ST depresyonu, slope -> ST segmentinin eğimi, ca -> büyük damarların sayısı, thal -> talasemi durumu, num -> kalp hastalığı var mı yok mu? Varsa ne kadar şiddetli? (0 -> yok, 1-4 -> var ve şiddeti)

print(df["num"].value_counts())
# Veri dengeli: 0 -> 411, 1 -> 265, 2-> 109, 3 -> 107, 4 -> 28 ; 0(Kalp hastalığı yok) -> 411, (1,2,3,4) (Kalp hastalığı var) -> 

print(df.isna().sum()) # Eksik veri var

df["target"] = df["num"].apply(lambda x: 0 if x == 0 else 1) # Kalp hastalığı var mı yok mu? (0 -> yok, 1 -> var)
cols_to_drop = ['id', 'dataset', 'num', 'ca', 'thal', 'slope'] # -> Fazla eksik veri olan veya gereksiz olan sütunları düşürdük
df = df.drop(columns=cols_to_drop)

df['chol'] = df['chol'].replace(0, np.nan) # Kolesterol seviyesi 0 olanları eksik veri olarak işaretledik
df['trestbps'] = df['trestbps'].replace(0, np.nan) # Dinlenme kan basıncı 0 olanları eksik veri olarak işaretledik

num_cols = ['trestbps', 'chol', 'thalch', 'oldpeak'] # Sayısal sütunlar
for col in num_cols:
    df[col] = df[col].fillna(df[col].median()) # Eksik verileri medyan ile doldurduk

cat_cols = ['fbs', 'restecg', 'exang'] # Kategorik sütunlar

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0]) # Eksik verileri mod ile doldurduk


print("Kalan eksik veri sayısı: \n", df.isnull().sum()) # Kalan eksik veri sayısı

# Görselleştirme
# 1 Cinsiyete göre kalp hastalığı dağılımı
plt.figure(figsize=(8,5))
sns.countplot(x='sex', hue='target', data=df, palette='coolwarm')
plt.title('Cinsiyete Göre Kalp Hastalığı Dağılımı')
plt.xlabel('Cinsiyet')
plt.ylabel('Kişi Sayısı')
plt.legend(title='Kalp Hastalığı', labels=['Yok', 'Var'])
plt.show()

# 2 Yaşa göre kalp hastalığı dağılımı
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='age', hue='target', kde = True, palette='magma', bins=25, element='step')
plt.title('Yaşa Göre Kalp Hastalığı Dağılımı')
plt.xlabel('Yaş')
plt.ylabel('Kişi Sayısı')
plt.show()

# 3 Göğüs ağrısı tipine göre kalp hastalığı dağılımı
plt.figure(figsize=(9, 5))
sns.countplot(x='cp', hue='target', data=df, palette='viridis')
plt.title('Göğüs Ağrısı Tipine (cp) Göre Kalp Hastalığı Oranı')
plt.xlabel('Göğüs Ağrısı Tipi')
plt.ylabel('Kişi Sayısı')
plt.legend(['Sağlıklı (0)', 'Hasta (1)'])
plt.show()

# 4 Maksimum kalp atış hızına göre kalp hastalığı dağılımı
plt.figure(figsize=(8,5))
sns.boxplot(x='target', y='thalch', data=df, palette='Set2')
plt.title('Hastalık Durumuna Göre Maksimum Kalp Atış Hızı Dağılımı')
plt.xlabel('Kalp Hastalığı Durumu')
plt.ylabel('Maksimum Kalp Atış Hızı (thalch)')
plt.show()

# 5 Sayısal sütunlar arasındaki korelasyon
plt.figure(figsize=(8,6))
numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'target']
corr = df[numerical_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Sayısal Değişkenler Arasındaki Korelasyon Isı Haritası')
plt.show()


df_encoded = pd.get_dummies(df, drop_first=True, dtype=int) # Kategorik sütunları one-hot encoding ile dönüştürdük


x = df_encoded.drop('target', axis=1) # Özellikler
y = df_encoded['target'] # Hedef değişken



x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=42) # Veriyi eğitim ve test olarak ayırdık
print("Eğitim verisi boyutu: ", len(x_train))
print("Test verisi boyutu: ", len(x_test))

# Lojistik Regresyon Modeli
log_model = LogisticRegression(max_iter=1000) # Lojistik regresyon modeli oluşturduk
log_model.fit(x_train, y_train) # Modeli eğittik
y_pred = log_model.predict(x_test) # Test verisi ile tahmin yaptık

# Decision Tree Modeli
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42) # Karar ağacı modeli oluşturduk
tree_model.fit(x_train, y_train) # Modeli eğittik
y_pred_tree = tree_model.predict(x_test) # Test verisi ile tahmin yaptık

print("Lojistik Regresyon Modeli Başarı Oranı: ", accuracy_score(y_test, y_pred))
print("precision: ", precision_score(y_test, y_pred))
print("recall: ", recall_score(y_test, y_pred))

print("\n--- Sınıflandırma Raporu (Lojistik Regresyon) ---\n", classification_report(y_test, y_pred))
# --- KARAR AĞACI (DECISION TREE) SONUÇLARI ---
print("\n==========================================")
print("Karar Ağacı Modeli Başarı Oranı (Accuracy): ", accuracy_score(y_test, y_pred_tree))
print("Karar Ağacı Precision (Kesinlik): ", precision_score(y_test, y_pred_tree))
print("Karar Ağacı Recall (Duyarlılık): ", recall_score(y_test, y_pred_tree))

print("\n--- Sınıflandırma Raporu (Karar Ağacı) ---\n", classification_report(y_test, y_pred_tree))
print("Karmaşıklık Matrisi (Confusion Matrix):\n", confusion_matrix(y_test, y_pred_tree))