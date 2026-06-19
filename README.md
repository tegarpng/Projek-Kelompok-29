# Klasifikasi Citra MRI Otak untuk Membedakan Tumor Glioma dan Non-Tumor Menggunakan Ekstraksi Fitur GLCM serta Perbandingan Metode KNN, SVM, dan Random Forest
## Nama Anggota
###  M. Alfatih : F1D02410013
###  Muhammad Tegar Bijanta : F1D02410081
###  Ridho Hidayat : F1D02410089

# Deskripsi Projek
Tumor otak merupakan salah satu penyakit yang paling berbahaya dan mengancam jiwa manusia. Berdasarkan data dari World Health Organization (WHO), tumor otak termasuk dalam 10 besar penyebab kematian akibat kanker di seluruh dunia. Deteksi dini sangat krusial karena semakin cepat tumor teridentifikasi, semakin besar peluang pasien untuk mendapatkan penanganan yang tepat dan meningkatkan angka harapan hidup. Oleh karena itu, kami ingin membuat klasifikasi citra MRI tentang perbedaan otak yang terkena kanker glioma dan yang sehat. Ini diharapkan dapat menjadi deteksi dini terkait tumor glioma pada otak manusia dalam pengecekan MRI. Nanti data MRI ini akan diolah sedemikian rupa untuk membuat model mesin yang mempelajari klasifikasi perbedaan dari tumor glioma dan yang sehat.

Project ini akan menggunakan total 4 percobaan, maka:
- Percobaan Pertama (3 Preprocessing menggunakan grayscale, resize, CLAHE (Contrast Limited Adaptive Histogram EqualizatioN))
- Percobaan Kedua (4 Preprocessing menggunakan grayscale, resize, CLAHE (Contrast Limited Adaptive Histogram EqualizatioN), dan median filter)
- Percobaan Ketiga (5 Preprocessing menggunakan grayscale, resize, median filter, sobel, dan roberts)
- Percobaan Keempat (5 Preprocessing menggunakan grayscale, resize, median filter, CLAHE(Contrast Limited Adaptive Histogram Equalization)dan thresholding)

# IMPORT LIBRARY
``` python
  import os
  import matplotlib.pyplot as plt
  import cv2 as cv
  import numpy as np
  import pandas as pd
  from sklearn.model_selection import train_test_split, cross_val_predict
  from sklearn.metrics import accuracy_score, classification_report
  from skimage.feature import graycomatrix, graycoprops
  from scipy.stats import entropy
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.svm import SVC
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
  from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay)
  import seaborn as sns
```
# Load Data
Setelah import library, dilanjutkan dengan tahapan membaca dataset. Pada data ini digunakan 300 data pada tiap kelas. Dan pada setiap kelas citra akan dikenakan grayscale dan resize sebesar 256x256, ini dibuat agar ukuran citra yang diproses akan memiliki ukuran yang sama. Pada load data ini, digunakan data yang urut dari data 1 hingga seterusnya (data ini diurutkan pada proses perulangan di kode).
``` python
  data = 600
  labels = glioma dan tumor
```
## Data Understanding
Pada data ini digunakan data yang memiliki background yang semuanya hitam, ini mempermudah dalam penggunaan percobaan yang akan dilakukan. Noise pada data ini memiliki noise tidak terlalu banyak. Distribusi data dari setiap kelas juga sudah ditentukan sebanyak 300. 
``` python
	unique, counts = np.unique(labels, return_counts=True)
	for u, c in zip(unique, counts):
	    print(f"Kelas {u}: {c} gambar")
	print(f"Total data: {len(data)} gambar")
```
Output: Visualisasi Distribusi Data: 

<img width="540" height="393" alt="download" src="https://github.com/user-attachments/assets/c44d66d9-a07d-494c-9f3a-62edd20e7924" />

Output: 

Sample Data Glioma:<img width="1444" height="299" alt="download (6)" src="https://github.com/user-attachments/assets/e2f743d6-bc9f-4cfa-8427-c82e4b0dd617" />

Sample Data Notumor:<img width="1403" height="299" alt="download (5)" src="https://github.com/user-attachments/assets/fcaecbdc-f207-4e94-86f1-026eb546a87a" />

## Preprocessing
Pada percobaan ini digunakan 
``` python
def prepro1():
    pass

def prepro2():
    pass

def prepro3():
    pass
```
## Feature Extraction
Pada tahapan ini, Anda diminta untuk melakukan ekstraksi fitur dengan metode Gray Level Co-occurrence Matrix (GLCM). Dengan GLCM sudut 0, 45, 90, dan 135 derajat, simetris, dan lakukan uji coba dengan distance 1-5. Anda dapat menghitung nilai dari beberapa fitur berikut:

- Contrast
- Dissimilarity
- Homogeneity
- Energy
- Correlation
- Entropy
- ASM
``` python
def glcm(image, derajat):
 ...........
```

## Feature Selection
Pada tahap ini, Anda diminta untuk melakukan seleksi fitur. Anda dapat menggunakan teknik seleksi fitur seperti correlation, PCA, atau teknik seleksi fitur lain yang Anda ketahui. NAH PADA TEMPLATE CODE FEATURE SELECTION MENGGUNAKAN CORRELATION YGY.
``` python
correlation = hasilEkstrak.drop(columns=['Label','Filename']).corr()
......
```

## Splitting Data
Pada tahap ini, Anda diminta untuk membagi data menjadi data training dan data testing. Anda dapat menggunakan perbandingan 80:20 atau 70:30 atau 90:10.
``` python
Dataset, Dataset, Dataset, Dataset = train_test_split(Dataset, y, test_size=0.2, random_state=42)
print(Dataset.shape)
print(Dataset.shape)
```

## Normalization
Pada tahap ini, Anda diminta untuk melakukan normalisasi data. Anda dapat menggunakan teknik normalisasi standarization atau min-max normalization.
``` python
Dataset = (Dataset - Dataset.mean()) / Dataset.std()
Dataset = (Dataset - Dataset.mean()) / Dataset.std()
```

# Modeling
Pada tahap ini, Anda diminta untuk membuat model klasifikasi. Berikut merupakan model yang dapat digunakan:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest

Gunakan akurasi sebagai metrik dalam menampilkan hasil klasifikasi.
```python
# Train Random Forest Classifier
rf.fit(dataset, dataset)
# Train SVM Classifier
svm.fit(dataset, dataset)
# Train KNN Classifier
knn.fit(dataset, dataset)

def inidiaClassificationReport(dataset, dataset):
	print(classification_report(dataset, dataset))

```
Output: Contoh Classsification Report
|               | Accuracy | Precision | Recall   | F1-Score |
| ------------- | -------- | --------- | -------- | -------- |
| KNN           | 0.948667 | 0.948664  | 0.948667 | 0.948504 |
| SVM           | 0.976333 | 0.976319  | 0.976333 | 0.976333 |
| Random Forest | 0.959667 | 0.959822  | 0.959667 | 0.959615 |

# Evaluation
Pada bagian ini, Anda perlu mengevaluasi model klasifikasi yang telah Anda buat dengan menampilkan Confusion Matrix, dan Clasification Report: Accuracy, Precision, Recall, F1 Score. **Jelaskan hasil evaluasi yang Anda dapatkan dan berikan analisis mengenai hasil evaluasi tersebut**.

``` python

def plot_confusion_matrix(dataset, dataset, title):
  print(confusion_matrix)
```

Output: Contoh Confusion Matrix
![image](https://github.com/user-attachments/assets/aec4ac9c-e687-4354-b02d-833caf26db6b)
