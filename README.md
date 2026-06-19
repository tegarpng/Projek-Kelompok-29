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
Pada percobaan ini digunakan preprocessing median, grayscale, resize, sobel, roberts, clahe, dan treshold. Preprocessing ini digunakan pada masing masing percobaan tergantung ketentuan tiap percobaan.
``` python

def median(image, rowkernel, columnkernel):
    row, column = int(rowkernel/2), int(columnkernel/2)
    image_pad = np.pad(image, [(row, row), (column, column)], mode="edge")
    result = np.zeros(image.shape)

    for i in range(row, image.shape[0] + row):
        for j in range(column, image.shape[1] + column):
            submatrix = image_pad[i-row:i+row+1, j-column:j+column+1]
            result[i-row, j-column] = np.median(submatrix)

    return result.astype(np.uint8)

def grayscale(image):
    if len(image.shape) == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image

def resize(image, target_size):
    return cv.resize(image, target_size)
	
def sobel(image):
    sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = (magnitude / magnitude.max()) * 255 if magnitude.max() != 0 else magnitude
    return magnitude.astype(np.uint8)

def roberts(image):
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)

    img_f = image.astype(np.float64)
    gx = cv.filter2D(img_f, -1, kernel_x)
    gy = cv.filter2D(img_f, -1, kernel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = (magnitude / magnitude.max()) * 255 if magnitude.max() != 0 else magnitude
    return magnitude.astype(np.uint8)

def clahe_manual(citra, block_size=32, clip_limit=2.5):
    height, width = citra.shape
    hasil = np.zeros_like(citra, dtype=np.uint8)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            blok = citra[i:i+block_size, j:j+block_size]
            h_blok, w_blok = blok.shape

            hist = np.zeros(256, dtype=int)
            for bi in range(h_blok):
                for bj in range(w_blok):
                    hist[blok[bi][bj]] += 1

            total_piksel = h_blok * w_blok
            limit = int(clip_limit * total_piksel / 256)
            if limit < 1:
                limit = 1

            kelebihan = 0
            for c in range(256):
                if hist[c] > limit:
                    kelebihan += hist[c] - limit
                    hist[c] = limit

            distribusi_rata = kelebihan // 256
            sisa = kelebihan % 256

            for c in range(256):
                hist[c] += distribusi_rata
            for c in range(sisa):
                hist[c] += 1

            cdf = np.zeros(256, dtype=int)
            cdf[0] = hist[0]
            for c in range(1, 256):
                cdf[c] = cdf[c-1] + hist[c]

            cdf_min = cdf[cdf > 0][0] if len(cdf[cdf > 0]) > 0 else 0
            cdf_normal = np.zeros(256, dtype=int)

            pembagi = total_piksel - cdf_min
            for c in range(256):
                if pembagi > 0:
                    val = np.round((cdf[c] - cdf_min) * 255 / pembagi)
                    cdf_normal[c] = int(max(0, min(255, val)))
                else:
                    cdf_normal[c] = 0

            for bi in range(h_blok):
                for bj in range(w_blok):
                    hasil[i+bi][j+bj] = cdf_normal[blok[bi][bj]]

    return hasil.astype(np.uint8)

def thresholding_manual(image, thresh_value=125):
    img_thresh = np.zeros_like(image)
    img_thresh[image > thresh_value] = 255
    return img_thresh
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
Berikut merupakan model yang digunakan:
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
