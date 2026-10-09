# mini_project_ocr
# How to Run

## 1. Persiapan

Sebelum menjalankan program, pastikan Python dan Tesseract OCR sudah terpasang di komputer. Kedua perangkat lunak ini diperlukan untuk menjalankan proses pengolahan citra dan membaca nomor ijazah secara otomatis.

## 2. Instal Library

Buka folder proyek di Visual Studio Code, kemudian buka terminal. Jalankan perintah `pip install -r requirements.txt` untuk menginstal library yang dibutuhkan oleh program.

## 3. Menyiapkan Dataset

Masukkan gambar ijazah yang akan diuji ke dalam folder dataset. Gambar yang digunakan terdiri dari sembilan citra dengan kondisi yang berbeda-beda, mulai dari citra berkualitas baik hingga citra yang buram, memiliki noise, atau kontras rendah.

## 4. Menjalankan Program

Setelah semua persiapan selesai, jalankan program dengan mengetikkan `python main.py` pada terminal. Program kemudian akan memulai proses pengolahan citra sesuai dengan tahapan yang telah dibuat.

## 5. Memilih Bagian Ijazah

Saat program dijalankan, pilih bagian nomor ijazah yang ingin dibaca. Setelah itu, pilih bagian tanda tangan kepala sekolah yang akan diperiksa. Pemilihan area dilakukan secara manual agar bagian yang diproses sesuai dengan kebutuhan.

## 6. Melihat Hasil

Setelah proses selesai, hasilnya dapat diperiksa untuk mengetahui nomor ijazah yang berhasil dibaca dan apakah tanda tangan terdeteksi atau tidak. Hasil pengujian OCR juga dibandingkan menggunakan nilai Character Error Rate (CER) untuk mengetahui tingkat kesalahan pembacaan karakter.

