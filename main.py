
import cv2
import numpy as np
import pandas as pd
import pytesseract
import re
from pathlib import Path

# =====================================================
# KONFIGURASI
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
OUTPUT_DIR = BASE_DIR / "output"

CROP_DIR = OUTPUT_DIR / "crop"
ENHANCEMENT_DIR = OUTPUT_DIR / "enhancement"
SIGNATURE_DIR = OUTPUT_DIR / "signature"

for folder in (CROP_DIR, ENHANCEMENT_DIR, SIGNATURE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

EXPECTED_NUMBER = "571012022000056"

IMAGE_FILES = [
    "01_HighQuality_Enhanced.jpg",
    "02_LowContrast.jpg",
    "03_Blurred.jpg",
    "04_HighNoise.jpg",
    "05_LowResolution_Upsampled.jpg",
    "06_Faded_Underexposed.jpg",
    "07_ColorShift_WarmTint.jpg",
    "08_JPEGCompression_Artifacts.jpg",
    "09_CombinedDegradation.jpg",
]


# =====================================================
# MENUTUP JENDELA DENGAN AMAN
# =====================================================
def close_windows():
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except cv2.error:
        pass


def show_image(title, image):
    if image is None or image.size == 0:
        return

    h, w = image.shape[:2]
    scale = min(1000 / w, 700 / h, 1.0)

    if scale < 1:
        preview = cv2.resize(
            image,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA
        )
    else:
        preview = image.copy()

    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, preview)
        print(f"{title}: tekan tombol pada jendela untuk melanjutkan.")
        cv2.waitKey(0)
    except cv2.error as error:
        print("Gagal menampilkan gambar:", error)
    finally:
        close_windows()


# =====================================================
# CROP MANUAL
# =====================================================
def manual_crop(image, title):
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    scale = min(1000 / w, 700 / h, 1.0)

    preview = cv2.resize(
        image,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )

    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            title, preview.shape[1], preview.shape[0]
        )

        print("\nPILIH AREA:", title)
        print("Tarik kotak menggunakan mouse kiri.")
        print("Tekan ENTER atau SPACE untuk menyimpan crop.")
        print("Tekan C untuk membatalkan.")

        x, y, rw, rh = cv2.selectROI(
            title,
            preview,
            showCrosshair=True,
            fromCenter=False
        )

    except cv2.error as error:
        print("Gagal membuka jendela crop:", error)
        return None

    finally:
        close_windows()

    if rw <= 0 or rh <= 0:
        print("Crop dibatalkan.")
        return None

    x1 = max(0, int(round(x / scale)))
    y1 = max(0, int(round(y / scale)))
    x2 = min(w, int(round((x + rw) / scale)))
    y2 = min(h, int(round((y + rh) / scale)))

    crop = image[y1:y2, x1:x2].copy()

    if crop.size == 0:
        print("Crop kosong.")
        return None

    # Otomatis ubah orientasi vertikal menjadi horizontal
    crop_h, crop_w = crop.shape[:2]

    if crop_h > crop_w:
        crop = cv2.rotate(
            crop,
            cv2.ROTATE_90_CLOCKWISE
        )
        print("Orientasi crop diubah menjadi horizontal.")

    print(
        f"Ukuran crop akhir: "
        f"{crop.shape[1]} x {crop.shape[0]}"
    )

    # Periksa hasil crop
    show_image("Periksa hasil crop", crop)

    return crop


def save_image(path, image):
    if image is not None and image.size > 0:
        cv2.imwrite(str(path), image)


def to_gray(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


# =====================================================
# IMAGE ENHANCEMENT
# =====================================================
def enhancement_methods(gray):
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    equalized = cv2.equalizeHist(gray)

    low, high = np.percentile(gray, (1, 99))
    if high > low:
        stretched = np.clip(
            (gray.astype(np.float32) - low)
            * 255.0 / (high - low),
            0, 255
        ).astype(np.uint8)
    else:
        stretched = gray.copy()

    return {
        "Grayscale": gray,
        "CLAHE": clahe,
        "Histogram_Equalization": equalized,
        "Contrast_Stretching": stretched
    }


# =====================================================
# OCR DAN CER
# =====================================================
def normalize_number(text):
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def calculate_cer(reference, prediction):
    ref = normalize_number(reference)
    pred = normalize_number(prediction)

    if not ref:
        return 1.0

    dp = [
        [0] * (len(pred) + 1)
        for _ in range(len(ref) + 1)
    ]

    for i in range(len(ref) + 1):
        dp[i][0] = i

    for j in range(len(pred) + 1):
        dp[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(pred) + 1):
            cost = 0 if ref[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1] / len(ref)


def run_ocr(image):
    enlarged = cv2.resize(
        image,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    blurred = cv2.GaussianBlur(enlarged, (3, 3), 0)

    _, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    config = (
        "--oem 3 --psm 7 "
        "-c tessedit_char_whitelist=0123456789"
    )

    text = pytesseract.image_to_string(
        binary, config=config
    )
    prediction = re.sub(r"[^0-9]", "", text)

    return prediction, binary


def process_number(crop, image_name, reference):
    gray = to_gray(crop)
    methods = enhancement_methods(gray)
    results = []

    for method, enhanced in methods.items():
        save_image(
            ENHANCEMENT_DIR / f"{image_name}_{method}.jpg",
            enhanced
        )

        prediction, binary = run_ocr(enhanced)

        save_image(
            ENHANCEMENT_DIR /
            f"{image_name}_{method}_ocr_input.jpg",
            binary
        )

        cer = calculate_cer(reference, prediction)

        results.append({
            "gambar": image_name,
            "metode_enhancement": method,
            "nomor_acuan": normalize_number(reference),
            "hasil_ocr": prediction,
            "CER": cer,
            "akurasi_karakter_persen": max(0, 1 - cer) * 100
        })

        print(
            f"{method:24} OCR={prediction or '(kosong)':18} "
            f"CER={cer:.4f}"
        )

    return results


# =====================================================
# DETEKSI TANDA TANGAN
# =====================================================
def detect_signature(crop, image_name):
    gray = to_gray(crop)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold global
    _, global_binary = cv2.threshold(
        blurred, 127, 255, cv2.THRESH_BINARY_INV
    )

    # Threshold Otsu
    _, otsu_binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2, 2)
    )

    opened = cv2.morphologyEx(
        otsu_binary, cv2.MORPH_OPEN, kernel
    )
    closed = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, kernel
    )

    foreground = cv2.countNonZero(closed)
    total = closed.size
    ratio = foreground / total if total else 0

    # Ambang baseline, perlu dievaluasi menggunakan label aktual.
    status = "PRESENT" if ratio >= 0.008 else "ABSENT"

    save_image(SIGNATURE_DIR / f"{image_name}_gray.jpg", gray)
    save_image(
        SIGNATURE_DIR / f"{image_name}_global.jpg",
        global_binary
    )
    save_image(
        SIGNATURE_DIR / f"{image_name}_otsu.jpg",
        otsu_binary
    )
    save_image(
        SIGNATURE_DIR / f"{image_name}_morphology.jpg",
        closed
    )

    print("\nHASIL DETEKSI TANDA TANGAN")
    print("Piksel foreground:", foreground)
    print("Rasio foreground:", f"{ratio:.4%}")
    print("Prediksi: SIGNATURE", status)

    return {
        "hasil_deteksi": status,
        "rasio_foreground": ratio,
        "piksel_global": cv2.countNonZero(global_binary),
        "piksel_otsu": cv2.countNonZero(otsu_binary)
    }


# =====================================================
# PROGRAM UTAMA
# =====================================================
def main():
    if not DATASET_DIR.exists():
        print("Folder Dataset tidak ditemukan:", DATASET_DIR)
        return

    try:
        print("Tesseract:", pytesseract.get_tesseract_version())
    except Exception as error:
        print("Tesseract tidak bisa dijalankan.")
        print("Periksa lokasi instalasi Tesseract.")
        print(error)
        return

    ocr_records = []
    signature_records = []

    for filename in IMAGE_FILES:
        path = DATASET_DIR / filename

        if not path.exists():
            print("\nFile tidak ditemukan:", filename)
            continue

        image = cv2.imread(str(path))
        if image is None:
            print("Gagal membaca:", filename)
            continue

        name = Path(filename).stem
        print("\n" + "=" * 60)
        print("Memproses:", filename)

        # AREA NOMOR IJAZAH
        number_crop = manual_crop(
            image, f"{name} - Crop Nomor Ijazah"
        )

        if number_crop is None:
            print("Gambar dilewati karena crop nomor dibatalkan.")
            continue

        save_image(CROP_DIR / f"{name}_nomor.jpg", number_crop)

        reference = input(
            f"Nomor acuan yang benar [{EXPECTED_NUMBER}]: "
        ).strip()

        if not reference:
            reference = EXPECTED_NUMBER

        print("\nPerbandingan enhancement:")
        ocr_records.extend(
            process_number(number_crop, name, reference)
        )

        # AREA TANDA TANGAN
        signature_crop = manual_crop(
            image, f"{name} - Crop Tanda Tangan"
        )

        if signature_crop is None:
            print("Crop tanda tangan dibatalkan.")
            continue

        save_image(
            CROP_DIR / f"{name}_tanda_tangan.jpg",
            signature_crop
        )

        result = detect_signature(signature_crop, name)

        actual = input(
            "Label aktual tanda tangan (PRESENT/ABSENT): "
        ).strip().upper()

        if actual not in ("PRESENT", "ABSENT"):
            actual = "UNKNOWN"

        result["gambar"] = filename
        result["label_aktual"] = actual
        signature_records.append(result)

    # Simpan hasil OCR
    if ocr_records:
        df = pd.DataFrame(ocr_records)
        df.to_csv(
            OUTPUT_DIR / "hasil_ocr.csv",
            index=False,
            encoding="utf-8-sig"
        )

        best = (
            df.sort_values("CER")
            .groupby("gambar", as_index=False)
            .first()
        )

        average_cer = best["CER"].mean()
        average_accuracy = max(0, 1 - average_cer) * 100

        with open(
            OUTPUT_DIR / "laporan_ocr.txt",
            "w",
            encoding="utf-8"
        ) as file:
            file.write("LAPORAN OCR NOMOR IJAZAH\n")
            file.write("=" * 40 + "\n\n")
            file.write("Hasil terbaik per gambar berdasarkan CER:\n")
            file.write(best.to_string(index=False))
            file.write("\n\n")
            file.write(f"CER rata-rata: {average_cer:.4f}\n")
            file.write(
                f"Akurasi karakter rata-rata: "
                f"{average_accuracy:.2f}%\n"
            )

        print("\nCSV OCR:", OUTPUT_DIR / "hasil_ocr.csv")
        print("Laporan:", OUTPUT_DIR / "laporan_ocr.txt")

    # Simpan hasil tanda tangan
    if signature_records:
        pd.DataFrame(signature_records).to_csv(
            OUTPUT_DIR / "hasil_tanda_tangan.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print(
            "CSV tanda tangan:",
            OUTPUT_DIR / "hasil_tanda_tangan.csv"
        )

    print("\nSELESAI. Periksa folder output.")


if __name__ == "__main__":
    main()
