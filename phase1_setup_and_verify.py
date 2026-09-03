import os
import pandas as pd

# ۱. تعریف ساختار پوشه‌های استاندارد پروژه
folders = [
    'data/raw',                # پوشه قرارگیری فایل‌های اصلی v2
    'data/extracted_frames',   # پوشه ذخیره فریم‌های استخراج‌شده از ویدیوها
    'data/filtered_frames',    # پوشه ذخیره فریم‌های شفاف (بعد از حذف تارها)
    'data/processed',          # داده‌های تمیزشده و انکودشده اکسل
    'models',                  # ذخیره وزن‌های مدل
    'outputs/plots',           # نمودارهای SHAP و ROC
    'outputs/tables'           # جداول خروجی نتایج
]

print("=== ۱. ساخت ساختار پوشه‌های پروژه ===")
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ پوشه ساخته شد یا از قبل وجود داشت: {folder}")

print("\n=== ۲. بررسی سلامت و ساختار دیتاست ===")
# جستجو برای پیدا کردن فایل اکسل متاداده
excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
if not excel_files:
    # بررسی داخل پوشه raw اگر فایل اکسل آنجا باشد
    if os.path.exists('data/raw'):
        excel_files = [os.path.join('data/raw', f) for f in os.listdir('data/raw') if f.endswith('.xlsx')]

if excel_files:
    metadata_path = excel_files[0]
    print(f"📄 فایل متاداده پیدا شد: {metadata_path}")
    df = pd.read_excel(metadata_path)
    print(f"📊 تعداد کل ردیف‌ها (بیماران): {len(df)}")
    print("📋 ستون‌های موجود در فایل اکسل:")
    for col in df.columns:
        print(f"  - {col}")
else:
    print("⚠️ فایل اکسل (.xlsx) در این مسیر پیدا نشد. لطفاً فایل اکسل متاداده را در پوشه اصلی قرار دهید.")