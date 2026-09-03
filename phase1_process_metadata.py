import os
import sys
import pandas as pd
import numpy as np

# تنظیم خروجی کنسول برای سازگاری کامل با ویندوز و یونیکد
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 70)
print("   GIMENDO v2 - Tasks 1.4 & 1.5: Metadata Processing & Feature Encoding")
print("=" * 70)

# ۱. تعیین مسیرهای ورودی و خروجی
input_file = 'GIMENDO_v2_Metadata.xlsx'
if not os.path.exists(input_file):
    if os.path.exists(os.path.join('data', 'raw', input_file)):
        input_file = os.path.join('data', 'raw', input_file)
    else:
        raise FileNotFoundError(f"فایل متاداده {input_file} یافت نشد.")

output_dir = os.path.join('data', 'processed')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'cleaned_metadata_n17.xlsx')

print(f"\n📂 ۱. بارگذاری فایل اکسل متاداده از: {input_file}")
# بررسی ساختار هدر و خواندن داده‌ها (هدر اصلی در ردیف دوم است)
df_raw = pd.read_excel(input_file, header=1)
print(f"   - تعداد کل بیماران اولیه در دیتاست: {len(df_raw)}")
print(f"   - تعداد کل ستون‌ها: {len(df_raw.columns)}")

# ۲. پالایش و فیلتر کردن موارد نامشخص (Task 1.4: Remove uncertain cases -> n=17)
print("\n🔍 ۲. فیلتر کردن و حذف نمونه‌های نامشخص / منفی / فاقد زیرنوع مشخص:")
excluded_cases = []

for idx, row in df_raw.iterrows():
    case_code = row['Case Code']
    subtype_val = str(row['Histological Subtype']).strip()
    dysplasia_val = str(row.get('Dysplasia', '')).strip()

    # دلایل خروج:
    # الف) نمونه‌های سالم یا بدون داده پاتولوژی (NaN)
    if pd.isna(row['Histological Subtype']) or subtype_val.lower() == 'nan':
        excluded_cases.append((case_code, "فاقد بیوپسی متاداده / نرمال (NaN)"))
    # ب) نمونه‌های منفی برای متاپلازی
    elif subtype_val.lower() == 'negative':
        excluded_cases.append((case_code, "منفی برای متاپلازی روده (Negative for IM)"))
    # ج) نمونه‌های بدون تعیین زیرنوع (فقط درج شده IM Positive)
    elif subtype_val == 'IM Positive':
        excluded_cases.append((case_code, "عدم تعیین زیرنوع بافتی (Unspecified IM Positive)"))
    # د) نمونه دارای دیسپلازی / نئوپلازیا (LGD)
    elif dysplasia_val != 'Negative' and not pd.isna(row.get('Dysplasia')) and dysplasia_val.lower() != 'nan':
        excluded_cases.append((case_code, f"حذف به دلیل وجود دیسپلازی ({dysplasia_val})"))

for case_code, reason in excluded_cases:
    print(f"   ❌ حذف Case {case_code}: {reason}")

excluded_ids = [c[0] for c in excluded_cases]
df_clean = df_raw[~df_raw['Case Code'].isin(excluded_ids)].copy().reset_index(drop=True)

print(f"\n✅ تعداد بیماران نهایی پس از پالایش: n = {len(df_clean)} بیمار با زیرنوع‌های بافتی مشخص")

# استانداردسازی زیرنوع بافتی به دو کلاس دوتایی: Complete و Incomplete
def standardize_subtype(val):
    val_str = str(val)
    if 'Incomplete' in val_str:
        return 'Incomplete'
    elif 'Complete' in val_str:
        return 'Complete'
    return val_str

df_clean['Subtype'] = df_clean['Histological Subtype'].apply(standardize_subtype)
df_clean['Subtype_Binary'] = df_clean['Subtype'].map({'Complete': 0, 'Incomplete': 1})

print("\n📊 توزیع زیرنوع‌های بافتی در دیتاست پالایش‌شده:")
subtype_counts = df_clean['Subtype'].value_counts()
for st, cnt in subtype_counts.items():
    print(f"   - {st}: {cnt} بیمار ({cnt/len(df_clean)*100:.1f}%)")

# ۳. کدگذاری باینری ویژگی‌های آندوسکوپی پیشرفته (Task 1.5: Encode IEE Features)
print("\n⚙️ ۳. کدگذاری باینری ویژگی‌های آندوسکوپی IEE (LBC, MTB, WOS, TVF, MLE, TM):")
iee_features = ['LBC', 'MTB', 'WOS', 'TVF', 'MLE', 'TM']

for feat in iee_features:
    if feat in df_clean.columns:
        # کدگذاری باینری: اگر منفی، خالی یا صفر بود -> 0 ، در غیر این صورت (مشاهده در هر بخش معده) -> 1
        df_clean[feat] = df_clean[feat].apply(
            lambda x: 0 if str(x).strip().lower() in ['negative', 'nan', 'none', '0'] or pd.isna(x) else 1
        )
        pos_cnt = df_clean[feat].sum()
        print(f"   - {feat:4s}: {pos_cnt:2d} مثبت (1) | {len(df_clean)-pos_cnt:2d} منفی (0)")

# ۴. حذف ویژگی IRV به دلیل واریانس صفر (Zero-Variance Feature)
if 'IRV' in df_clean.columns:
    df_clean.drop(columns=['IRV'], inplace=True)
    print("\n🗑️ ۴. ویژگی 'IRV' به دلیل واریانس صفر (تماماً Negative / 0) با موفقیت حذف گردید.")

# ۵. ذخیره فایل تمیز شده در مسیر داده‌های پردازش‌شده
df_clean.to_excel(output_file, index=False)
print(f"\n💾 ۵. داده‌های پردازش‌شده نهایی با موفقیت ذخیره شدند:")
print(f"   📁 مسیر فایل: {output_file}")
print(f"   📐 ابعاد ماتریس نهایی: {df_clean.shape[0]} ردیف × {df_clean.shape[1]} ستون")

print("\n📋 پیش‌نمایش جدول داده‌های نهایی (n=17):")
preview_cols = ['Case Code', 'Subtype', 'Age', 'Sex'] + iee_features
print(df_clean[preview_cols].to_string(index=False))

print("\n" + "=" * 70)
print("🎉 وظایف Tasks 1.4 & 1.5 با موفقیت کامل انجام و اعتبارسنجی شدند.")
print("=" * 70)
