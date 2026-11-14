"""
Dashboard kullanımı (src/services.py üzerinden):
    cfg = GeneratorConfig(n_customers=20_000, churn_target_rate=0.30)
    df, report = generate_and_store(cfg, progress=lambda p, msg: bar.progress(p, text=msg))

Hedef değişkenler:
    loyalty_score   : 0-100, yüzde (regresyon)
    loyalty_segment : skordan türetilir (modele GİRMEMELİ, sızıntı olur)
    churn           : REF_DATE'ten sonraki 90 günde hiç alışveriş yok (1) / var (0)
Sezon çıktıları     : season_prob_<ürün>, top_season_product, expected_season_orders
                      (sistem tarihine göre: önümüzdeki 90 günün mevsimlerine bağlı)
"""
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


class DataValidationError(Exception):
    """Üretilen veri tutarlılık kontrollerinden geçemedi."""


@dataclass
class GeneratorConfig:
    """Dashboard'daki form/slider değerleri buraya gelir."""
    n_customers: int = 10_000
    seed: int = 42
    ref_date: str | None = None         # None = bugünün tarihi (sistem saati). Sezon buna göre belirlenir
    missing_rate: float = 0.02          # seçili sütunlarda eksik değer oranı
    churn_target_rate: float = 0.25     # hedeflenen churn oranı
    churn_decay: float = 0.10           # kendi ritminden geç kalmanın alışveriş hızını düşürme şiddeti
    churn_shock_sigma: float = 0.35     # gözlenmeyen şokların şiddeti (0 = şok yok)

    def resolved_ref_date(self) -> pd.Timestamp:
        """ref_date boşsa bilgisayarın bugünkü tarihi."""
        return pd.Timestamp(self.ref_date) if self.ref_date else pd.Timestamp.today().normalize()

    def validate(self) -> None:
        if not 1_000 <= self.n_customers <= 200_000:
            raise ValueError("Müşteri sayısı 1.000 ile 200.000 arasında olmalı.")
        if not 0.05 <= self.churn_target_rate <= 0.60:
            raise ValueError("Hedef churn oranı %5 ile %60 arasında olmalı.")
        if not 0.0 <= self.churn_decay <= 1.0:
            raise ValueError("Kopma etkisi 0 ile 1 arasında olmalı.")
        if not 0.0 <= self.churn_shock_sigma <= 1.0:
            raise ValueError("Şok şiddeti 0 ile 1 arasında olmalı.")
        if not 0.0 <= self.missing_rate <= 0.20:
            raise ValueError("Eksik değer oranı %0 ile %20 arasında olmalı.")
        try:
            self.resolved_ref_date()
        except Exception:
            raise ValueError("Referans tarih geçerli değil (YYYY-AA-GG).")
        if self.seed < 0:
            raise ValueError("Seed negatif olamaz.")


# ----------------------------------------------------------------------
# 81 il: (nüfus [bin, yaklaşık], kışın soğukluk indeksi 0-1, bölge)
# ----------------------------------------------------------------------
PROVINCES = {
    # Marmara
    "İstanbul": (15655, 0.40, "Marmara"), "Bursa": (3214, 0.45, "Marmara"),
    "Kocaeli": (2102, 0.42, "Marmara"), "Tekirdağ": (1167, 0.42, "Marmara"),
    "Sakarya": (1088, 0.42, "Marmara"), "Balıkesir": (1273, 0.45, "Marmara"),
    "Çanakkale": (570, 0.35, "Marmara"), "Edirne": (413, 0.50, "Marmara"),
    "Kırklareli": (365, 0.50, "Marmara"), "Bilecik": (228, 0.50, "Marmara"),
    "Yalova": (297, 0.38, "Marmara"),
    # Ege
    "İzmir": (4479, 0.20, "Ege"), "Manisa": (1480, 0.25, "Ege"),
    "Aydın": (1161, 0.15, "Ege"), "Denizli": (1054, 0.30, "Ege"),
    "Muğla": (1049, 0.15, "Ege"), "Afyonkarahisar": (745, 0.70, "Ege"),
    "Kütahya": (580, 0.70, "Ege"), "Uşak": (376, 0.55, "Ege"),
    # Akdeniz
    "Antalya": (2688, 0.10, "Akdeniz"), "Adana": (2263, 0.12, "Akdeniz"),
    "Mersin": (1938, 0.08, "Akdeniz"), "Hatay": (1544, 0.15, "Akdeniz"),
    "Kahramanmaraş": (1177, 0.40, "Akdeniz"), "Osmaniye": (559, 0.20, "Akdeniz"),
    "Isparta": (441, 0.50, "Akdeniz"), "Burdur": (276, 0.50, "Akdeniz"),
    # İç Anadolu
    "Ankara": (5803, 0.65, "İç Anadolu"), "Konya": (2296, 0.65, "İç Anadolu"),
    "Kayseri": (1445, 0.75, "İç Anadolu"), "Eskişehir": (914, 0.65, "İç Anadolu"),
    "Sivas": (636, 0.85, "İç Anadolu"), "Aksaray": (429, 0.70, "İç Anadolu"),
    "Yozgat": (418, 0.75, "İç Anadolu"), "Niğde": (365, 0.70, "İç Anadolu"),
    "Nevşehir": (306, 0.65, "İç Anadolu"), "Kırıkkale": (278, 0.65, "İç Anadolu"),
    "Karaman": (257, 0.60, "İç Anadolu"), "Kırşehir": (245, 0.65, "İç Anadolu"),
    "Çankırı": (195, 0.70, "İç Anadolu"),
    # Karadeniz
    "Samsun": (1368, 0.42, "Karadeniz"), "Trabzon": (818, 0.42, "Karadeniz"),
    "Ordu": (761, 0.42, "Karadeniz"), "Tokat": (603, 0.60, "Karadeniz"),
    "Zonguldak": (590, 0.45, "Karadeniz"), "Çorum": (523, 0.65, "Karadeniz"),
    "Giresun": (453, 0.45, "Karadeniz"), "Düzce": (405, 0.45, "Karadeniz"),
    "Kastamonu": (379, 0.65, "Karadeniz"), "Rize": (344, 0.40, "Karadeniz"),
    "Amasya": (338, 0.50, "Karadeniz"), "Bolu": (316, 0.65, "Karadeniz"),
    "Karabük": (252, 0.55, "Karadeniz"), "Sinop": (220, 0.40, "Karadeniz"),
    "Bartın": (201, 0.45, "Karadeniz"), "Artvin": (169, 0.60, "Karadeniz"),
    "Gümüşhane": (143, 0.75, "Karadeniz"), "Bayburt": (87, 0.85, "Karadeniz"),
    # Doğu Anadolu
    "Van": (1128, 0.85, "Doğu Anadolu"), "Malatya": (812, 0.60, "Doğu Anadolu"),
    "Erzurum": (749, 0.95, "Doğu Anadolu"), "Elazığ": (591, 0.75, "Doğu Anadolu"),
    "Ağrı": (510, 0.95, "Doğu Anadolu"), "Muş": (395, 0.85, "Doğu Anadolu"),
    "Bitlis": (353, 0.85, "Doğu Anadolu"), "Bingöl": (282, 0.80, "Doğu Anadolu"),
    "Kars": (280, 1.00, "Doğu Anadolu"), "Hakkari": (280, 0.90, "Doğu Anadolu"),
    "Erzincan": (238, 0.80, "Doğu Anadolu"), "Iğdır": (205, 0.70, "Doğu Anadolu"),
    "Ardahan": (92, 1.00, "Doğu Anadolu"), "Tunceli": (83, 0.80, "Doğu Anadolu"),
    # Güneydoğu Anadolu
    "Gaziantep": (2154, 0.40, "Güneydoğu Anadolu"), "Şanlıurfa": (2200, 0.30, "Güneydoğu Anadolu"),
    "Diyarbakır": (1804, 0.45, "Güneydoğu Anadolu"), "Mardin": (872, 0.40, "Güneydoğu Anadolu"),
    "Adıyaman": (635, 0.40, "Güneydoğu Anadolu"), "Batman": (631, 0.40, "Güneydoğu Anadolu"),
    "Şırnak": (557, 0.60, "Güneydoğu Anadolu"), "Siirt": (331, 0.50, "Güneydoğu Anadolu"),
    "Kilis": (147, 0.35, "Güneydoğu Anadolu"),
}
assert len(PROVINCES) == 81, f"81 il olmalı, {len(PROVINCES)} var"

CITIES = list(PROVINCES.keys())
CITY_POP = np.array([v[0] for v in PROVINCES.values()], dtype=float)
CITY_P = CITY_POP / CITY_POP.sum()
CITY_COLD = np.array([v[1] for v in PROVINCES.values()])
CITY_REGION = np.array([v[2] for v in PROVINCES.values()])

PAYMENTS = ["kredi_karti", "kapida_odeme", "havale", "dijital_cuzdan"]
PAYMENT_P = [0.50, 0.15, 0.10, 0.25]

# ----------------------------------------------------------------------
# Ürün kataloğu: (ad, temel popülerlik, ortalama fiyat TL, kadın ilgisi, erkek ilgisi,
#                 mevsim çarpanları: (kış, ilkbahar, yaz, sonbahar))
# Mevsim çarpanı > 1: o mevsimde daha çok satar, < 1: daha az satar
# ----------------------------------------------------------------------
PRODUCT_TABLE = [
    ("tisort",             1.0,  250, 1.0, 1.0, (0.45, 1.10, 1.80, 0.80)),
    ("gomlek_bluz",        0.7,  400, 1.2, 1.0, (0.90, 1.20, 0.90, 1.10)),
    ("kazak",              0.7,  550, 1.0, 1.0, (2.60, 1.00, 0.15, 1.80)),
    ("hirka",              0.4,  500, 1.3, 0.6, (2.00, 1.40, 0.20, 2.00)),
    ("sweatshirt",         0.8,  450, 1.0, 1.0, (1.80, 1.20, 0.30, 1.60)),
    ("pantolon",           0.9,  600, 1.0, 1.0, (1.10, 1.10, 0.60, 1.20)),
    ("jean",               0.9,  650, 1.0, 1.0, (1.00, 1.00, 0.80, 1.10)),
    ("elbise",             0.8,  700, 1.0, 0.0, (0.70, 1.40, 1.80, 0.80)),
    ("etek",               0.4,  400, 1.0, 0.0, (0.60, 1.30, 1.50, 0.70)),
    ("mont_kaban",         0.5, 1800, 1.0, 1.0, (3.50, 0.50, 0.05, 1.60)),
    ("ceket_blazer",       0.4, 1100, 0.8, 1.0, (1.20, 1.30, 0.60, 1.30)),
    ("sort_bermuda",       0.5,  300, 0.5, 1.3, (0.15, 1.00, 3.00, 0.40)),
    ("atki_bere_eldiven",  0.3,  200, 1.0, 1.0, (4.00, 0.40, 0.02, 1.50)),
    ("spor_giyim",         0.6,  500, 1.0, 1.0, (1.00, 1.30, 1.40, 1.00)),
    ("ic_giyim_corap",     0.6,  150, 1.0, 1.0, (1.30, 1.00, 0.80, 1.10)),
]
PRODUCTS = [p[0] for p in PRODUCT_TABLE]
P_BASE = np.array([p[1] for p in PRODUCT_TABLE])
P_PRICE = np.array([p[2] for p in PRODUCT_TABLE], dtype=float)
P_W_KADIN = np.array([p[3] for p in PRODUCT_TABLE])
P_W_ERKEK = np.array([p[4] for p in PRODUCT_TABLE])
P_SEASON = np.array([p[5] for p in PRODUCT_TABLE])      # (ürün, 4 mevsim)
P_WINTER = P_SEASON[:, 0]                                # yıllık ürün karışımında iklim etkisi için

SEASONS = ["kis", "ilkbahar", "yaz", "sonbahar"]
SEASON_LABELS = {"kis": "Kış", "ilkbahar": "İlkbahar", "yaz": "Yaz", "sonbahar": "Sonbahar"}
SEASON_ORDER_INTENSITY = np.array([1.10, 1.00, 0.90, 1.05])  # mevsime göre genel alışveriş yoğunluğu
_MONTH_TO_SEASON = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
HORIZON_DAYS = 90                      # hem churn etiketi hem sezon tahmini için ileriye bakış
HORIZON_MONTHS = HORIZON_DAYS / 30


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pct(x):
    """0-1 arası yüzdelik sıra."""
    return pd.Series(x).rank(pct=True).to_numpy()


def dirichlet_rows(rng, alpha):
    g = rng.gamma(alpha)
    return g / g.sum(axis=1, keepdims=True)


def lifetime_product_shares(rng, is_kadin, cold):
    """Müşterinin tüm zamanlardaki ürün payları (satır toplamı 1).
    Cinsiyete göre ürün ilgisi + soğuk şehirlerde kışlık ürünlere hafif kayma."""
    gender_w = np.where(is_kadin[:, None], P_W_KADIN[None, :], P_W_ERKEK[None, :])
    annual_mult = (0.5 * (1 + P_WINTER))[None, :] ** (0.5 * cold[:, None])
    w = P_BASE[None, :] * gender_w * annual_mult
    w = w / w.sum(axis=1, keepdims=True)
    alpha = (0.2 + 8 * w) * (w > 0)          # o cinsiyete uymayan ürün: 0 pay
    return dirichlet_rows(rng, alpha)


def season_info(ref_date=None, horizon_days: int = HORIZON_DAYS) -> dict:
    """Tarihe göre sezon bilgisi (dashboard başlığı için).
    ref_date boşsa bilgisayarın bugünkü tarihi. 'window', önümüzdeki 90 günün mevsim payları."""
    ref = pd.Timestamp(ref_date) if ref_date is not None else pd.Timestamp.today().normalize()
    days = pd.date_range(ref + pd.Timedelta(days=1), periods=horizon_days)
    frac = np.bincount([_MONTH_TO_SEASON[m] for m in days.month], minlength=4) / horizon_days
    current = SEASONS[_MONTH_TO_SEASON[ref.month]]
    dominant = SEASONS[int(frac.argmax())]
    return {
        "date": ref.strftime("%Y-%m-%d"),
        "current_season": current,
        "current_season_label": SEASON_LABELS[current],
        "dominant_season": dominant,
        "dominant_season_label": SEASON_LABELS[dominant],
        "window_days": horizon_days,
        "window_fractions": {SEASONS[i]: round(float(f), 3) for i, f in enumerate(frac) if f > 0},
        "fractions_array": frac,
    }


def season_purchase_probs(shares, cold, lam, trend, recency_ratio, ref_date,
                          horizon_days: int = HORIZON_DAYS):
    """Önümüzdeki 90 günde her ürünü en az 1 kez alma olasılığı (tarihe duyarlı).

    1) 90 günlük pencerenin hangi mevsimlere düştüğü hesaplanır (örn. %79 kış, %21 sonbahar).
    2) Her mevsim için ürün payı: q_i ~ s_i * mevsim_çarpanı_i ^ (0.5 + şiddet)
       şiddet: kışın şehrin soğukluğu, yazın sıcaklığı (1 - soğukluk), ara mevsimlerde 0.5.
       (soğuk şehirde mont daha çok, sıcak şehirde yazın tişört/şort daha çok)
    3) Pencere payı: q = sum_mevsim( pay_mevsim * q_mevsim ).
    4) Pencere sipariş sayısı N = ay * aylık hız * trend * mevsim yoğunluğu * exp(-0.3 * recency_ratio)
       (bağlılığı kopmakta olan müşteri az alışveriş yapar)
    5) Poisson varsayımı:  P_i = 1 - exp(-N * q_i)
    """
    frac = season_info(ref_date, horizon_days)["fractions_array"]
    q_mix = np.zeros_like(shares)
    intensity = float(frac @ SEASON_ORDER_INTENSITY)
    for s, f in enumerate(frac):
        if f == 0:
            continue
        severity = cold if s == 0 else (1 - cold if s == 2 else np.full_like(cold, 0.5))
        q = shares * P_SEASON[None, :, s] ** (0.5 + severity[:, None])
        q_mix += f * q / q.sum(axis=1, keepdims=True)
    n_window = (horizon_days / 30) * lam * trend * intensity * np.exp(-0.3 * recency_ratio)
    return 1 - np.exp(-n_window[:, None] * q_mix), n_window


def generate(cfg: GeneratorConfig | None = None, progress=None) -> pd.DataFrame:
    """Veri setini üretir ve doğrular. progress: callback(oran 0-1, mesaj)."""
    cfg = cfg or GeneratorConfig()
    cfg.validate()
    step = (lambda p, msg: progress(p, msg)) if progress else (lambda p, msg: None)

    n = cfg.n_customers
    ref_date = cfg.resolved_ref_date()
    rng = np.random.default_rng(cfg.seed)

    # 0) Gizli özellikler (veri setinde yer almaz)
    step(0.05, "Müşteri profilleri oluşturuluyor...")
    e = rng.normal(0, 1, n)            # latent bağlılık
    price_sens = rng.beta(2, 3, n)     # fiyat hassasiyeti

    # 1) Demografik
    age = np.clip(rng.normal(37, 11, n), 18, 75).round().astype(int)
    gender = rng.choice(["kadin", "erkek"], n, p=[0.55, 0.45])
    is_kadin = gender == "kadin"
    city_idx = rng.choice(len(CITIES), n, p=CITY_P)
    city = np.array(CITIES)[city_idx]
    region = CITY_REGION[city_idx]
    cold = CITY_COLD[city_idx]
    payment_method = rng.choice(PAYMENTS, n, p=PAYMENT_P)

    income_z = 0.25 * (age - 37) / 11 + rng.normal(0, 1, n)
    q30, q80 = np.quantile(income_z, [0.30, 0.80])
    income_level = np.where(income_z < q30, "dusuk", np.where(income_z < q80, "orta", "yuksek"))

    # 2) İlişki süresi
    tenure_months = np.clip(rng.gamma(2.5, 10, n) * np.exp(0.15 * e), 1, 120).round().astype(int)
    tenure_days = (tenure_months * 30.44 + rng.uniform(0, 30, n)).astype(int)

    # 3) Ürün payları
    step(0.15, "Ürün tercihleri hesaplanıyor...")
    shares = lifetime_product_shares(rng, is_kadin, cold)
    preferred_product = np.array(PRODUCTS)[shares.argmax(axis=1)]
    price_mix = shares @ P_PRICE   # müşterinin ürün karışımına göre ortalama ürün fiyatı

    # 4) Sipariş davranışı (RFM çekirdeği)
    step(0.30, "Sipariş geçmişi üretiliyor...")
    lam = np.exp(-0.35 + 0.45 * e + 0.12 * income_z + rng.normal(0, 0.25, n))  # aylık sipariş hızı
    order_count = 1 + rng.poisson(lam * tenure_months)
    avg_gap = np.clip(tenure_days / order_count, 1, 365)

    recency_days = rng.exponential(avg_gap * np.exp(-0.55 * e))
    recency_days = np.clip(recency_days, 0, np.minimum(tenure_days, 540)).round().astype(int)
    recency_ratio = np.clip(recency_days / avg_gap, 0, 20)

    # Son 3 ay / önceki 3 ay / daha eski siparişler: recency ile tutarlı
    trend = np.exp(0.30 * e + rng.normal(0, 0.30, n))
    o3 = rng.poisson(3 * lam * trend)
    o3 = np.where(recency_days > 90, 0, np.maximum(o3, 1))
    p3 = rng.poisson(3 * lam / trend)
    p3 = np.where((recency_days > 180) | (tenure_days <= 90), 0, p3)
    p3 = np.where((recency_days > 90) & (recency_days <= 180), np.maximum(p3, 1), p3)
    older_months = np.clip(np.minimum(tenure_months, 12) - 6, 0, 6)
    o_old = rng.poisson(lam * older_months)
    o_old = np.where(recency_days > 365, 0, o_old)
    o_old = np.where((recency_days > 180) & (recency_days <= 365), np.maximum(o_old, 1), o_old)
    orders_last_12m = o3 + p3 + o_old
    order_count = np.clip(np.maximum(order_count, orders_last_12m), 1, 400)
    orders_last_12m = np.minimum(orders_last_12m, order_count)
    o3 = np.minimum(o3, orders_last_12m)
    p3 = np.minimum(p3, orders_last_12m - o3)

    # Parasal: sepet = ürün karışımı fiyatı x gelir etkisi x gürültü
    avg_basket = np.clip(
        price_mix * np.exp(0.25 * income_z + 0.05 * e + rng.normal(0, 0.20, n)), 40, 8000
    ).round(2)
    total_spent = (order_count * avg_basket * rng.uniform(0.9, 1.1, n)).round(2)
    spend_last_12m = (orders_last_12m * avg_basket * rng.uniform(0.9, 1.1, n)).round(2)

    # 5) İndirim, iade, şikayet, memnuniyet
    step(0.50, "Memnuniyet ve dijital davranış üretiliyor...")
    discount_order_ratio = rng.beta(1 + 6 * price_sens, 1 + 6 * (1 - price_sens)).round(3)
    return_rate_true = np.clip(0.03 + 0.10 * discount_order_ratio - 0.015 * e + rng.normal(0, 0.02, n), 0, 0.5)
    return_count = rng.binomial(order_count, return_rate_true)
    return_rate = (return_count / order_count).round(4)

    complaint_rate = np.clip(0.02 * np.exp(-0.5 * e) + 0.10 * return_rate_true, 0.001, 0.2)
    complaint_count = rng.poisson(order_count * complaint_rate)
    support_tickets = complaint_count + rng.poisson(0.03 * order_count + 0.2)

    csat = np.clip(3.9 + 0.35 * e - 0.15 * complaint_count + rng.normal(0, 0.6, n), 1, 5).round(1)
    nps = np.clip(np.round(7 + 1.4 * e + 1.2 * (csat - 3.9) + rng.normal(0, 1.2, n)), 0, 10).astype(int)

    # 6) Dijital etkileşim ve kanal payları
    app_sessions = np.clip(rng.lognormal(np.log(6) + 0.35 * e + 0.25 * (age < 35), 0.6), 0, 120).round(1)
    web_visits = np.clip(rng.lognormal(np.log(5) + 0.20 * e, 0.6), 0, 100).round(1)
    ch_alpha = np.column_stack([1.5 + 2 * (age < 35), np.full(n, 2.0), 1.5 + 2 * (age >= 45)])
    share_app, share_web, share_store = dirichlet_rows(rng, ch_alpha).T

    # 7) Kampanya hunisi
    step(0.65, "Kampanya geçmişi üretiliyor...")
    campaigns_received = rng.poisson(20 * np.minimum(tenure_months, 12) / 12)
    campaign_opens = rng.binomial(campaigns_received, sigmoid(-1.1 + 0.5 * e))
    campaign_clicks = rng.binomial(campaign_opens, sigmoid(-1.6 + 0.35 * e + 0.5 * price_sens))
    campaign_conversions = rng.binomial(campaign_clicks, sigmoid(-1.4 + 0.30 * e + 1.2 * (price_sens - 0.4)))
    safe = lambda a, b: np.where(b > 0, a / np.maximum(b, 1), 0.0)
    email_open_rate = safe(campaign_opens, campaigns_received).round(3)
    email_click_rate = safe(campaign_clicks, campaign_opens).round(3)
    campaign_conversion_rate = safe(campaign_conversions, campaigns_received).round(3)
    email_unsubscribed = rng.binomial(1, sigmoid(-2.2 - 0.8 * e))

    # 8) Tavsiye (referral)
    referral_count = rng.poisson(0.10 * np.exp(0.6 * e))

    # 9) Sezon: önümüzdeki 90 günde ürün bazlı satın alma olasılıkları (tarihe göre)
    season = season_info(ref_date)
    step(0.75, f"Sezon olasılıkları hesaplanıyor ({season['dominant_season_label']} ağırlıklı)...")
    season_prob, n_window = season_purchase_probs(shares, cold, lam, trend, recency_ratio, ref_date)
    top_season_product = np.array(PRODUCTS)[season_prob.argmax(axis=1)]

    # 10) Sadakat skoru (0-100)
    step(0.85, "Sadakat skoru ve churn etiketi hesaplanıyor...")
    R = np.exp(-recency_ratio / 1.5)
    F = pct(orders_last_12m)
    M = pct(spend_last_12m)
    T = 1 - np.exp(-tenure_months / 30)
    S = 0.5 * nps / 10 + 0.5 * (csat - 1) / 4
    E = (0.5 * pct(app_sessions + web_visits) + 0.3 * email_open_rate
         + 0.2 * np.clip(referral_count / 2, 0, 1))
    Tr = sigmoid(1.2 * np.log((o3 + 1) / (p3 + 1)))
    penalty = (15 * return_rate
               + 8 * np.clip(complaint_count / order_count * 10, 0, 1)
               + 5 * email_unsubscribed)
    raw = (100 * (0.25 * R + 0.20 * F + 0.15 * M + 0.10 * T + 0.15 * S + 0.10 * E + 0.05 * Tr)
           - penalty + 5 * e + rng.normal(0, 3, n))
    loyalty_score = np.clip(55 + 17 * (raw - raw.mean()) / raw.std(), 0, 100).round(1)
    loyalty_segment = pd.cut(
        loyalty_score, bins=[-0.1, 35, 50, 65, 80, 100.1],
        labels=["Kayip_Riski", "Riskli", "Potansiyel", "Sadik", "Sampiyon"],
    ).astype(str)

    # 10b) Churn etiketi: REF_DATE'ten sonraki 90 günü simüle et.
    # Güncel alışveriş hızı (aylık) = geçmiş hız x mantıksal çarpanlar:
    #   trend (son 3 ay / önceki 3 ay)          -> yükselen müşteri daha çok alır
    #   kopma: recency_ratio büyüdükçe düşer    -> kendi ritminden geç kalan azalır
    #   şikayet, abonelikten çıkma              -> hızı düşürür
    #   memnuniyet (csat)                       -> hızı artırır
    #   gözlenmeyen şok                         -> taşınma, maddi durum vb.
    unobserved_shock = np.exp(rng.normal(0, cfg.churn_shock_sigma, n))
    lam_base = (lam * trend
                * np.exp(-cfg.churn_decay * np.minimum(recency_ratio, 8))
                * np.exp(-0.10 * np.minimum(complaint_count, 4))
                * np.exp(-0.30 * email_unsubscribed)
                * np.exp(0.10 * (csat - 3.9))
                * unobserved_shock)
    # Pazar genelindeki alışveriş canlılığı (tüm müşterilere aynı çarpan): hedef churn oranını
    # yakalamak için ayarlanır. Poisson'da 0 sipariş olasılığı = exp(-ay * hız).
    lo, hi = -6.0, 6.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if np.exp(-HORIZON_MONTHS * lam_base * np.exp(mid)).mean() > cfg.churn_target_rate:
            lo = mid          # churn fazla -> canlılığı artır
        else:
            hi = mid
    lam_now = lam_base * np.exp((lo + hi) / 2)
    future_orders_90d = rng.poisson(HORIZON_MONTHS * lam_now)   # 90 günde verilen sipariş sayısı
    churn = (future_orders_90d == 0).astype(int)                # hiç sipariş yok -> churn

    # 11) DataFrame
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
        "snapshot_date": ref_date.strftime("%Y-%m-%d"),
        "age": age, "gender": gender, "city": city, "region": region,
        "city_cold_index": cold, "income_level": income_level,
        "payment_method": payment_method,
        "tenure_months": tenure_months,
        "first_purchase_date": (ref_date - pd.to_timedelta(tenure_days, unit="D")).strftime("%Y-%m-%d"),
        "last_purchase_date": (ref_date - pd.to_timedelta(recency_days, unit="D")).strftime("%Y-%m-%d"),
        "recency_days": recency_days,
        "order_count": order_count,
        "orders_last_12m": orders_last_12m,
        "orders_last_3m": o3,
        "orders_prev_3m": p3,
        "avg_days_between_orders": avg_gap.round(1),
        "recency_ratio": recency_ratio.round(3),
        "total_spent": total_spent,
        "spend_last_12m": spend_last_12m,
        "avg_basket": avg_basket,
        "discount_order_ratio": discount_order_ratio,
        "return_count": return_count,
        "return_rate": return_rate,
        "complaint_count": complaint_count,
        "support_tickets": support_tickets,
        "csat": csat,
        "nps": nps,
        "app_sessions_per_month": app_sessions,
        "web_visits_per_month": web_visits,
        "share_app": share_app.round(4), "share_web": share_web.round(4), "share_store": share_store.round(4),
        "preferred_product": preferred_product,
        **{f"share_{p}": shares[:, i].round(4) for i, p in enumerate(PRODUCTS)},
        "campaigns_received": campaigns_received,
        "campaign_opens": campaign_opens,
        "campaign_clicks": campaign_clicks,
        "campaign_conversions": campaign_conversions,
        "email_open_rate": email_open_rate,
        "email_click_rate": email_click_rate,
        "campaign_conversion_rate": campaign_conversion_rate,
        "email_unsubscribed": email_unsubscribed,
        "referral_count": referral_count,
        "expected_season_orders": n_window.round(2),
        **{f"season_prob_{p}": season_prob[:, i].round(4) for i, p in enumerate(PRODUCTS)},
        "top_season_product": top_season_product,
        "loyalty_score": loyalty_score,
        "loyalty_segment": loyalty_segment,
        "churn": churn,
    })

    # 12) Doğrulama (eksik değer eklenmeden önce)
    step(0.93, "Tutarlılık kontrolleri yapılıyor...")
    checks = validate_dataset(df, cfg)
    failed = [c for c in checks if not c["passed"]]
    if failed:
        raise DataValidationError("Kontrol başarısız: " + "; ".join(f"{c['check']} ({c['detail']})" for c in failed))
    df.attrs["checks"] = checks

    # 13) Gerçekçi eksik değerler
    for col in ["age", "csat", "nps", "avg_basket", "app_sessions_per_month", "web_visits_per_month"]:
        mask = rng.random(n) < cfg.missing_rate
        df[col] = df[col].astype("float").mask(mask)

    step(1.0, "Tamamlandı")
    return df


def validate_dataset(df: pd.DataFrame, cfg: GeneratorConfig) -> list[dict]:
    """Tutarlılık kontrolleri. Dashboard'da tablo olarak gösterilebilir."""
    ref = pd.Timestamp(df.snapshot_date.iloc[0])
    ref_days = (ref - pd.to_datetime(df.first_purchase_date)).dt.days
    rec = df.recency_days
    results = []

    def add(name, passed, detail=""):
        results.append({"check": name, "passed": bool(passed), "detail": detail})

    add("Müşteri numaraları benzersiz", df.customer_id.is_unique)
    add("12 aylık sipariş <= toplam sipariş", (df.orders_last_12m <= df.order_count).all())
    add("Son 3 ay + önceki 3 ay <= 12 ay",
        (df.orders_last_3m + df.orders_prev_3m <= df.orders_last_12m).all())
    add("Son alışveriş tarihi ile sipariş pencereleri uyumlu",
        (((rec <= 90) <= (df.orders_last_3m >= 1)).all()
         and ((rec > 365) <= (df.orders_last_12m == 0)).all()))
    add("Son alışveriş, ilk alışverişten sonra", (rec <= ref_days).all())
    add("Kampanya hunisi (gönderilen >= açılan >= tıklanan >= dönüşen)",
        ((df.campaign_conversions <= df.campaign_clicks)
         & (df.campaign_clicks <= df.campaign_opens)
         & (df.campaign_opens <= df.campaigns_received)).all())
    add("Kanal payları toplamı 1",
        np.allclose(df[["share_app", "share_web", "share_store"]].sum(axis=1), 1, atol=1e-3))
    add("Ürün payları toplamı 1",
        np.allclose(df[[f"share_{p}" for p in PRODUCTS]].sum(axis=1), 1, atol=2e-3))
    add("Erkek müşterilerde elbise/etek yok",
        ((df.gender == "erkek") & ((df.share_elbise > 0) | (df.share_etek > 0))).sum() == 0)
    add("İl sayısı en fazla 81", df.city.nunique() <= 81, f"{df.city.nunique()} il")
    add("Sadakat skoru 0-100 arasında", df.loyalty_score.between(0, 100).all())
    rate = df.churn.mean()
    add("Churn oranı hedefe yakın", abs(rate - cfg.churn_target_rate) <= 0.03,
        f"gerçekleşen %{rate * 100:.1f}, hedef %{cfg.churn_target_rate * 100:.0f}")
    sp = season_info(ref)
    heavy_winter = df[["season_prob_kazak", "season_prob_mont_kaban", "season_prob_atki_bere_eldiven"]].mean().mean()
    heavy_summer = df[["season_prob_tisort", "season_prob_sort_bermuda"]].mean().mean()
    ok = (heavy_winter > heavy_summer if sp["dominant_season"] == "kis"
          else heavy_summer > heavy_winter if sp["dominant_season"] == "yaz" else True)
    add("Sezon profili tarihle uyumlu", ok,
        f"{sp['dominant_season_label']} ağırlıklı pencere; kışlık ort. %{heavy_winter * 100:.1f}, yazlık ort. %{heavy_summer * 100:.1f}")
    prob_cols = [f"season_prob_{p}" for p in PRODUCTS]
    add("Sezon olasılıkları 0-1 arasında", df[prob_cols].stack().between(0, 1).all())
    undocumented = [c for c in df.columns if c not in _dictionary()]
    add("Tüm sütunlar veri sözlüğünde", not undocumented, ", ".join(undocumented))
    return results


# ----------------------------------------------------------------------
# Veri sözlüğü (dashboard'da gösterilir, SQL şeması için kaynak)
# rol: id | date | feature | target | leakage (hedeften türeyen) | output (sezon modeli çıktısı)
# ----------------------------------------------------------------------
def _dictionary() -> dict:
    d = {
        "customer_id": ("Kimlik", "id", "Müşteri numarası (benzersiz)"),
        "snapshot_date": ("Kimlik", "date", "Verinin referans (gözlem) tarihi; sezon buna göre belirlenir"),
        "age": ("Demografik", "feature", "Yaş"),
        "gender": ("Demografik", "feature", "Cinsiyet (kadin / erkek)"),
        "city": ("Demografik", "feature", "Yaşadığı il (81 il)"),
        "region": ("Demografik", "feature", "Coğrafi bölge (7 bölge)"),
        "city_cold_index": ("Demografik", "feature", "İlin kış soğukluk indeksi (0 sıcak - 1 çok soğuk)"),
        "income_level": ("Demografik", "feature", "Gelir düzeyi (dusuk / orta / yuksek)"),
        "payment_method": ("Demografik", "feature", "Tercih ettiği ödeme yöntemi"),
        "tenure_months": ("İlişki", "feature", "Müşteri olma süresi (ay)"),
        "first_purchase_date": ("İlişki", "date", "İlk alışveriş tarihi"),
        "last_purchase_date": ("İlişki", "date", "Son alışveriş tarihi"),
        "recency_days": ("RFM", "feature", "Son alışverişten bu yana geçen gün"),
        "order_count": ("RFM", "feature", "Toplam sipariş sayısı"),
        "orders_last_12m": ("RFM", "feature", "Son 12 aydaki sipariş sayısı"),
        "orders_last_3m": ("RFM", "feature", "Son 3 aydaki sipariş sayısı"),
        "orders_prev_3m": ("RFM", "feature", "Önceki 3 aydaki (4-6. ay) sipariş sayısı"),
        "avg_days_between_orders": ("RFM", "feature", "Ortalama sipariş aralığı (gün)"),
        "recency_ratio": ("RFM", "feature", "recency_days / ortalama sipariş aralığı (1'den büyük: kendi ritminden geç)"),
        "total_spent": ("RFM", "feature", "Toplam harcama (TL)"),
        "spend_last_12m": ("RFM", "feature", "Son 12 ay harcaması (TL)"),
        "avg_basket": ("RFM", "feature", "Ortalama sepet tutarı (TL)"),
        "discount_order_ratio": ("Davranış", "feature", "İndirimli siparişlerin oranı (0-1)"),
        "return_count": ("Davranış", "feature", "İade sayısı"),
        "return_rate": ("Davranış", "feature", "İade oranı (iade / sipariş)"),
        "complaint_count": ("Memnuniyet", "feature", "Şikayet sayısı"),
        "support_tickets": ("Memnuniyet", "feature", "Destek talebi sayısı"),
        "csat": ("Memnuniyet", "feature", "Müşteri memnuniyeti (1-5)"),
        "nps": ("Memnuniyet", "feature", "Tavsiye skoru (0-10)"),
        "app_sessions_per_month": ("Dijital", "feature", "Aylık uygulama oturumu"),
        "web_visits_per_month": ("Dijital", "feature", "Aylık web sitesi ziyareti"),
        "share_app": ("Dijital", "feature", "Siparişlerin uygulamadan oranı (0-1)"),
        "share_web": ("Dijital", "feature", "Siparişlerin web sitesinden oranı (0-1)"),
        "share_store": ("Dijital", "feature", "Siparişlerin mağazadan oranı (0-1)"),
        "preferred_product": ("Ürün tercihi", "feature", "En çok aldığı ürün"),
        "campaigns_received": ("Kampanya", "feature", "Son 12 ayda gönderilen kampanya sayısı"),
        "campaign_opens": ("Kampanya", "feature", "Açılan kampanya sayısı"),
        "campaign_clicks": ("Kampanya", "feature", "Tıklanan kampanya sayısı"),
        "campaign_conversions": ("Kampanya", "feature", "Siparişe dönen kampanya sayısı"),
        "email_open_rate": ("Kampanya", "feature", "Kampanya açma oranı (0-1)"),
        "email_click_rate": ("Kampanya", "feature", "Açılanlar içinde tıklama oranı (0-1)"),
        "campaign_conversion_rate": ("Kampanya", "feature", "Kampanyadan sipariş oranı (0-1)"),
        "email_unsubscribed": ("Kampanya", "feature", "E-posta aboneliğinden çıktı mı (0/1)"),
        "referral_count": ("Kampanya", "feature", "Tavsiye ederek getirdiği müşteri sayısı"),
        "expected_season_orders": ("Sezon", "output", "Önümüzdeki 90 günde beklenen sipariş sayısı"),
        "top_season_product": ("Sezon", "output", "Önümüzdeki 90 günde alma olasılığı en yüksek ürün"),
        "loyalty_score": ("Hedef", "target", "Sadakat skoru (0-100, yüzde)"),
        "loyalty_segment": ("Hedef", "leakage", "Sadakat segmenti (skordan türer, modele girmez)"),
        "churn": ("Hedef", "target", "Sonraki 90 günde alışveriş yok (1) / var (0)"),
    }
    for p in PRODUCTS:
        d[f"share_{p}"] = ("Ürün tercihi", "feature", f"Tüm siparişlerde {p} ürününün payı (0-1)")
        d[f"season_prob_{p}"] = ("Sezon", "output", f"Önümüzdeki 90 günde {p} alma olasılığı (0-1)")
    return d


def data_dictionary(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Sütun açıklamaları. df verilirse veri tipi ve eksik oranı da eklenir."""
    d = _dictionary()
    cols = list(df.columns) if df is not None else list(d)
    out = pd.DataFrame({
        "column": cols,
        "group": [d[c][0] if c in d else "?" for c in cols],
        "role": [d[c][1] if c in d else "?" for c in cols],
        "description": [d[c][2] if c in d else "Belgelenmemiş" for c in cols],
    })
    if df is not None:
        out["dtype"] = [str(df[c].dtype) for c in cols]
        out["missing_pct"] = [round(df[c].isna().mean() * 100, 1) for c in cols]
    return out


def build_report(df: pd.DataFrame, cfg: GeneratorConfig, seconds: float) -> dict:
    """Dashboard'ın üretim sonrası gösterdiği özet."""
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "churn_rate": float(df["churn"].mean()),
        "loyalty_mean": float(df["loyalty_score"].mean()),
        "loyalty_std": float(df["loyalty_score"].std()),
        "segment_counts": df["loyalty_segment"].value_counts().to_dict(),
        "n_cities": int(df["city"].nunique()),
        "season": {k: v for k, v in season_info(df["snapshot_date"].iloc[0]).items() if k != "fractions_array"},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "checks": df.attrs.get("checks", []),
        "config": asdict(cfg),
        "seconds": round(seconds, 1),
    }


def generate_and_store(cfg: GeneratorConfig | None = None, to_db: bool = True,
                       to_csv: bool = True, progress=None) -> tuple[pd.DataFrame, dict]:
    """Veriyi üretir, CSV'ye ve/veya veritabanına (customers tablosu) yazar.
    Döndürür: (DataFrame, özet rapor)."""
    from src.config import RAW_CSV
    from src.data_loader import save_customers

    cfg = cfg or GeneratorConfig()
    t0 = time.time()
    df = generate(cfg, progress=progress)
    if to_csv:
        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_CSV, index=False)
    if to_db:
        save_customers(df)
    return df, build_report(df, cfg, time.time() - t0)