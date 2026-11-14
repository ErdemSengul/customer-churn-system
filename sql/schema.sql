-- =====================================================================
-- Müşteri churn projesi - PostgreSQL veritabanı yapısı
-- Sorumlu: Arkın
--
-- Ne yapar?
--   Müşteri verisi 8 ayrı kaynaktan (CRM, sipariş, müşteri hizmetleri, ...)
--   geliyormuş gibi 8 tabloda tutulur. Hepsi customer_id ile birbirine bağlıdır.
--   "churn_dataset" görünümü bu 8 tabloyu tek satır = tek müşteri olacak
--   şekilde birleştirir. Ekibin geri kalanı SADECE bu görünümü okur.
--
-- Çalıştırma:
--   python -m src.data_loader setup      (önerilen, her şeyi yapar)
--   veya: psql -U churn -d churn -f sql/schema.sql
--
-- Bu dosya tekrar çalıştırılabilir: tablolar zaten varsa dokunmaz.
-- Sıfırdan kurmak için: python -m src.data_loader reset
-- =====================================================================

-- Kaynak 1 - Müşteri kartı (CRM): kim, nerede yaşıyor, ne zamandır müşteri
CREATE TABLE IF NOT EXISTS customers (
    customer_id                TEXT PRIMARY KEY,
    snapshot_date              DATE NOT NULL,
    age                        INTEGER CHECK (age BETWEEN 18 AND 100),
    gender                     TEXT NOT NULL CHECK (gender IN ('kadin', 'erkek')),
    city                       TEXT NOT NULL,
    region                     TEXT NOT NULL CHECK (region IN ('Marmara', 'Ege', 'Akdeniz', 'İç Anadolu', 'Karadeniz', 'Doğu Anadolu', 'Güneydoğu Anadolu')),
    city_cold_index            DOUBLE PRECISION NOT NULL CHECK (city_cold_index BETWEEN 0 AND 1),
    income_level               TEXT NOT NULL CHECK (income_level IN ('dusuk', 'orta', 'yuksek')),
    payment_method             TEXT NOT NULL CHECK (payment_method IN ('kredi_karti', 'kapida_odeme', 'havale', 'dijital_cuzdan')),
    tenure_months              INTEGER NOT NULL CHECK (tenure_months >= 1),
    first_purchase_date        DATE NOT NULL,
    last_purchase_date         DATE NOT NULL,
    CHECK (first_purchase_date <= last_purchase_date)
);

-- Kaynak 2 - Sipariş sistemi: ne sıklıkla, ne kadar alışveriş yapıyor, ne kadar iade ediyor
CREATE TABLE IF NOT EXISTS orders_summary (
    customer_id                TEXT PRIMARY KEY REFERENCES customers(customer_id) ON DELETE CASCADE,
    recency_days               INTEGER NOT NULL CHECK (recency_days >= 0),
    order_count                INTEGER NOT NULL CHECK (order_count >= 1),
    orders_last_12m            INTEGER NOT NULL CHECK (orders_last_12m >= 0),
    orders_last_3m             INTEGER NOT NULL CHECK (orders_last_3m >= 0),
    orders_prev_3m             INTEGER NOT NULL CHECK (orders_prev_3m >= 0),
    avg_days_between_orders    DOUBLE PRECISION NOT NULL CHECK (avg_days_between_orders > 0),
    recency_ratio              DOUBLE PRECISION NOT NULL CHECK (recency_ratio >= 0),
    total_spent                DOUBLE PRECISION NOT NULL CHECK (total_spent >= 0),
    spend_last_12m             DOUBLE PRECISION NOT NULL CHECK (spend_last_12m >= 0),
    avg_basket                 DOUBLE PRECISION CHECK (avg_basket >= 0),
    discount_order_ratio       DOUBLE PRECISION NOT NULL CHECK (discount_order_ratio BETWEEN 0 AND 1),
    return_count               INTEGER NOT NULL CHECK (return_count >= 0),
    return_rate                DOUBLE PRECISION NOT NULL CHECK (return_rate BETWEEN 0 AND 1),
    CHECK (orders_last_12m <= order_count),
    CHECK (orders_last_3m + orders_prev_3m <= orders_last_12m),
    CHECK (return_count <= order_count)
);

-- Kaynak 3 - Müşteri hizmetleri: şikayetler, destek talepleri, memnuniyet anketleri
CREATE TABLE IF NOT EXISTS customer_support (
    customer_id                TEXT PRIMARY KEY REFERENCES customers(customer_id) ON DELETE CASCADE,
    complaint_count            INTEGER NOT NULL CHECK (complaint_count >= 0),
    support_tickets            INTEGER NOT NULL CHECK (support_tickets >= 0),
    csat                       DOUBLE PRECISION CHECK (csat BETWEEN 1 AND 5),
    nps                        INTEGER CHECK (nps BETWEEN 0 AND 10)
);

-- Kaynak 4 - Uygulama / web analitiği: dijitalde ne kadar aktif, hangi kanaldan alıyor
CREATE TABLE IF NOT EXISTS digital_activity (
    customer_id                TEXT PRIMARY KEY REFERENCES customers(customer_id) ON DELETE CASCADE,
    app_sessions_per_month     DOUBLE PRECISION CHECK (app_sessions_per_month >= 0),
    web_visits_per_month       DOUBLE PRECISION CHECK (web_visits_per_month >= 0),
    share_app                  DOUBLE PRECISION NOT NULL CHECK (share_app BETWEEN 0 AND 1),
    share_web                  DOUBLE PRECISION NOT NULL CHECK (share_web BETWEEN 0 AND 1),
    share_store                DOUBLE PRECISION NOT NULL CHECK (share_store BETWEEN 0 AND 1)
);

-- Kaynak 5 - Ürün geçmişi: hangi ürünleri ne oranda alıyor
CREATE TABLE IF NOT EXISTS product_preferences (
    customer_id                TEXT PRIMARY KEY REFERENCES customers(customer_id) ON DELETE CASCADE,
    preferred_product          TEXT NOT NULL CHECK (preferred_product IN ('tisort', 'gomlek_bluz', 'kazak', 'hirka', 'sweatshirt', 'pantolon', 'jean', 'elbise', 'etek', 'mont_kaban', 'ceket_blazer', 'sort_bermuda', 'atki_bere_eldiven', 'spor_giyim', 'ic_giyim_corap')),
    share_tisort               DOUBLE PRECISION NOT NULL CHECK (share_tisort BETWEEN 0 AND 1),
    share_gomlek_bluz          DOUBLE PRECISION NOT NULL CHECK (share_gomlek_bluz BETWEEN 0 AND 1),
    share_kazak                DOUBLE PRECISION NOT NULL CHECK (share_kazak BETWEEN 0 AND 1),
    share_hirka                DOUBLE PRECISION NOT NULL CHECK (share_hirka BETWEEN 0 AND 1),
    share_sweatshirt           DOUBLE PRECISION NOT NULL CHECK (share_sweatshirt BETWEEN 0 AND 1),
    share_pantolon             DOUBLE PRECISION NOT NULL CHECK (share_pantolon BETWEEN 0 AND 1),
    share_jean                 DOUBLE PRECISION NOT NULL CHECK (share_jean BETWEEN 0 AND 1),
    share_elbise               DOUBLE PRECISION NOT NULL CHECK (share_elbise BETWEEN 0 AND 1),
    share_etek                 DOUBLE PRECISION NOT NULL CHECK (share_etek BETWEEN 0 AND 1),
    share_mont_kaban           DOUBLE PRECISION NOT NULL CHECK (share_mont_kaban BETWEEN 0 AND 1),
    share_ceket_blazer         DOUBLE PRECISION NOT NULL CHECK (share_ceket_blazer BETWEEN 0 AND 1),
    share_sort_bermuda         DOUBLE PRECISION NOT NULL CHECK (share_sort_bermuda BETWEEN 0 AND 1),
    share_atki_bere_eldiven    DOUBLE PRECISION NOT NULL CHECK (share_atki_bere_eldiven BETWEEN 0 AND 1),
    share_spor_giyim           DOUBLE PRECISION NOT NULL CHECK (share_spor_giyim BETWEEN 0 AND 1),
    share_ic_giyim_corap       DOUBLE PRECISION NOT NULL CHECK (share_ic_giyim_corap BETWEEN 0 AND 1)
);

-- Kaynak 6 - Pazarlama: kampanya e-postaları ve tavsiyeler
CREATE TABLE IF NOT EXISTS campaigns (
    customer_id                TEXT PRIMARY KEY REFERENCES customers(customer_id) ON DELETE CASCADE,
    campaigns_received         INTEGER NOT NULL CHECK (campaigns_received >= 0),
    campaign_opens             INTEGER NOT NULL CHECK (campaign_opens >= 0),
    campaign_clicks            INTEGER NOT NULL CHECK (campaign_clicks >= 0),
    campaign_conversions       INTEGER NOT NULL CHECK (campaign_conversions >= 0),
    email_open_rate            DOUBLE PRECISION NOT NULL CHECK (email_open_rate BETWEEN 0 AND 1),
    email_click_rate           DOUBLE PRECISION NOT NULL CHECK (email_click_rate BETWEEN 0 AND 1),
    campaign_conversion_rate   DOUBLE PRECISION NOT NULL CHECK (campaign_conversion_rate BETWEEN 0 AND 1),
    email_unsubscribed         SMALLINT NOT NULL CHECK (email_unsubscribed IN (0, 1)),
    referral_count             INTEGER NOT NULL CHECK (referral_count >= 0),
    CHECK (campaign_conversions <= campaign_clicks AND campaign_clicks <= campaign_opens AND campaign_opens <= campaigns_received)
);

-- Kaynak 7 - Sezon tahmini: önümüzdeki 90 günde hangi ürünü alma olasılığı (öneri modeli için)
CREATE TABLE IF NOT EXISTS season_forecast (
    customer_id                TEXT PRIMARY KEY REFERENCES customers(customer_id) ON DELETE CASCADE,
    expected_season_orders     DOUBLE PRECISION NOT NULL CHECK (expected_season_orders >= 0),
    season_prob_tisort         DOUBLE PRECISION NOT NULL CHECK (season_prob_tisort BETWEEN 0 AND 1),
    season_prob_gomlek_bluz    DOUBLE PRECISION NOT NULL CHECK (season_prob_gomlek_bluz BETWEEN 0 AND 1),
    season_prob_kazak          DOUBLE PRECISION NOT NULL CHECK (season_prob_kazak BETWEEN 0 AND 1),
    season_prob_hirka          DOUBLE PRECISION NOT NULL CHECK (season_prob_hirka BETWEEN 0 AND 1),
    season_prob_sweatshirt     DOUBLE PRECISION NOT NULL CHECK (season_prob_sweatshirt BETWEEN 0 AND 1),
    season_prob_pantolon       DOUBLE PRECISION NOT NULL CHECK (season_prob_pantolon BETWEEN 0 AND 1),
    season_prob_jean           DOUBLE PRECISION NOT NULL CHECK (season_prob_jean BETWEEN 0 AND 1),
    season_prob_elbise         DOUBLE PRECISION NOT NULL CHECK (season_prob_elbise BETWEEN 0 AND 1),
    season_prob_etek           DOUBLE PRECISION NOT NULL CHECK (season_prob_etek BETWEEN 0 AND 1),
    season_prob_mont_kaban     DOUBLE PRECISION NOT NULL CHECK (season_prob_mont_kaban BETWEEN 0 AND 1),
    season_prob_ceket_blazer   DOUBLE PRECISION NOT NULL CHECK (season_prob_ceket_blazer BETWEEN 0 AND 1),
    season_prob_sort_bermuda   DOUBLE PRECISION NOT NULL CHECK (season_prob_sort_bermuda BETWEEN 0 AND 1),
    season_prob_atki_bere_eldiven DOUBLE PRECISION NOT NULL CHECK (season_prob_atki_bere_eldiven BETWEEN 0 AND 1),
    season_prob_spor_giyim     DOUBLE PRECISION NOT NULL CHECK (season_prob_spor_giyim BETWEEN 0 AND 1),
    season_prob_ic_giyim_corap DOUBLE PRECISION NOT NULL CHECK (season_prob_ic_giyim_corap BETWEEN 0 AND 1),
    top_season_product         TEXT NOT NULL CHECK (top_season_product IN ('tisort', 'gomlek_bluz', 'kazak', 'hirka', 'sweatshirt', 'pantolon', 'jean', 'elbise', 'etek', 'mont_kaban', 'ceket_blazer', 'sort_bermuda', 'atki_bere_eldiven', 'spor_giyim', 'ic_giyim_corap'))
);

-- Kaynak 8 - Sonuçlar (tahmin edilecek hedefler): sadakat skoru ve churn
CREATE TABLE IF NOT EXISTS labels (
    customer_id                TEXT PRIMARY KEY REFERENCES customers(customer_id) ON DELETE CASCADE,
    loyalty_score              DOUBLE PRECISION NOT NULL CHECK (loyalty_score BETWEEN 0 AND 100),
    loyalty_segment            TEXT NOT NULL CHECK (loyalty_segment IN ('Kayip_Riski', 'Riskli', 'Potansiyel', 'Sadik', 'Sampiyon')),
    churn                      SMALLINT NOT NULL CHECK (churn IN (0, 1))
);

-- =====================================================================
-- BİRLEŞİK VERİ SETİ: 8 tabloyu müşteri numarasıyla yan yana getirir.
-- Sütun sırası generate_synthetic.py'nin ürettiği CSV ile birebir aynıdır.
-- Emirhan (temizlik), Semih (özellikler), Erdem (model) buradan okur.
-- =====================================================================
CREATE OR REPLACE VIEW churn_dataset AS
SELECT
    c.customer_id,
    c.snapshot_date,
    c.age,
    c.gender,
    c.city,
    c.region,
    c.city_cold_index,
    c.income_level,
    c.payment_method,
    c.tenure_months,
    c.first_purchase_date,
    c.last_purchase_date,
    o.recency_days,
    o.order_count,
    o.orders_last_12m,
    o.orders_last_3m,
    o.orders_prev_3m,
    o.avg_days_between_orders,
    o.recency_ratio,
    o.total_spent,
    o.spend_last_12m,
    o.avg_basket,
    o.discount_order_ratio,
    o.return_count,
    o.return_rate,
    s.complaint_count,
    s.support_tickets,
    s.csat,
    s.nps,
    d.app_sessions_per_month,
    d.web_visits_per_month,
    d.share_app,
    d.share_web,
    d.share_store,
    p.preferred_product,
    p.share_tisort,
    p.share_gomlek_bluz,
    p.share_kazak,
    p.share_hirka,
    p.share_sweatshirt,
    p.share_pantolon,
    p.share_jean,
    p.share_elbise,
    p.share_etek,
    p.share_mont_kaban,
    p.share_ceket_blazer,
    p.share_sort_bermuda,
    p.share_atki_bere_eldiven,
    p.share_spor_giyim,
    p.share_ic_giyim_corap,
    k.campaigns_received,
    k.campaign_opens,
    k.campaign_clicks,
    k.campaign_conversions,
    k.email_open_rate,
    k.email_click_rate,
    k.campaign_conversion_rate,
    k.email_unsubscribed,
    k.referral_count,
    f.expected_season_orders,
    f.season_prob_tisort,
    f.season_prob_gomlek_bluz,
    f.season_prob_kazak,
    f.season_prob_hirka,
    f.season_prob_sweatshirt,
    f.season_prob_pantolon,
    f.season_prob_jean,
    f.season_prob_elbise,
    f.season_prob_etek,
    f.season_prob_mont_kaban,
    f.season_prob_ceket_blazer,
    f.season_prob_sort_bermuda,
    f.season_prob_atki_bere_eldiven,
    f.season_prob_spor_giyim,
    f.season_prob_ic_giyim_corap,
    f.top_season_product,
    l.loyalty_score,
    l.loyalty_segment,
    l.churn
FROM customers c
LEFT JOIN orders_summary       o USING (customer_id)
LEFT JOIN customer_support     s USING (customer_id)
LEFT JOIN digital_activity     d USING (customer_id)
LEFT JOIN product_preferences  p USING (customer_id)
LEFT JOIN campaigns            k USING (customer_id)
LEFT JOIN season_forecast      f USING (customer_id)
LEFT JOIN labels               l USING (customer_id);

