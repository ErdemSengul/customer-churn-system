-- =====================================================================
-- Hazır analiz sorguları (pgAdmin / DBeaver / psql'de tek tek çalıştırılabilir)
-- Python'dan:  from src.data_loader import run_query;  run_query("SELECT ...")
-- =====================================================================

-- 1) Genel tablo: kaç müşteri var, churn oranı ne?
SELECT count(*)                         AS musteri_sayisi,
       sum(churn)                       AS churn_olan,
       round(avg(churn) * 100, 1)       AS churn_yuzde,
       round(avg(loyalty_score)::numeric, 1) AS ort_sadakat
FROM churn_dataset;

-- 2) Bölgeye göre churn
SELECT region, count(*) AS musteri, round(avg(churn) * 100, 1) AS churn_yuzde
FROM churn_dataset
GROUP BY region
ORDER BY churn_yuzde DESC;

-- 3) Gelir düzeyine göre churn ve ortalama sepet
SELECT income_level, count(*) AS musteri,
       round(avg(churn) * 100, 1)              AS churn_yuzde,
       round(avg(avg_basket)::numeric, 0)      AS ort_sepet_tl
FROM churn_dataset
GROUP BY income_level
ORDER BY churn_yuzde DESC;

-- 4) Sadakat segmentine göre churn (sadakat arttıkça churn düşmeli)
SELECT loyalty_segment, count(*) AS musteri, round(avg(churn) * 100, 1) AS churn_yuzde
FROM churn_dataset
GROUP BY loyalty_segment
ORDER BY min(loyalty_score);

-- 5) Son alışverişten bu yana geçen süreye göre churn
SELECT CASE
         WHEN recency_days <= 30  THEN '1) 0-30 gün'
         WHEN recency_days <= 90  THEN '2) 31-90 gün'
         WHEN recency_days <= 180 THEN '3) 91-180 gün'
         ELSE                          '4) 180+ gün'
       END AS son_alisveris,
       count(*) AS musteri,
       round(avg(churn) * 100, 1) AS churn_yuzde
FROM churn_dataset
GROUP BY 1
ORDER BY 1;

-- 6) Şikayet sayısına göre churn
SELECT LEAST(complaint_count, 4) AS sikayet_sayisi_4_ve_ustu,
       count(*) AS musteri,
       round(avg(churn) * 100, 1) AS churn_yuzde
FROM churn_dataset
GROUP BY 1
ORDER BY 1;

-- 7) E-posta aboneliğinden çıkanlar daha mı çok kaybediliyor?
SELECT email_unsubscribed, count(*) AS musteri, round(avg(churn) * 100, 1) AS churn_yuzde
FROM churn_dataset
GROUP BY email_unsubscribed;

-- 8) Eksik veri sayıları (Emirhan - veri temizleme için)
SELECT count(*) FILTER (WHERE age IS NULL)                    AS yas_eksik,
       count(*) FILTER (WHERE csat IS NULL)                   AS csat_eksik,
       count(*) FILTER (WHERE nps IS NULL)                    AS nps_eksik,
       count(*) FILTER (WHERE avg_basket IS NULL)             AS sepet_eksik,
       count(*) FILTER (WHERE app_sessions_per_month IS NULL) AS uygulama_eksik,
       count(*) FILTER (WHERE web_visits_per_month IS NULL)   AS web_eksik
FROM churn_dataset;

-- 9) Kendi alışveriş ritminden en çok geç kalan 20 müşteri (riskli liste örneği)
SELECT customer_id, city, recency_days, avg_days_between_orders, recency_ratio,
       loyalty_score, loyalty_segment
FROM churn_dataset
ORDER BY recency_ratio DESC
LIMIT 20;

-- 10) Önümüzdeki 90 günde en çok önerilecek ürünler (sezon modeli çıktısı)
SELECT top_season_product, count(*) AS musteri
FROM churn_dataset
GROUP BY top_season_product
ORDER BY musteri DESC;

-- 11) Örnek: iki kaynağı elle birleştirmek (churn_dataset bunu 8 tablo için yapar)
SELECT c.customer_id, c.city, o.order_count, o.total_spent, l.churn
FROM customers c
JOIN orders_summary o ON o.customer_id = c.customer_id
JOIN labels         l ON l.customer_id = c.customer_id
ORDER BY o.total_spent DESC
LIMIT 10;
