-- =====================================================================
-- Veri kalite kontrolleri
-- Her satır bir kontrol. "hatali_kayit" sütunu 0 ise kontrol geçti.
--
-- Çalıştırma:
--   python -m src.data_loader check
--   veya pgAdmin / DBeaver'da bu dosyayı açıp çalıştırın.
-- =====================================================================
WITH
p AS (SELECT * FROM product_preferences),
f AS (SELECT * FROM season_forecast)

-- 1) Her müşterinin 8 kaynağın hepsinde kaydı var mı?
SELECT 1 AS no, 'Sipariş kaydı olmayan müşteri' AS kontrol,
       (SELECT count(*) FROM customers c WHERE NOT EXISTS (SELECT 1 FROM orders_summary x WHERE x.customer_id = c.customer_id)) AS hatali_kayit
UNION ALL
SELECT 2, 'Müşteri hizmetleri kaydı olmayan müşteri',
       (SELECT count(*) FROM customers c WHERE NOT EXISTS (SELECT 1 FROM customer_support x WHERE x.customer_id = c.customer_id))
UNION ALL
SELECT 3, 'Dijital kaydı olmayan müşteri',
       (SELECT count(*) FROM customers c WHERE NOT EXISTS (SELECT 1 FROM digital_activity x WHERE x.customer_id = c.customer_id))
UNION ALL
SELECT 4, 'Ürün tercihi kaydı olmayan müşteri',
       (SELECT count(*) FROM customers c WHERE NOT EXISTS (SELECT 1 FROM product_preferences x WHERE x.customer_id = c.customer_id))
UNION ALL
SELECT 5, 'Kampanya kaydı olmayan müşteri',
       (SELECT count(*) FROM customers c WHERE NOT EXISTS (SELECT 1 FROM campaigns x WHERE x.customer_id = c.customer_id))
UNION ALL
SELECT 6, 'Sezon tahmini olmayan müşteri',
       (SELECT count(*) FROM customers c WHERE NOT EXISTS (SELECT 1 FROM season_forecast x WHERE x.customer_id = c.customer_id))
UNION ALL
SELECT 7, 'Churn / sadakat etiketi olmayan müşteri',
       (SELECT count(*) FROM customers c WHERE NOT EXISTS (SELECT 1 FROM labels x WHERE x.customer_id = c.customer_id))

-- 2) Birleştirme sonrası satır çoğalması var mı? (1 müşteri = 1 satır olmalı)
UNION ALL
SELECT 8, 'Birleşik veride tekrar eden müşteri',
       (SELECT count(*) - count(DISTINCT customer_id) FROM churn_dataset)

-- 3) Tarih tutarlılığı
UNION ALL
SELECT 9, 'Birden fazla referans tarihi (snapshot_date)',
       (SELECT GREATEST(count(DISTINCT snapshot_date) - 1, 0) FROM customers)
UNION ALL
SELECT 10, 'Son alışveriş tarihi referans tarihinden sonra',
       (SELECT count(*) FROM customers WHERE last_purchase_date > snapshot_date)
UNION ALL
SELECT 11, 'recency_days ile son alışveriş tarihi uyuşmuyor',
       (SELECT count(*) FROM churn_dataset WHERE snapshot_date - last_purchase_date <> recency_days)

-- 4) Oranlar
UNION ALL
SELECT 12, 'Kanal payları toplamı 1 değil (uygulama + web + mağaza)',
       (SELECT count(*) FROM digital_activity WHERE abs(share_app + share_web + share_store - 1) > 0.001)
UNION ALL
SELECT 13, 'Ürün payları toplamı 1 değil',
       (SELECT count(*) FROM p WHERE abs(p.share_tisort + p.share_gomlek_bluz + p.share_kazak + p.share_hirka + p.share_sweatshirt + p.share_pantolon + p.share_jean + p.share_elbise + p.share_etek + p.share_mont_kaban + p.share_ceket_blazer + p.share_sort_bermuda + p.share_atki_bere_eldiven + p.share_spor_giyim + p.share_ic_giyim_corap - 1) > 0.002)
UNION ALL
SELECT 14, 'En çok alınan ürün (preferred_product) en yüksek paya sahip değil',
       (SELECT count(*) FROM p
         WHERE (to_jsonb(p) ->> ('share_' || p.preferred_product))::float8
             < GREATEST(p.share_tisort, p.share_gomlek_bluz, p.share_kazak, p.share_hirka, p.share_sweatshirt, p.share_pantolon, p.share_jean, p.share_elbise, p.share_etek, p.share_mont_kaban, p.share_ceket_blazer, p.share_sort_bermuda, p.share_atki_bere_eldiven, p.share_spor_giyim, p.share_ic_giyim_corap))
UNION ALL
SELECT 15, 'Sezon ürünü (top_season_product) en yüksek olasılığa sahip değil',
       (SELECT count(*) FROM f
         WHERE (to_jsonb(f) ->> ('season_prob_' || f.top_season_product))::float8
             < GREATEST(f.season_prob_tisort, f.season_prob_gomlek_bluz, f.season_prob_kazak, f.season_prob_hirka, f.season_prob_sweatshirt, f.season_prob_pantolon, f.season_prob_jean, f.season_prob_elbise, f.season_prob_etek, f.season_prob_mont_kaban, f.season_prob_ceket_blazer, f.season_prob_sort_bermuda, f.season_prob_atki_bere_eldiven, f.season_prob_spor_giyim, f.season_prob_ic_giyim_corap))

-- 5) İş kuralları
UNION ALL
SELECT 16, 'Erkek müşteride elbise / etek alışverişi',
       (SELECT count(*) FROM customers c JOIN p USING (customer_id)
         WHERE c.gender = 'erkek' AND (p.share_elbise > 0 OR p.share_etek > 0))
UNION ALL
SELECT 17, 'Sadakat segmenti skorla uyuşmuyor',
       (SELECT count(*) FROM labels WHERE loyalty_segment <> CASE
            WHEN loyalty_score <= 35 THEN 'Kayip_Riski'
            WHEN loyalty_score <= 50 THEN 'Riskli'
            WHEN loyalty_score <= 65 THEN 'Potansiyel'
            WHEN loyalty_score <= 80 THEN 'Sadik'
            ELSE 'Sampiyon' END)
ORDER BY no;
