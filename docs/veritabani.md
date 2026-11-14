# Veritabanı (PostgreSQL) — Arkın'ın teslim notu

## Kısaca ne yapıldı?

Müşteri bilgileri gerçek bir şirkette tek bir yerde durmaz. Her departmanın kendi defteri vardır:
müşteri kartı bir yerde, siparişler başka yerde, şikayetler müşteri hizmetlerinde, kampanya e-postaları
pazarlamada... Bu projede de veriyi böyle **8 ayrı defter (tablo)** halinde kurduk. Her defterde
müşteri numarası (`customer_id`) yazıyor; bu numara sayesinde defterler birbirine bağlanıyor.

Sonra bu 8 defteri müşteri numarasına göre yan yana koyup **tek bir büyük liste** çıkardık:
**`churn_dataset`**. Bu listede her satır bir müşteri, her sütun o müşteriyle ilgili bir bilgi
(toplam 79 sütun). **Ekibin geri kalanı sadece bu listeyi okur**, 8 defterle uğraşmaz.

```
 customers  orders_summary  customer_support  digital_activity
      \            \              |              /
       \            \             |             /
        ------------  customer_id ile birleşir ------------> churn_dataset  (1 satır = 1 müşteri)
       /            /             |             \
 product_preferences  campaigns  season_forecast  labels
```

| Tablo | Hangi "departmanın" defteri | İçinde ne var |
|---|---|---|
| `customers` | Müşteri kartı (CRM) | yaş, cinsiyet, il, bölge, gelir, ödeme yöntemi, müşteri olma süresi, ilk/son alışveriş tarihi |
| `orders_summary` | Sipariş sistemi | sipariş sayıları, harcama, sepet tutarı, indirim, iade |
| `customer_support` | Müşteri hizmetleri | şikayet, destek talebi, memnuniyet (csat), tavsiye skoru (nps) |
| `digital_activity` | Uygulama / web | aylık uygulama ve site ziyareti, hangi kanaldan aldığı |
| `product_preferences` | Ürün geçmişi | en çok aldığı ürün, 15 ürünün payları |
| `campaigns` | Pazarlama | kampanya e-postaları, açma/tıklama, abonelikten çıkma, tavsiye |
| `season_forecast` | Sezon tahmini | önümüzdeki 90 günde hangi ürünü alma olasılığı |
| `labels` | Sonuçlar (tahmin edilecek) | sadakat skoru, sadakat segmenti, **churn** |

### Veritabanı "kapı görevlisi" gibi çalışır

Tablolara kurallar koyduk. Kurala uymayan veri **içeri giremez** ve o yükleme tamamen iptal olur
(yarım yamalak veri kalmaz, eski veri olduğu gibi durur). Örnekler:

- cinsiyet sadece `kadin` / `erkek` olabilir, bölge sadece 7 bölgeden biri olabilir
- oranlar 0 ile 1 arasında, memnuniyet 1-5, tavsiye skoru 0-10, sadakat skoru 0-100 arasında olmalı
- son 12 aydaki sipariş, toplam siparişten fazla olamaz; iade sayısı sipariş sayısından fazla olamaz
- kampanya hunisi: gönderilen ≥ açılan ≥ tıklanan ≥ siparişe dönen
- aynı müşteri numarası iki kez olamaz; 8 defterden birinde olmayan bir müşteri diğerlerine yazılamaz

Ayrıca yükledikten sonra çalışan **17 kalite kontrolü** var (`python -m src.data_loader check`):
her müşterinin 8 defterde de kaydı var mı, birleştirince satır çoğaldı mı, tarihler tutuyor mu,
ürün payları toplamı 1 mi, sadakat segmenti skorla uyuşuyor mu, vb.

---

## Kurulum (ilk sefer, ~10 dakika)

Önce projeyi bilgisayara indirip (git clone / git pull) proje klasöründe bir terminal açın.

### 1. Python paketleri

```bash
# Windows
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Mac / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ayar dosyası

```bash
copy .env.example .env      # Windows
cp .env.example .env        # Mac / Linux
```

`.env` içindeki şifre GitHub'a gitmez (`.gitignore`'da). Varsayılan değerlerle her şey çalışır.

### 3. PostgreSQL'i açın — iki yoldan birini seçin

**Yol A — Docker (önerilen, herkeste aynı kurulum):**
[Docker Desktop](https://www.docker.com/products/docker-desktop/) kurulu ve açık olsun, sonra:

```bash
docker compose up -d        # veritabanını açar (ilk sefer internetten indirir)
docker compose ps           # "healthy" yazıyorsa hazır
```

Kapatmak: `docker compose down` · Verileri de silmek: `docker compose down -v`

**Yol B — PostgreSQL'i doğrudan kurmak:**
[postgresql.org/download](https://www.postgresql.org/download/) adresinden kurun (kurulumda verilen
`postgres` şifresini not edin). Sonra kurulumla gelen **SQL Shell (psql)** veya **pgAdmin** ile
`postgres` kullanıcısıyla bağlanıp şunu çalıştırın:

```sql
CREATE USER churn WITH PASSWORD 'churn';
CREATE DATABASE churn OWNER churn;
```

> Bilgisayarda hem Docker'lı hem kurulu PostgreSQL varsa ikisi aynı kapıyı (5432) kullanmak ister.
> Bu durumda `.env` içinde `POSTGRES_PORT=5433` yapın.

### 4. Tek komutla hazırla

```bash
python -m src.data_loader setup
```

Bu komut tabloları kurar, `data/raw/customers.csv` dosyasını veritabanına yükler (dosya yoksa yeni veri
üretir) ve 17 kontrolü çalıştırır. Sonunda `17/17 kontrol geçti.` görmelisiniz.

### 5. Test

```bash
python -m pytest -v test/test_database.py
```

---

## Komutlar

| Komut | Ne yapar |
|---|---|
| `python -m src.data_loader setup` | Her şeyi hazırlar (ilk sefer bunu çalıştırın) |
| `python -m src.data_loader generate --n 20000 --churn 0.30` | Yeni veri üretir, CSV'ye ve veritabanına yazar |
| `python -m src.data_loader load` | `data/raw/customers.csv` dosyasını veritabanına yükler |
| `python -m src.data_loader export` | Veritabanındaki birleşik listeyi CSV'ye yazar |
| `python -m src.data_loader check` | 17 kalite kontrolü |
| `python -m src.data_loader reset` | Tabloları silip boş kurar (veriler gider) |

Hazır analiz sorguları `sql/queries.sql` içinde (bölgeye / gelire / sadakate göre churn, eksik
veri sayıları, riskli müşteriler...). pgAdmin veya DBeaver'da açıp tek tek çalıştırabilirsiniz.

---

## Ekip arkadaşları veriyi nasıl kullanır?

```python
from src.data_loader import load_customers, run_query

df = load_customers()     # birleşik veri seti, 10.000 satır x 79 sütun (pandas DataFrame)

# Kendi sorgunuz:
ege = run_query("SELECT * FROM churn_dataset WHERE region = %s", ("Ege",))
```

`load_customers()` ile `pd.read_csv("data/raw/customers.csv")` **birebir aynı** tabloyu verir
(sütunlar, sırası, tipleri, eksik değerler aynı — test edildi). Veritabanı kuramayan biri CSV ile
devam edebilir.

- **Emirhan (temizlik):** `load_customers()` ile oku. Eksik değerler bilerek bırakıldı:
  `age, csat, nps, avg_basket, app_sessions_per_month, web_visits_per_month` (her birinde ~%2).
- **Semih (özellikler):** aynı tablo. RFM sütunları `orders_summary` tablosundan geliyor.
- **Erdem (model):** hedefler `churn` ve `loyalty_score`. `loyalty_segment` skordan türediği için
  modele girmemeli. Dosya yolu için `from src.config import RAW_CSV` kullanılabilir.
- **Enes (API / dashboard):** dashboard'da tablo veya grafik için `run_query(...)` kullanılabilir.

---

## Bu teslimde eklenen / değişen dosyalar

| Dosya | Açıklama |
|---|---|
| `sql/schema.sql` | 8 tablo + kurallar + `churn_dataset` birleşik görünümü |
| `sql/quality_checks.sql` | 17 kalite kontrolü |
| `sql/queries.sql` | Hazır analiz sorguları |
| `src/config.py` | Klasör yolları ve veritabanı ayarları (`.env`'den okur) |
| `src/data_loader.py` | Veritabanına yazma / okuma / kontrol (eski boş `dataloader.py` yerine; `generate_synthetic.py` zaten `src.data_loader` adını bekliyordu) |
| `test/test_database.py` | Veritabanı testleri (PostgreSQL kapalıysa "atlandı" olur) |
| `docker-compose.yml` | PostgreSQL'i tek komutla açar |
| `.env.example`, `.gitignore` | Ayar şablonu; şifrelerin GitHub'a gitmemesi |
| `requirements.txt` | `psycopg` ve `python-dotenv` eklendi; dosya UTF-8'e çevrildi |
| `data/raw/customers.csv` | Güncel üreticiyle yeniden üretilmiş veri (10.000 müşteri, %25 churn) |
| `data/raw.db/customers.csv` | **Kaldırıldı.** Üreticinin eski sürümünden kalmıştı (`winter_*` sütunları, `snapshot_date` yoktu) |

## Ekibin bilmesi gerekenler

- `test/test_dataset.py` hâlâ üreticinin **eski** sütun adlarını bekliyor (`winter_prob_*`,
  `top_winter_product`, `expected_winter_orders`, `loyalty_member`, `loyalty_tier`, `points_balance`).
  Üretici artık `season_prob_*`, `top_season_product`, `expected_season_orders`, `snapshot_date`
  üretiyor. Bu yüzden 24 test kırmızı; testin yeni adlara göre güncellenmesi gerekiyor.
- `src/pipeline.py` içindeki `DATA_PATH` proje klasörünün bir üstünü gösteriyor ve `winter_*`
  sütunlarını arıyor. `from src.config import RAW_CSV` ile ve `season_*` adlarıyla düzeltilebilir.
- Üreticide (`generate_synthetic.py`) sütun eklenir/çıkarılırsa `sql/schema.sql` de güncellenmeli.
  `test_database.py` içindeki bir test bunu otomatik yakalar.
