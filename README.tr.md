<div align="center">

# VectraX

Operatör güdümlü, gerçek zamanlı görsel takip.

[![License](https://img.shields.io/github/license/ismailcelik-tr/vectrax)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-macOS-000000?logo=apple&logoColor=white)](#platform-desteği)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Last commit](https://img.shields.io/github/last-commit/ismailcelik-tr/vectrax)](https://github.com/ismailcelik-tr/vectrax/commits/main)

[![English](https://img.shields.io/badge/English-555555?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2MCAzMCI%2BPGNsaXBQYXRoIGlkPSJzIj48cGF0aCBkPSJNMCwwdjMwaDYwVjB6Ii8%2BPC9jbGlwUGF0aD48Y2xpcFBhdGggaWQ9InQiPjxwYXRoIGQ9Ik0zMCwxNWgzMHYxNXp2MTVIMHpIMFYwelYwaDMweiIvPjwvY2xpcFBhdGg%2BPGcgY2xpcC1wYXRoPSJ1cmwoI3MpIj48cGF0aCBkPSJNMCwwdjMwaDYwVjB6IiBmaWxsPSIjMDEyMTY5Ii8%2BPHBhdGggZD0iTTAsMEw2MCwzME02MCwwTDAsMzAiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSI2Ii8%2BPHBhdGggZD0iTTAsMEw2MCwzME02MCwwTDAsMzAiIGNsaXAtcGF0aD0idXJsKCN0KSIgc3Ryb2tlPSIjQzgxMDJFIiBzdHJva2Utd2lkdGg9IjQiLz48cGF0aCBkPSJNMzAsMHYzME0wLDE1aDYwIiBzdHJva2U9IiNmZmYiIHN0cm9rZS13aWR0aD0iMTAiLz48cGF0aCBkPSJNMzAsMHYzME0wLDE1aDYwIiBzdHJva2U9IiNDODEwMkUiIHN0cm9rZS13aWR0aD0iNiIvPjwvZz48L3N2Zz4%3D&logoSize=auto)](README.md)
[![Türkçe](https://img.shields.io/badge/T%C3%BCrk%C3%A7e-555555?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzMCAyMCI%2BPHJlY3Qgd2lkdGg9IjMwIiBoZWlnaHQ9IjIwIiBmaWxsPSIjRTMwQTE3Ii8%2BPGNpcmNsZSBjeD0iMTAuMCIgY3k9IjEwIiByPSI1LjAiIGZpbGw9IiNmZmYiLz48Y2lyY2xlIGN4PSIxMS4yNSIgY3k9IjEwIiByPSI0LjAiIGZpbGw9IiNFMzBBMTciLz48cG9seWdvbiBwb2ludHM9IjE1LjkxNywxMC4wMDAgMTcuNjQ0LDkuNDM5IDE3LjY0NCw3LjYyMiAxOC43MTIsOS4wOTIgMjAuNDM5LDguNTMxIDE5LjM3MiwxMC4wMDAgMjAuNDM5LDExLjQ2OSAxOC43MTIsMTAuOTA4IDE3LjY0NCwxMi4zNzggMTcuNjQ0LDEwLjU2MSIgZmlsbD0iI2ZmZiIvPjwvc3ZnPg%3D%3D&logoSize=auto)](README.tr.md)

</div>

Operatör herhangi bir nesnenin etrafına kutu çizer; VectraX onu kare kare
izler, ne kadar emin olduğunu bildirir ve hiçbir detector'ün tanımadığı
nesnelerde de çalışır. Kullanım alanları: izleme, robotik, denetim ve
araştırma. v1 hiçbir donanımı hareket ettirmez.

## Durum

7 fazın 3.'sü kapandı: tracker'ların yanında bir detector çalışıyor ve bir
track'in güvenini yükseltebiliyor. Sıradaki Phase 4, occlusion ve yeniden
bulma (reacquisition): kaybolup geri gelen bir hedef henüz yeniden
bulunmuyor. Tahmin ve ekranda yönlendirme Phase 5'te gelecek. İlerleme:
[docs/ROADMAP.md](docs/ROADMAP.md).

## Nasıl çalışır

```
Kamera / dosya → LatestFrameBuffer (drop-oldest) → pipeline tick, her karede:
  1. her track'i propagate et
  2. biten detector sonuçlarını birleştir (late-result fusion)
  3. TrackManager: eşleştir, kaliteyi güncelle, state machine'i çalıştır
  4. TrackSnapshot yayınla → UI, recorder, metrics

InferenceWorker (kendi thread'i, her N karede bir) → sonuçlar → 2. adım.
Tick asla inference'ı beklemez.
```

- **Propagator:** her hedef için bir class-agnostic single-object tracker:
  OpenCV NanoTrack; kaliteyi bizim NCC görünüm skorumuz belirler (ADR-008).
- **Detector:** Core ML fp16 üzerinde RF-DETR Nano (ADR-009). Yalnız kanıt
  sağlar: eşleşen bir detection track'i yükseltebilir, detection'ın yokluğu
  onu asla düşürmez (ADR-010).
- **TrackManager:** track oluşturan, durumunu değiştiren ya da silen tek
  bileşen. Her track'in bir Kalman filtresi var. Durumlar: INITIALIZING,
  TRACKING, DEGRADED, OCCLUDED, LOST, PAUSED, STOPPED.
- **Çalışma modları:** realtime kare düşürür ve inference'ı asenkron
  çalıştırır; deterministic her kareyi işler ve bir koşuyu birebir
  tekrarlar. Tüm zaman enjekte edilen monotonic clock'tan gelir.

Tasarım kararları ve kanıtları: [docs/DECISIONS.md](docs/DECISIONS.md).

## Platform desteği

Kareler `CameraSource` arayüzünden girer; takip, bir kareyi hangi kaynağın
ürettiğini bilmez. Yeni bir kamera yeni bir source demektir; tracker
değişmez.

| Katman | Bugün | Sonra ([SPEC](docs/SPEC.md)) |
|---|---|---|
| Kamera kaynağı | macOS kameraları (AVFoundation) | USB (UVC), RTSP |
| Dosya kaynağı | video klipleri | — |
| Detector backend | Core ML | CUDA / TensorRT, edge cihazlar |

## Gereksinimler

- Bugün: Apple Silicon üzerinde macOS. Geliştirme makinesi:
  [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
- Python 3.13 (coremltools'un 3.14 wheel'i yok) ve [uv](https://docs.astral.sh/uv/).
- macOS kamerası: terminale kamera izni ver; Center Stage'i kapat.

## Kurulum

```sh
git clone https://github.com/ismailcelik-tr/vectrax.git
cd vectrax
uv sync
```

`uv sync` tüm dependency grup'larını kurar; torch ve yalnız benchmark
referansı olan AGPL lisanslı `ultralytics` de dahil. Gruplar:
[docs/SETUP.md](docs/SETUP.md).

Model ağırlıkları repoda yok (`models/` git-ignored). Kaynaklar ve SHA-256
checksum'ları: [docs/SETUP.md](docs/SETUP.md).

- **Tracker (zorunlu):** `models/trackers/nanotrack_backbone_sim.onnx` ve
  `models/trackers/nanotrack_head_sim.onnx`.
- **Detector (isteğe bağlı, `--detect` için):**
  `models/detectors/rf-detr-nano.pth` dosyasını yerleştir, sonra Core ML'e
  export et:

  ```sh
  uv run benchmarks/export_detectors.py --detector rfdetr_n --format coreml
  ```

  Çıktı: `models/detectors/exported/rfdetr_n/rfdetr-nano_fp16.mlpackage`.

## Kullanım

```sh
# Canlı kamera; her hedefin etrafına kutu sürükle
uv run vectrax --source camera:MacBook

# Detector açık, oturum data/sessions/NAME altına kaydediliyor
uv run vectrax --source camera:MacBook \
  --detect models/detectors/exported/rfdetr_n/rfdetr-nano_fp16.mlpackage \
  --record NAME

# Video dosyası; hedefler çizilebilsin diye 0. karede donmuş açılır
uv run vectrax --source file:path/to/clip.mp4

# Headless, deterministic, kare başına JSONL
uv run vectrax --source file:clip.mp4 --headless \
  --init-boxes 320,180,120,90 --out run.jsonl

# Kayıtlı oturumu tekrar oynat ve inceleme videosu üret
uv run vectrax --source file:data/sessions/NAME/video.mp4 --headless \
  --render-out review.mp4
```

macOS'ta `camera:NAME` cihaz adının bir parçasıyla ya da unique ID'siyle eşleşir.
Headless koşular operatör girdisini bulunan ilk kaynaktan alır:
`--init-boxes`, oturumun `operator.jsonl` dosyası, klibin
`<clip>.init.json` dosyası.

Diğer flag'ler: `--scale` (propagator küçültme oranı), `--detect-stride`
(her N karede bir detection), `--metrics-out` (latency özeti JSON).
Bkz. `uv run vectrax --help`.

### Tuşlar

| Tuş | İşlev |
|---|---|
| sürükle | yeni hedef seç |
| tıkla, `1`–`9` | bir track'e odaklan |
| `c` | odağı kaldır |
| `p` | odaktaki track'i duraklat / sürdür |
| `s` | odaktaki track'i durdur |
| `x` | odaktaki track'i sil |
| `r`, sonra sürükle | odaktaki track'i yeniden seç |
| space | dondur / çöz (yalnız dosya kaynağında) |
| `q`, Esc | çık |

## Geliştirme

```sh
uv run pytest
uv run ruff check .
VECTRAX_CAMERA=1 uv run pytest tests/test_mac_camera.py  # kamera gerekir
```

Kodu şekillendiren kurallar ([CLAUDE.md](CLAUDE.md)):

- Algoritmik kod failing test ile başlar.
- Gerçek zamanlı yolda veritabanı ve ağ yoktur; bu yol inference'ı asla
  beklemez.
- Her performans ya da kalite sayısı bu repoda ölçülür; komutunu ve ham
  çıktısını gösterir.

## Benchmark ve değerlendirme

| Script | Ölçtüğü |
|---|---|
| `benchmarks/tracking_eval.py` | etiketli fixture'larda takip doğruluğu |
| `benchmarks/live_latency.py` | canlı kamera latency'si, 1–3 hedef |
| `benchmarks/detection_eval.py` | fixture'larda detector precision / recall |
| `benchmarks/detector_latency.py` | backend başına detector latency, bellek, güç |

Ham sonuçlar: `benchmarks/results/`. Sayılar:
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) (hız),
[docs/EVALUATION.md](docs/EVALUATION.md) (doğruluk).

Fixture'lar geliştirme kamerasıyla kaydedilip CVAT'ta etiketlenmiş
kliplerdir. git-ignored olan `data/fixtures/` altında durur, bu yüzden
dağıtılmaz. Kayıt ve etiketleme: [docs/FIXTURES.md](docs/FIXTURES.md).

## Proje yapısı

```
src/vectrax/
  sources/      CameraSource arayüzü; macOS kamerası, dosya
  tracking/     propagator'lar, Kalman, kalite, durumlar, association, TrackManager
  detection/    Core ML detector, inference worker
  evaluation/   ground-truth yükleyici, metrikler
  ui/           OpenCV penceresi
  pipeline.py   pipeline tick, çalışma modları
  recording.py  oturum kaydı
benchmarks/     ölçümler
scripts/        kamera probu, fixture kaydı, CVAT, SAM 2 ön etiketleri
tests/
docs/
```

## Dokümantasyon

Belgeler İngilizcedir.

| Dosya | İçerik |
|---|---|
| [SPEC.md](docs/SPEC.md) | gereksinimler, mimari, fazlar |
| [ROADMAP.md](docs/ROADMAP.md) | mevcut faz, açık işler |
| [DECISIONS.md](docs/DECISIONS.md) | ADR kaydı |
| [PERFORMANCE.md](docs/PERFORMANCE.md) | ölçülmüş latency ve throughput |
| [EVALUATION.md](docs/EVALUATION.md) | ölçülmüş doğruluk |
| [FIXTURES.md](docs/FIXTURES.md) | ground-truth klipleri |
| [SETUP.md](docs/SETUP.md) | kurulum kaydı, model ağırlıkları, bilinen sorunlar |
| [ENVIRONMENT.md](docs/ENVIRONMENT.md) | geliştirme makinesi |

## Gizlilik

Kayıtlar, fixture'lar ve ağırlıklar makinede kalır: `data/`, `models/` ve
`tools/` git-ignored. Yalnız kendini ve onay veren kişileri kaydet.

## Kapsam

v1 fiziksel bir kamerayı kontrol etmez. Silah ya da angajman mantığı
kapsam dışıdır ([docs/SPEC.md](docs/SPEC.md), Non-goals).

## Kaldırma

`scripts/uninstall.sh`, [docs/SETUP.md](docs/SETUP.md)'de kayıtlı tüm
kurulumları kaldırır; `.venv/` de dahil. Proje klasörünü silmez.

## Lisans

[Apache License 2.0](LICENSE).

Model ağırlıkları ve üçüncü taraf paketler kendi lisanslarını korur
([docs/SETUP.md](docs/SETUP.md)). `ultralytics` (AGPL-3.0) yalnız benchmark
referansıdır, asla runtime dependency değildir.
