# YZM5109 Assignment I · Option C

**Soru:** Tek bir LoRA fine-tuning çalıştırması, orijinal modelde prompt mühendisliğini geçebilir mi?

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Veri:** `openai/gsm8k`, sonuç sayısı üzerinden exact-match.
- **Koşullar:** C1 zero-shot direct, C2 zero-shot CoT, C3 PAL, C4 LoRA (r=16, α=32) + C1 promptu. C5 4-shot isteğe bağlıdır.
- **Sabit tutulanlar:** aynı test seti, aynı sistem promptu, aynı cevap formatı, greedy decoding, `max_new_tokens=512`.
- **Raporlananlar:** doğruluk ve %95 bootstrap güven aralığı, ikili McNemar testleri, ortalama token, gecikme, eğitilebilir parametre sayısı, eğitim süresi, GPU belleği.

## Çalıştırma

1. Notebook'u Colab'da aç:
   https://colab.research.google.com/github/MeteHanYilmaz0/Data_Ai/blob/ccr-36b08569-obh9ku/YZM5109_Assignment1/YZM5109_A1_OptionC.ipynb
2. `Runtime → Change runtime type` menüsünden A100 seç. Yoksa L4, o da yoksa T4 seç.
3. `Runtime → Run all` de. Drive izni isterse ver. Varsayılan ayarlar gerçek deneydir: tam test seti, 300 LoRA adımı.
4. Colab bağlantıyı koparırsa yine `Run all` de. Biten koşullar ve eğitilmiş adaptör Drive'dan yüklenir.
5. Sonda inen küçük zip'i aç ve içindekileri `results/full_run/` klasörüne yükle.

Tahmini süre A100'de 30–40 dakika, L4'te yaklaşık 1 saat, T4'te 1.5–2 saattir.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `YZM5109_A1_OptionC.ipynb` | Tek başına çalışan Colab notebook'u |
| `src/a1lib.py` | Promptlar, cevap çıkarma, PAL çalıştırıcı, LoRA eğitim döngüsü, istatistikler |
| `tests/smoke_test.py` | Ağ ve GPU gerektirmeden küçük rastgele bir modelle uçtan uca test |
| `build_notebook.py` | `a1lib.py` dosyasını notebook'a gömerek notebook'u yeniden üretir |
| `report/main.tex` | IEEE konferans formatında rapor taslağı, `\todo{}` yerleri sonuçla doldurulur |

`src/a1lib.py` değişirse `python build_notebook.py` çalıştırılarak notebook yeniden üretilir.
