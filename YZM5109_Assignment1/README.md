# YZM5109 Assignment I · Option C

**Soru:** Tek bir LoRA fine-tuning çalıştırması, orijinal modelde prompt mühendisliğini geçebilir mi?

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Veri:** `openai/gsm8k`, sonuç sayısı üzerinden exact-match.
- **Koşullar:** C1 zero-shot direct, C2 zero-shot CoT, C3 PAL, C4 LoRA (r=16, α=32) + C1 promptu. C5 4-shot isteğe bağlıdır.
- **Sabit tutulanlar:** aynı test seti, aynı sistem promptu, aynı cevap formatı, greedy decoding, `max_new_tokens=512`.
- **Raporlananlar:** doğruluk ve %95 bootstrap güven aralığı, ikili McNemar testleri, ortalama token, gecikme, eğitilebilir parametre sayısı, eğitim süresi, GPU belleği.

## Çalıştırma

1. `YZM5109_A1_OptionC.ipynb` dosyasını Colab'da aç. GitHub'dan aç ya da dosyayı yükle.
2. `Runtime → Change runtime type → T4 GPU` seç.
3. `QUICK_RUN = True` bırakıp `Run all` yap. Yaklaşık 5 dakika sürer ve her şeyin çalıştığını gösterir.
4. `QUICK_RUN = False` yapıp tekrar `Run all` yap. T4'te tam test setiyle tahmini süre 1–1.5 saattir. Zamanın azsa `N_TEST = 500` yap.
5. Colab bağlantıyı koparırsa yine `Run all` yap. Biten koşullar ve eğitilmiş adaptör Drive'dan yüklenir.
6. Sonda inen zip içinden `results_table.tex` ve `fig_*.pdf` dosyalarını `report/` klasörüyle birlikte Overleaf'e yükle.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `YZM5109_A1_OptionC.ipynb` | Tek başına çalışan Colab notebook'u |
| `src/a1lib.py` | Promptlar, cevap çıkarma, PAL çalıştırıcı, LoRA eğitim döngüsü, istatistikler |
| `tests/smoke_test.py` | Ağ ve GPU gerektirmeden küçük rastgele bir modelle uçtan uca test |
| `build_notebook.py` | `a1lib.py` dosyasını notebook'a gömerek notebook'u yeniden üretir |
| `report/main.tex` | IEEE konferans formatında rapor taslağı, `\todo{}` yerleri sonuçla doldurulur |

`src/a1lib.py` değişirse `python build_notebook.py` çalıştırılarak notebook yeniden üretilir.
