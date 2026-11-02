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
6. Ek analiz için `python analysis/deep_analysis.py results/full_run` çalıştır.

## Sonuçlar (A100, tam test seti, n=1319)

| Koşul | Doğruluk | %95 GA | Ort. token |
|---|---|---|---|
| C1 Zero-shot direct | 70.9 | 68.4–73.3 | 217 |
| C2 Zero-shot CoT | **75.4** | 73.0–77.7 | 278 |
| C3 PAL | 59.7 | 57.0–62.4 | 110 |
| C4 LoRA + direct | 59.6 | 56.9–62.2 | 95 |

LoRA ince ayarı orijinal modelden 11.3 puan kötü sonuç verdi. McNemar testinde p < 1e-13.
Sebebi akıl yürütmenin kısalması: ince ayar modeli GSM8K'nın kısa çözüm stiline itti ve fark uzun problemlerde 7 puandan 21 puana çıkıyor.

## Rapor

`report_overleaf.zip` dosyasını Overleaf'te `New Project → Upload Project` ile yükle ve pdfLaTeX ile derle.

Tahmini süre A100'de 30–40 dakika, L4'te yaklaşık 1 saat, T4'te 1.5–2 saattir.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `YZM5109_A1_OptionC.ipynb` | Tek başına çalışan Colab notebook'u |
| `src/a1lib.py` | Promptlar, cevap çıkarma, PAL çalıştırıcı, LoRA eğitim döngüsü, istatistikler |
| `tests/smoke_test.py` | Ağ ve GPU gerektirmeden küçük rastgele bir modelle uçtan uca test |
| `build_notebook.py` | `a1lib.py` dosyasını notebook'a gömerek notebook'u yeniden üretir |
| `report/main.tex` | IEEE konferans formatında tamamlanmış rapor |
| `analysis/deep_analysis.py` | Ham tahminlerden aritmetik hata, problem uzunluğu ve tamamlayıcılık analizi |
| `results/full_run/` | Gerçek çalıştırmanın tüm çıktıları ve soru bazında tahminler |

`src/a1lib.py` değişirse `python build_notebook.py` çalıştırılarak notebook yeniden üretilir.
