> **âš ï¸ GayriresmÃ® Ã§eviri.** Bu belge, PlumeRank deposunun Ä°ngilizce
> [`README.md`](README.md) dosyasÄ±nÄ±n TÃ¼rkÃ§e topluluk Ã§evirisidir ve
> [@attilayener](https://github.com/attilayener) tarafÄ±ndan hazÄ±rlanmÄ±ÅŸtÄ±r.
> Yetkili sÃ¼rÃ¼m her zaman Ä°ngilizce orijinaldir; sayÄ±lar, teknik ifadeler ve
> herhangi bir uyuÅŸmazlÄ±kta orijinal baÄŸlayÄ±cÄ±dÄ±r. Ã‡eviri, upstream'in
> `3c68bbb` commit'ine (10 Ekim 2026) dayanÄ±r. Telif ve lisans sahibi Edy Cu'dur
> (Apache License 2.0).

<div align="center">
  <img src="docs/assets/icon-animated.svg" width="144" alt="PlumeRank Simgesi">

  <h1>PlumeRank ğŸ”¥ğŸ“·</h1>
  <p><em>YangÄ±n kameralarÄ±ndan oluÅŸan bir duvar. Tek bir sevk gÃ¶revlisi. SÄ±radaki hangisi aÃ§Ä±lÄ±r?</em></p>
  <img src="docs/assets/readme-hero-animated.svg" alt="PlumeRank â€” yangÄ±n kameralarÄ±nÄ± kendi sakin gecesine karÅŸÄ±, kamera-gecesi baÅŸÄ±na 2 alarm bÃ¼tÃ§esinde sÄ±ralar; ÅŸematik duvar ve izleme listesi, elde tutulan sonuÃ§la birlikte 27/54 tespit, p50 38,0 dk" width="100%">

  <br/>

  [![CanlÄ± UÃ§ Nokta](https://img.shields.io/badge/ğŸŸ¢_Live-live.plumerank.edycu.dev-06b6d4?style=for-the-badge)](https://live.plumerank.edycu.dev)
  [![/judge](https://img.shields.io/badge/ğŸ‘©â€âš–ï¸_For-Judges-f59e0b?style=for-the-badge)](https://live.plumerank.edycu.dev/judge)
  [![Site](https://img.shields.io/badge/ğŸŒ_Site-plumerank.edycu.dev-FF6B4A?style=for-the-badge)](https://plumerank.edycu.dev)
  [![Pitch Sunumu](https://img.shields.io/badge/ğŸ“Š_Pitch-Deck-f59e0b?style=for-the-badge)](https://plumerank.edycu.dev/pitch/)
  [![Demo Video](https://img.shields.io/badge/ğŸ¬_Demo-Video-ef4444?style=for-the-badge)](https://youtu.be/QqX85JuNfVM)
  [![OpenCV AI YarÄ±ÅŸmasÄ± 2026 iÃ§in yapÄ±ldÄ±](https://img.shields.io/badge/OpenCV_AI_Competition-2026-8b5cf6?style=for-the-badge)](https://opencv26.devpost.com/)

  <br/>

  ![OpenCV 5](https://img.shields.io/badge/OpenCV-5.0.0-5C3EE8?style=flat&logo=opencv&logoColor=white)
  ![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
  ![FastAPI](https://img.shields.io/badge/FastAPI-0.142-009688?style=flat&logo=fastapi&logoColor=white)
  ![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat&logo=next.js)
  ![ONNX](https://img.shields.io/badge/ONNX-runtime%3A_cv2.dnn-005CED?style=flat&logo=onnx&logoColor=white)
  ![AWS Graviton](https://img.shields.io/badge/AWS-Graviton_t4g.small-FF9900?style=flat&logo=amazonaws&logoColor=white)
  ![Lisans: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat)
  [![CI](https://github.com/edycutjong/plumerank/actions/workflows/ci.yml/badge.svg)](https://github.com/edycutjong/plumerank/actions/workflows/ci.yml)
  [![CodeQL](https://github.com/edycutjong/plumerank/actions/workflows/codeql.yml/badge.svg)](https://github.com/edycutjong/plumerank/actions/workflows/codeql.yml)
  [![gitleaks](https://github.com/edycutjong/plumerank/actions/workflows/gitleaks.yml/badge.svg)](https://github.com/edycutjong/plumerank/actions/workflows/gitleaks.yml)

</div>

---

## Ã‡alÄ±ÅŸÄ±rken gÃ¶rÃ¼n

**Demo video (3:34):** https://youtu.be/QqX85JuNfVM â€” canlÄ± duvar, 1. sÄ±raya yÃ¼kselen bir yangÄ±n, elde tutulan sonuÃ§lar, mimari ve sÄ±nÄ±rlamalar.

<div align="center">
  <img src="docs/assets/live-wall.webp" alt="AWS Ã¼zerindeki canlÄ± PlumeRank duvarÄ±: 62 HPWREN kamera karosu ve istatistik baÅŸÄ±na puanlarÄ±yla sÄ±ralÄ± izleme listesi" width="100%">
</div>

> **Tek bir Graviton makinesindeki canlÄ± duvar.** HPWREN FIgLib gÃ¶rÃ¼ntÃ¼leri gerÃ§ek
> OpenCV 5 hattÄ±ndan yeniden oynatÄ±lÄ±r â†’ her kamera kendi son 15 dakikasÄ±na gÃ¶re
> puanlanÄ±r â†’ izleme listesi, kamera-penceresi baÅŸÄ±na 2 alarm bÃ¼tÃ§esinde, saniyede
> bir 62 kameranÄ±n tamamÄ±nÄ± yeniden sÄ±ralar.

---

## Temel soru

Saat 04:50, bir CAL FIRE acil durum komuta merkezinde gece vardiyasÄ±ndayÄ±z ve ayakta
olan tek sevk gÃ¶revlisi, Ã¶nÃ¼ndeki onlarca kamera karosuna bakÄ±yor ([ALERTCalifornia](https://alertcalifornia.org/)
tek baÅŸÄ±na 1.200'den fazla kamera iÅŸletiyor) â€” ve Ã¶nÃ¼mÃ¼zdeki altmÄ±ÅŸ saniyede hangi
dÃ¶rt kamerayÄ± aÃ§masÄ± gerektiÄŸini sÃ¶yleyen hiÃ§bir alet yok.

YayÄ±mlanmÄ±ÅŸ duman dedektÃ¶rleri **kare baÅŸÄ±na doÄŸruluk** (F1, kesinlik, duyarlÄ±lÄ±k) veya
yanlÄ±ÅŸ alarm sÄ±nÄ±rÄ± olmaksÄ±zÄ±n tespit sÃ¼resi Ã¼zerinden puanlanÄ±r. BunlarÄ±n hiÃ§biri ona
hangi karoyu aÃ§acaÄŸÄ±nÄ± sÃ¶ylemez; hiÃ§biri bir yangÄ±n kurumuna, sistemin gerÃ§ek bir
vardiyayla temas edip susturulmadan Ã¶nce ayakta kalÄ±p kalamayacaÄŸÄ±nÄ± sÃ¶ylemez. Ã–nemli
olan sayÄ± farklÄ±dÄ±r: **her kameraya yalnÄ±zca sabit sayÄ±da kesintiye katlanacaÄŸÄ±
gÃ¶z Ã¶nÃ¼ne alÄ±ndÄ±ÄŸÄ±nda, tutuÅŸmadan kaÃ§ dakika sonra bir insan gerÃ§ekten doÄŸru kameraya
bakÄ±yor?**

**PlumeRank bu sayÄ±yÄ± yayÄ±mlar.** Bir kamera aÄŸÄ±nÄ± izler ve her karede (dakikalÄ±k
gÃ¶rÃ¼ntÃ¼de) bir insanÄ±n ÅŸimdi hangi kameralara bakmasÄ± gerektiÄŸinin sÄ±ralÄ± kÄ±sa
listesini Ã¼retir â€” her satÄ±r kendi kÄ±rpÄ±lmÄ±ÅŸ kanÄ±t karosunu taÅŸÄ±r â€” **kamera-gecesi
baÅŸÄ±na 2 alarm**lÄ±k sabit bir bÃ¼tÃ§eye kalibre edilmiÅŸtir. HiÃ§bir kamerayÄ± asla
bastÄ±rmaz ve asla bir tim sevketmez. Ä°nsan dikkatini yeniden sÄ±ralar.

> **Birim.** Bu projede bir *kamera-gecesi* (camera-night) â€” aÃ§Ä±lÄ±ÅŸ sayfasÄ±nda
> *kamera-penceresi* (camera-window), Devpost aÃ§Ä±klamasÄ±nda ise *81-dakikalÄ±k pencere*
> olarak adlandÄ±rÄ±lÄ±r; bunun 12 saatlik bir gece gibi okunmasÄ±nÄ± engellemek iÃ§in â€”
> tek bir deÄŸerlendirme penceresidir: **bir kameradaki 81 dakikalÄ±k gÃ¶rÃ¼ntÃ¼**, bir
> FIgLib dizisinin uzunluÄŸu (`plumerank/constants.py`). DolayÄ±sÄ±yla bÃ¼tÃ§e, kamera
> baÅŸÄ±na 81 dakikalÄ±k pencere baÅŸÄ±na 2 alarmdÄ±r â€” kesintisiz Ã§alÄ±ÅŸtÄ±rÄ±ldÄ±ÄŸÄ±nda
> 12 saatlik bir vardiyada kamera baÅŸÄ±na toplam yaklaÅŸÄ±k 18 eder; gecede 2 deÄŸil.
> Ad kodundadÄ±r; pencereler gÃ¼ndÃ¼zdÃ¼r.

---

## GerÃ§ekte neyi Ã¶lÃ§Ã¼yor

> **BaÅŸlÄ±k sonucu â€” elde tutulan kÃ¼me, yalnÄ±zca bir kez okundu.** **Elde tutulan 54
> FIgLib yangÄ±nÄ±nda** (`data/manifest_hold.json` dosyasÄ±nda 60 tanesi, 6 tanesi
> `data/excluded.json` dosyasÄ±ndaki yayÄ±mlanmÄ±ÅŸ kural dÄ±ÅŸÄ±nda bÄ±rakÄ±ldÄ±) â€” hiÃ§bir
> sabitin Ã¼zerinde oturtulmadÄ±ÄŸÄ± dizilerde, sabitler dondurulduktan sonra tam olarak
> bir kez okundu (`results/constants.sha256`, `results/hold_access.log`) â€” PlumeRank
> **54'Ã¼n 27'sini** tespit etti; her kaÃ§Ä±rma "asla" sayÄ±ldÄ±ÄŸÄ±nda **medyan tespit
> sÃ¼resi 38,0 dakika** oldu; yangÄ±nsÄ±z karÄ±ÅŸÄ±klÄ±k kÃ¼mesi Ã¼zerinde
> (`results/bench_2026-10-10.json`) **kamera-gecesi baÅŸÄ±na 2 alarm** bÃ¼tÃ§esine
> kalibre edilmiÅŸ durumdadÄ±r. HiÃ§bir taban Ã§izgi medyana ulaÅŸamaz: PlumeRank'Ä±n
> kendi kamera-baÅŸÄ± taban Ã§izgisinden geÃ§irildiÄŸinde aynÄ± CNN 18, kare farkÄ± alma
> ise 20 tespit eder.
>
> **Medyan bir bÄ±Ã§aÄŸÄ±n sÄ±rtÄ±nda duruyor â€” ve bu sonucun bir parÃ§asÄ±.** Tam olarak
> yangÄ±nlarÄ±n yarÄ±sÄ± tespit edildiÄŸi iÃ§in var. TutuÅŸma saati âˆ’5â€¦0 dakika kaydÄ±rÄ±larak
> yeniden puanlanÄ±rsa 32â€“38 dk okunur; +1â€¦+5 dakika kaydÄ±rmada yarÄ±sÄ±ndan azÄ± tespit
> edilmiÅŸ sayÄ±lÄ±r ve medyan ortadan kalkar (`results/sensitivity_2026-10-10.csv`).
> Tespit sayÄ±larÄ± daha saÄŸlam karÅŸÄ±laÅŸtÄ±rmadÄ±r.

| Ã–lÃ§Ã¼m | **HOLD â€” baÅŸlÄ±k sonucu** (54 yangÄ±n) | DEV â€” ayar kolu, Ã¶rnek iÃ§i (29 yangÄ±n) |
|---|---|---|
| PlumeRank tespit | **27 / 54** | 17 / 29 |
| Medyan tespit sÃ¼resi (kaÃ§Ä±rma = asla) | **38,0 dk** | 32,0 dk |
| Tespit edilen yangÄ±nlarÄ±n ortalama tespit sÃ¼resi | 13,3 dk | 16,9 dk |
| Alarm oranÄ± | KarÄ±ÅŸÄ±klÄ±k kÃ¼mesinde kamera-gecesi baÅŸÄ±na 1,99 (hedef 2,0); HOLD'da gerÃ§ekleÅŸen 1,92 | KarÄ±ÅŸÄ±klÄ±k kÃ¼mesinde 1,99; DEV'de gerÃ§ekleÅŸen 2,37 |
| YÃ¼kseltme oranÄ± (ONNX sÄ±nÄ±flandÄ±rÄ±cÄ±sÄ±na gÃ¶nderilen kareler) | %1,76 â€” %2 sÄ±nÄ±rÄ±nÄ±n iÃ§inde | %2,14 â€” sÄ±nÄ±rÄ±n Ã¼zerinde |
| Kamera-gecesi baÅŸÄ±na CNN Ã§aÄŸrÄ±sÄ± | 1,4 (PlumeRank) karÅŸÄ±sÄ±nda 3.645 (B1) | 1,8 karÅŸÄ±sÄ±nda 3.645 |
| **AynÄ± kamera-baÅŸÄ± taban Ã§izgisinden geÃ§en B1 (birebir karÅŸÄ±laÅŸtÄ±rma)** | **18 / 54 â€” medyan yok** | 14 / 29 â€” medyan yok |
| **AynÄ± kamera-baÅŸÄ± taban Ã§izgisinden geÃ§en B2 (birebir karÅŸÄ±laÅŸtÄ±rma)** | **20 / 54 â€” medyan yok** | 10 / 29 â€” medyan yok |
| B1 konuÅŸlandÄ±rÄ±ldÄ±ÄŸÄ± haliyle (aynÄ± CNN, her karede) | 2 / 54 â€” medyan yok (eÅŸik 1,0'a 1e-6 iÃ§inde: doymuÅŸ olasÄ±lÄ±klar) | 6 / 29 â€” medyan yok |
| B2 konuÅŸlandÄ±rÄ±ldÄ±ÄŸÄ± haliyle (kare farkÄ± alma) | 4 / 54 â€” medyan yok | 3 / 29 â€” medyan yok |

Her sistem **aynÄ± 33 yangÄ±nsÄ±z pencere Ã¼zerinde** aynÄ± bÃ¼tÃ§eye kalibre edilmiÅŸtir
(beÅŸi de 1,99'a dÃ¼ÅŸer). HOLD koÅŸusu yerel bir arm64 makinede yÃ¼rÃ¼tÃ¼ldÃ¼
(`run.instance: local (not EC2)`, duvar saati 12dk29sn) â€” uÃ§ nokta makinesinde deÄŸil;
yani jÃ¼riye hizmet asla riske atÄ±lmadÄ±; `make fetch && make bench` bunu her yerde
yeniden Ã¼retir. Her sistemin dizibaÅŸÄ±na sonuÃ§larÄ± her dosyanÄ±n `per_sequence`
alanÄ±ndadÄ±r; [`REPORT.md`](REPORT.md) Â§6 DEV olanlarÄ± tablolaÅŸtÄ±rÄ±r, sitenin
gÃ¼nlÃ¼ÄŸÃ¼ elde tutulan 54 satÄ±rÄ±n tamamÄ±nÄ± listeler. CanlÄ± duvar aynÄ± donmuÅŸ eÅŸikle
(0,849) Ã§alÄ±ÅŸÄ±r; bu eÅŸik yalnÄ±zca karÄ±ÅŸÄ±klÄ±k kÃ¼mesine baÄŸlÄ±dÄ±r.

## DÃ¼rÃ¼st sÄ±nÄ±rlamalar

1. **YalnÄ±zca gÃ¼ndÃ¼z ve pencere boyutunda bir bÃ¼tÃ§e.** HPWREN FIgLib (gerÃ§ek veri
   kÃ¼mesi) neredeyse tamamen gÃ¼ndÃ¼zdÃ¼r (elde tutulan koldaki birkaÃ§ gece penceresi,
   belirtilen 6 hariÃ§ tutmanÄ±n 5'idir) ve bu yapÄ±mda IR kamera yoktur; gece vardiyasÄ±
   sevk gÃ¶revlisi bu sistemin kapsayamadÄ±ÄŸÄ± saatlerde Ã§alÄ±ÅŸÄ±r. BÃ¼tÃ§e 81 dakikalÄ±k
   pencere baÅŸÄ±na kalibre edilmiÅŸtir (yukarÄ±daki "Birim"e bakÄ±n); gerÃ§ek bir
   konuÅŸlandÄ±rma vardiya baÅŸÄ±na bÃ¼tÃ§e belirler ve eÅŸiÄŸi ona gÃ¶re yeniden Ã§Ã¶zerdi.
2. **DÃ¼ÅŸman bulut â€” bu projenin tasarlandÄ±ÄŸÄ± sis deÄŸil â€” ve istatistikler tasarÄ±mÄ±n
   varsaydÄ±ÄŸÄ±ndan zayÄ±f.** 2026-09-06'da, *taban Ã§izgi Ã¶ncesi* hat altÄ±nda Ã¶lÃ§Ã¼ldÃ¼
   (eÅŸik 0,722; gÃ¶nderilen hat altÄ±nda yeniden Ã¶lÃ§Ã¼lmedi): alÃ§ak bulut 4,25 ve bulut
   gÃ¶lgesi 3,86 alarm/kamera-gecesi maliyetindeyken sisinki 1,15'ti
   (`results/class_split_2026-09-06.json`) â€” ve kÃ¶r bir sayÄ±m, sis pencerelerini seÃ§en
   kuralÄ±n bir deniz katmanÄ±nÄ± rastgelelikten daha sÄ±k seÃ§mediÄŸini gÃ¶steriyor
   (eleme yapÄ±lmamÄ±ÅŸ ÅŸafak karelerinde %29'a karÅŸÄ± %29 taban oranÄ±;
   `results/fog_census_2026-10-04.md`); dolayÄ±sÄ±yla "sis" sÄ±nÄ±fÄ± Ã§oÄŸunlukla kapalÄ±
   hava. EÅŸik tÃ¼m karÄ±ÅŸÄ±klÄ±k pencereleri birlikte Ã¼zerinde Ã§Ã¶zÃ¼lÃ¼r, yani bu yalnÄ±zca
   sÄ±nÄ±f baÅŸÄ±na iddialarÄ± etkiler. Ä°statistik bazÄ±nda, bileÅŸeni olan DEV karelerinde
   tutuÅŸma Ã¶ncesi/sonrasÄ±: S3 (yaygÄ±n sÄ±nÄ±r) AUC 0,44 â€” ters-Ã¶ngÃ¶rÃ¼mlÃ¼ â€” S1 0,528 ve
   S2 0,50 ise ÅŸansa yakÄ±n (`results/statistics_confound_2026-09-06.md`). S3 hiÃ§bir
   zaman doÄŸrudan sise karÅŸÄ± test edilmedi. SÄ±ralamayÄ± taÅŸÄ±yan ÅŸey herhangi bir tek
   istatistik deÄŸil, her kameranÄ±n kendi izleyen 15 dakikasÄ±na gÃ¶re puanlanmasÄ±dÄ±r.
3. **Ä°ki birleÅŸim aÄŸÄ±rlÄ±ÄŸÄ± tasarÄ±mÄ±n varsaydÄ±ÄŸÄ± gibi deÄŸil.** S3, tek baÅŸÄ±na AUC'si
   0,44 olmasÄ±na raÄŸmen en bÃ¼yÃ¼k birleÅŸim aÄŸÄ±rlÄ±ÄŸÄ±nÄ± taÅŸÄ±yor
   (`FUSION_W_S3 = 4.0931`, `plumerank/constants.py`) ve henÃ¼z S3-kapalÄ± ablasyonu
   koÅŸulmadÄ±. AyrÄ±ca tek bir aÄŸÄ±rlÄ±k tanÄ±mlÄ± deÄŸil: yeniden oturtma,
   `FUSION_W_ESCALATION`'Ä± dizibaÅŸÄ±na-bÄ±rak AUC'de 0,0006'lÄ±k bir deÄŸiÅŸim iÃ§in +0,78'den
   âˆ’0,38'e taÅŸÄ±yor; gÃ¶nderilen deÄŸer, DEV'de iki diziyi ve p50'nin altÄ± dakikasÄ±nÄ± daha
   iyi puanladÄ±ÄŸÄ± iÃ§in tutuluyor (`results/refit_through_baseline_2026-09-16.md`).
   DEV ayar koludur, yani bu seÃ§im yapÄ± gereÄŸi Ã¶rnek iÃ§idir. Taban Ã§izgi penceresi de
   Ã¶yle: 10 dakika 18/29 (p50 30,0) puanlarken gÃ¶nderilen 15 dakika 17/29 (p50 32,0) â€”
   bir dizi fark, gÃ¼rÃ¼ltÃ¼ iÃ§inde, ayarlanmak yerine tasarÄ±m tercihi olarak tutuldu
   (`results/baseline_calibrated_2026-09-16.md`).
4. **Yenilik bir metrik yeniden Ã§erÃ§eveleme, yeni bir sensÃ¶r deÄŸil.** Tek tek OpenCV
   iÅŸlemleri iyi bilinir; yeni olan onlarÄ± birleÅŸtiren ilkel ve yÃ¶neltildikleri
   dikkat-bÃ¼tÃ§esi hedefidir.
5. **Tek Ã¶rnek, HA yok, statik yansÄ±ma yok.** UÃ§ nokta tek bir Graviton `t4g.small`
   (sÄ±nÄ±rsÄ±z kredi modu), yarÄ±ÅŸmanÄ±n AWS bursuyla finanse ediliyor. YayÄ±mlanan her
   sayÄ± `results/` iÃ§inde commit'lenmiÅŸtir; iddia, makine kapansa da ayakta kalÄ±r.
6. **YÃ¶netilen bulut gÃ¶rÃ¼ iÅŸlemleri API taban Ã§izgisi yok.** Ä°ki taban Ã§izgi de
   karÄ±ÅŸÄ±klÄ±k kÃ¼mesi Ã¼zerinde PlumeRank ile aynÄ± alarm bÃ¼tÃ§esine kalibre edilir;
   bunlardan biri (B1) PlumeRank'Ä±n kendi sinir aÄŸÄ±nÄ± paylaÅŸÄ±r â€” klasik aÅŸamanÄ±n
   yÃ¼kselttiÄŸi ~%2 yerine her karede Ã§alÄ±ÅŸtÄ±rÄ±lÄ±r â€” bu farklÄ± boyutlu modeller
   yarÄ±ÅŸÄ± deÄŸil, kontrollÃ¼ bir deneydir.
7. **YÃ¼kleme yolu yok, sevk yolu yok, bastÄ±rma yolu yok â€” tasarÄ±m gereÄŸi, unutma
   sonucu deÄŸil.** AÅŸaÄŸÄ±daki "Sorumlu kullanÄ±m"a bakÄ±n.
8. **SÄ±nÄ±flandÄ±rÄ±cÄ± yÃ¼kseltmesi DEV'de bir tespit kazandÄ±rmÄ±yor ve gÃ¶nderilen hÃ¼cre,
   sayÄ±m bakÄ±mÄ±ndan en iyi DEV hÃ¼cresi deÄŸil.** On iki hÃ¼creli Ä±zgara
   (`results/ablation_2026-10-10.csv`, {MOG2, LSBP, GSOC} Ã— S4 aÃ§Ä±k/kapalÄ± Ã—
   yÃ¼kseltme aÃ§Ä±k/kapalÄ±, hepsi 2,0 alarm/kamera-gecesi bÃ¼tÃ§esinde gÃ¶nderilen
   kamera-baÅŸÄ± taban Ã§izgisinden geÃ§er) yÃ¼kseltme aÃ§Ä±kken ve kapalÄ±ken altÄ± Ã§iftin
   beÅŸinde aynÄ± sayÄ±da yangÄ±n tespit ettiÄŸini, altÄ±ncÄ±da aÃ§Ä±kken bir eksik
   tespit ettiÄŸini gÃ¶sterir. SayÄ±mÄ± oynatan bileÅŸen S4'tÃ¼r (MOG2: 17 varken,
   yokken 14). S4 kapalÄ± GSOC, gÃ¶nderilen hÃ¼crenin 17/29 ve p50 32,0'sÄ±na karÅŸÄ±
   18/29 tespit eder; p50 33,0. DEV ayar koludur; yani bunlarÄ±n hiÃ§biri donmuÅŸ
   yapÄ±landÄ±rmayÄ± yeniden aÃ§madÄ±: elde tutulan koÅŸu gÃ¶nderilen hÃ¼creyi Ã¶lÃ§tÃ¼,
   yÃ¼kseltme dahil â€” Ã§Ã¼nkÃ¼ `constants.py` bu Ä±zgara koÅŸulmadan Ã¶nce sabitlendi
   (2026-10-04 DEV koÅŸusundan beri hash `02752e86â€¦`) ve Ä±zgara onu deÄŸiÅŸtirmedi;
   dondurma dosyasÄ± Ä±zgaradan sonra, elde tutulan okumadan Ã¶nce commit'lendi.
   DEV'de sÄ±nÄ±flandÄ±rÄ±cÄ±ya yardÄ±mcÄ± olduÄŸu gÃ¶sterilemiyor.
9. **Stabilizasyon her kareyi ham Ã¶ncÃ¼lÃ¼ne kaydeder ve hareket eden bulut altÄ±nda
   tahmini kamerayÄ± deÄŸil gÃ¶kyÃ¼zÃ¼nÃ¼ izler.** `stabilise(prev, cur)`, *t* karesini
   ham *tâˆ’1* karesine hizalar; yani gerÃ§ek kamera hareketi arka plan modeline bir
   kare gecikmeyle hÃ¢lÃ¢ ulaÅŸÄ±r (`api/main.py` `Replay._tick`, `scripts/bench.py`).
   StabilizatÃ¶rÃ¼n â‰¥ 1 px bÃ¼kÃ¼m uyguladÄ±ÄŸÄ± her Ã§ift zemine karÅŸÄ± hakem edildi â€”
   karenin satÄ±r %55â€“88'i; zeminin hizasÄ±nÄ± iyileÅŸtiren bÃ¼kÃ¼m kameranÄ±nkidir,
   kÃ¶tÃ¼leÅŸtiren ise deÄŸildir (`scripts/measure_registration.py`,
   `results/registration_2026-10-10.json`). DEV'de 55'in 48'i gerÃ§ek harekettir,
   hepsi tek kameradan (`20250107_PalisadesFire_69bravo-e-mobo-c`, dakikada 5,7
   px'e kadar) ve bÃ¼kÃ¼m hepsinde zemin hizasÄ±nÄ± iyileÅŸtirir. DEV'deki diÄŸer 7 ve
   CONFUSE'daki 39 deÄŸil: zemin medyan 0,07 px (DEV) / 0,04 px (CONFUSE) hareket
   ederken gÃ¶kyÃ¼zÃ¼ bandÄ± 3,3â€“14,8 px hareket etti (pencere baÅŸÄ±na medyanlar) ve
   tam-kare faz korelasyonu ona doÄŸru Ã§ekildi. EÅŸkin Ã¼zerinde Ã§Ã¶zÃ¼ldÃ¼ÄŸÃ¼ yangÄ±nsÄ±z
   kÃ¼mede â‰¥ 1 px'lik her bÃ¼kÃ¼m bu yÃ¼zden duraÄŸÄ±n bir sÄ±rt Ã§izgisini kaydÄ±rÄ±r â€” ve
   maliyeti azdÄ±r, karÅŸÄ±olgusal olarak Ã¶lÃ§Ã¼ldÃ¼: CONFUSE'u commit'lenen koÅŸunun
   yaptÄ±ÄŸÄ± gibi yeniden oynatmak (kamera-gecesi baÅŸÄ±na 1,9885 alarmÄ±nÄ± ve eÅŸiÄŸini
   tam olarak yeniden Ã¼retmek) ve ardÄ±ndan o 39 bÃ¼kÃ¼m uygulanmadan tekrar oynatmak,
   eÅŸiksiz eÅŸik aÅŸmalarÄ±nÄ± 64'ten 63'e, 2,0 bÃ¼tÃ§esinin Ã§Ã¶zdÃ¼ÄŸÃ¼ eÅŸiÄŸi 0,8492'den
   0,8482'ye deÄŸiÅŸtirir (`scripts/measure_registration_alerts.py`,
   `results/registration_alerts_2026-10-10.json`). Etki bÃ¼kÃ¼len kareler yerine
   bÃ¼tÃ¼n pencerelere yayÄ±lÄ±r â€” bir pencerede bÃ¼kÃ¼mleri kaldÄ±rmak bir aÅŸma ekler â€”
   yani kare kare sayÄ±lmaz; yangÄ±n tespitine etkisi Ã¶lÃ§Ã¼lmedi. DeÄŸiÅŸmeyen:
   elde tutulan okuma bu hattÄ± Ã¶lÃ§tÃ¼; kaymayÄ± yalnÄ±zca zemin bandÄ±nda tahmin etmek
   gelecek Ã§alÄ±ÅŸmanÄ±n ilk maddesidir.
10. **KanÄ±t kÄ±rpÄ±mÄ± dumanÄ± zamanÄ±n yarÄ±sÄ±ndan azÄ±nda gÃ¶sterir.** Her satÄ±rÄ±n
    kÄ±rpÄ±mÄ± kameranÄ±n izlenen en bÃ¼yÃ¼k Ã¶n plan bileÅŸenidir ve sÄ±ralama kameraya
    gÃ¶redir â€” yani bir yangÄ±n kamerasÄ±, en bÃ¼yÃ¼k hareket eden ÅŸeyi bir bulut
    olduÄŸunda bile tepreye ulaÅŸabilir. Her DEV yangÄ±nÄ±nÄ±n tutuÅŸma sonrasÄ± ilk
    1. sÄ±rada veya alarm tikinde, canlÄ± duvarÄ±n bir turunda, kÄ±rpÄ±m bÃ¶lgesi 21'in
    9'unda dumanÄ±; 7'sinde bulut, 2'sinde gÃ¼neÅŸ veya Ã§Ä±plak gÃ¶kyÃ¼zÃ¼, 2'sinde
    arazi, 1'inde belirsizi iÃ§erir (`scripts/audit_evidence_crops.py`,
    `results/evidence_first_hits_2026-10-10.json` ve tam Ã§Ã¶zÃ¼nÃ¼rlÃ¼klÃ¼ temas
    sayfasÄ±; etiketler gÃ¶zle, `results/evidence_first_hits_labels.json` iÃ§inde).
    Sevk gÃ¶revlisi kÄ±rpÄ±mdaki bulutu "tÃ¼m karoya bak" olarak okumalÄ±dÄ±r, "temiz"
    olarak deÄŸil. GÃ¶sterilecek bileÅŸeni boyuta gÃ¶re deÄŸil istatistiÄŸe gÃ¶re seÃ§mek
    gelecek Ã§alÄ±ÅŸmadÄ±r; hiÃ§bir sÄ±ralamayÄ± deÄŸiÅŸtirmez.

---

## NasÄ±l Ã§alÄ±ÅŸÄ±r

Yuvarlanan, stabilize edilmiÅŸ bir pencere Ã¼zerinde, her kamera Ã¼zerindeki her Ã¶n plan
bileÅŸeni ÅŸunlar iÃ§in Ã¶lÃ§Ã¼lÃ¼r:

| # | Ä°statistik | TasarÄ±m hipotezi | Tek baÅŸÄ±na Ã¶lÃ§Ã¼len (DEV, tutuÅŸma Ã¶ncesi/sonrasÄ± AUC) |
|---|---|---|---|
| S1 | **Tabana baÄŸlanma** | bir yangÄ±n zemine baÄŸlÄ±dÄ±r; bulut deÄŸildir | 0,528 â€” ÅŸansa yakÄ±n |
| S2 | **Dikey bÃ¼yÃ¼me anizotropisi** | bir duman sÃ¼tunu yayÄ±ldÄ±ÄŸÄ±ndan hÄ±zlÄ± yÃ¼kselir | 0,50 â€” ÅŸans |
| S3 | **YaygÄ±n-sÄ±nÄ±r kontrastÄ±** | dumanÄ±n kenarÄ± yumuÅŸaktÄ±r | 0,44 â€” ters-Ã¶ngÃ¶rÃ¼mlÃ¼ (sÄ±nÄ±rlama 2 ve 3) |
| S4 | **Kutu iÃ§i optik akÄ±ÅŸ sÃ¼rÃ¼klenmesi** | bir duman tutarlÄ± sÃ¼rÃ¼klenir; sensÃ¶r gÃ¼rÃ¼ltÃ¼sÃ¼ sÃ¼rÃ¼klenmez | 0,538 â€” zayÄ±f; tespit kanÄ±tÄ± Ã¼zerine tutulur: gÃ¶nderilen taban Ã§izgisi altÄ±nda varken 17/29, yokken 14/29 â€” yine de S4-kapalÄ± koÅŸu Ã§ok daha az kare yÃ¼kseltir (aynÄ± Ä±zgarada %0,17'ye karÅŸÄ± %2,14; o Ä±zgaranÄ±n gÃ¶nderilen hÃ¼cre yeniden koÅŸusu, aynÄ± sabit hash'i ve eÅŸiÄŸinde 2026-10-04 koÅŸusunun 51'ine karÅŸÄ± 49 kare yÃ¼kseltti, tespitler Ã¶zdeÅŸ), yani iki etki ayrÄ±ÅŸtÄ±rÄ±lmamÄ±ÅŸtÄ±r |

DÃ¶rdÃ¼nÃ¼n hiÃ§biri tek baÅŸÄ±na iyi bir dedektÃ¶r deÄŸildir ve kare baÅŸÄ±na, birleÅŸmiÅŸ
model "bir bileÅŸen var" tek bitini Ã¶rnek iÃ§inde yalnÄ±zca 0,0015 AUC ile geÃ§er
(`results/statistics_confound_2026-09-06.md`). Tespiti kaldÄ±ran aÅŸaÄŸÄ±daki
kamera-baÅŸÄ± taban Ã§izgisidir, kare-baÅŸÄ± puan deÄŸil.

Bu dÃ¶rt, kamera baÅŸÄ±na bir tehlikeye katlanÄ±r (sabit bir lojistik, aÄŸÄ±rlÄ±klar DEV
Ã¼zerinde oturtulmuÅŸ, varlÄ±k terimi yok). ArdÄ±ndan her kamera **kendi izleyen 15
dakikasÄ±na** gÃ¶re puanlanÄ±r â€” bÃ¼tÃ¼n gece sÄ±cak Ã§alÄ±ÅŸan bir kamera yangÄ±n deÄŸildir â€”
ve bundan sonra bir Ã§Ã¶zÃ¼cÃ¼, alarm eÅŸiÄŸini bÃ¼tÃ§eye kalibre eder.

**Ã–ÄŸrenilen bileÅŸen hiÃ§bir Kaliforniya pikseli gÃ¶rmez.** Tek eÄŸitilmiÅŸ model â€”
`cv2.dnn` Ã¼zerinden eriÅŸilen kÃ¼Ã§Ã¼k bir ONNX karo sÄ±nÄ±flandÄ±rÄ±cÄ±sÄ± â€” yalnÄ±zca
Pyro-SDIS (FransÄ±z yangÄ±n servisleri, Apache-2.0) Ã¼zerinde eÄŸitilmiÅŸtir ve
yalnÄ±zca klasik aÅŸamanÄ±n yÃ¼kselttiÄŸi karelerin ~%2'sine dokunur. AynÄ± model taban
Ã§izgi B1'in ta kendisidir â€” %2 yerine her karede Ã§alÄ±ÅŸtÄ±rÄ±lÄ±r: PlumeRank bu
karÅŸÄ±laÅŸtÄ±rmayÄ± daha bÃ¼yÃ¼k bir modelle kazanamaz, Ã§Ã¼nkÃ¼ model iki tarafta da
Ã¶zdeÅŸtir.

### Neden Ã¶zellikle OpenCV 5

**YarÄ±ÅŸma gereÄŸi**, ve sÃ¼s deÄŸil yapÄ±sal: 6 modÃ¼lde 23 savunulan OpenCV 5 yÃ¼zeyi â€”
18 analiz Ã§aÄŸrÄ±sÄ± artÄ± 5 Ã§alÄ±ÅŸma zamanÄ±/introspeksiyon yÃ¼zeyi (iÅŸ parÃ§acÄ±ÄŸÄ± sabitleme,
zamanlama, yapÄ± bilgisi ve iki motor/tahminci bayraÄŸÄ±) â€” her `make test` koÅŸusunda
kaynak Ã¼zerinden mekanik olarak sayÄ±lÄ±r `scripts/check_claims.py` (asla elle
yazÄ±lmaz):

| YÃ¼zey | BulunduÄŸu yer | OlmasaydÄ± |
|---|---|---|
| `createBackgroundSubtractorMOG2` + `.apply` | `plumerank/segment.py` | makale baÅŸÄ±na yeniden yazÄ±lacak, piksel baÅŸÄ±na Gauss karÄ±ÅŸÄ±mÄ± arka plan modeli |
| `phaseCorrelate` + `createHanningWindow` (+ OpenCV 4.13 / 5.0'da eklenen `phaseCorrelateIterative`) | `plumerank/stabilise.py` | alt-piksel iyileÅŸtirmeli, dolgulu DFT Ã§ap-gÃ¼Ã§-spektrumu stabilize edici |
| `estimateAffine2D(..., USAC_MAGSAC)` | `plumerank/stabilise.py` | doÄŸru bir MAGSAC++ uygulamasÄ± |
| `connectedComponentsWithStats`, `Sobel`, `Laplacian` | `plumerank/segment.py`, `plumerank/statistic.py` | S1/S2'nin okuduÄŸu bileÅŸen kutularÄ± ve S3'Ã¼n okuduÄŸu sÄ±nÄ±r enerjileri |
| `DISOpticalFlow_create` | `plumerank/statistic.py` (S4) | kullanÄ±labilir hÄ±zda yoÄŸun bir akÄ±ÅŸ alanÄ± |
| `dnn.readNetFromONNX` (`dnn.ENGINE_NEW`) + `dnn.blobFromImage` + `dnn.Net.forward` | `plumerank/escalate.py` | arm64 makinesinde ikinci bir Ã§Ä±karÄ±m Ã§alÄ±ÅŸma zamanÄ± |
| `getBuildInformation` | `api/main.py` â†’ `GET /build-info` | bir jÃ¼rinin OpenCV 5 iddiasÄ±nÄ± doÄŸrulamak iÃ§in `curl`'leyebileceÄŸi bir ÅŸey |

Kalan yÃ¼zeyler (ablasyon Ä±zgarasÄ± iÃ§in tutulan iki arka plan Ã§Ä±karÄ±cÄ± varyantÄ±,
CLAHE kontrast normalleÅŸtirmesi, morfoloji temizliÄŸi, yapÄ±/zamanlama
introspeksiyonu) `scripts/check_claims.py` Ã§Ä±ktÄ±sÄ±nda eksiksiz listelenir.

### Mimari

```
HPWREN FIgLib yeniden oynatma â”€â”€â–¶ stabilize et (phaseCorrelate + MAGSAC) â”€â”€â–¶ bÃ¶lÃ¼tle (MOG2 + baÄŸlÄ± bileÅŸenler)
                                                                        â”‚
                                                            S1Â·S2Â·S3Â·S4 istatistikleri
                                                                        â”‚
                                               yÃ¼kselt (karelerin ~%2'si) â”€â”€â–¶ ONNX karo sÄ±nÄ±flandÄ±rÄ±cÄ±sÄ± (cv2.dnn)
                                                                        â”‚
                                                      birleÅŸtir â”€â”€â–¶ kamera-baÅŸÄ± taban Ã§izgi (izleyen 15 dk)
                                                                        â”‚
                                              eÅŸik Ã§Ã¶z (alarm bÃ¼tÃ§esi) â”€â”€â–¶ yayÄ±mla (sÄ±ralÄ± izleme listesi)
                                                                        â”‚
                                   FastAPI (/api/watchlist SSE, /api/benchmark, /health, /build-info)
                                                                        â”‚
                                                        Next.js â€” Duvar, /results, /judge
```

Tek t4g.small (Graviton/ARM) Ã¶rneÄŸi, kÃ¼me iÃ§in tek S3 kovasÄ±, alarm bÃ¼tÃ§esi iÃ§in
CloudWatch (duvarÄ±n yangÄ±nsÄ±z kameralarÄ±ndaki oran; Â±%15 sapma alarmlarÄ±yla â€” duvar
bu pencereleri kÄ±yaslamanÄ±n yaptÄ±ÄŸÄ± gibi aynen yeniden oynatÄ±r, yani saÄŸlÄ±klÄ± bir
makine kalibre edilmiÅŸ 1,99'u okur) ve servis atmayÄ± kestiÄŸinde Ã¶rneÄŸi yeniden
baÅŸlatan bir kalp-atÄ±ÅŸÄ±-sessizlik alarmÄ±. Lambda yok, yÃ¶netilen gÃ¶rÃ¼ API'si yok,
ikinci Ã¶rnek yok.

---

## Sorumlu kullanÄ±m

1. **Ä°nsan iÃ§in sÄ±ralama yardÄ±mÄ±, sevk deÄŸil.** Bu kod tabanÄ±nda hiÃ§bir yerde sevk
   yolu yoktur (`tests/test_api.py::test_there_is_no_upload_route` ve eÅŸdeÄŸer bir
   sevk yolu da yoktur â€” `test_exactly_the_four_documented_routes_exist`).
2. **HiÃ§bir kamerayÄ± asla gizlemez.** Kadronin tamamÄ± ekranda kalÄ±r; sistem yalnÄ±zca
   sÄ±rasÄ±nÄ± deÄŸiÅŸtirir.
3. **Alarm bÃ¼tÃ§esi yayÄ±mlanmÄ±ÅŸ, gÃ¼nlÃ¼ÄŸe yazÄ±lan, ayarlanabilir bir parametredir** â€”
   bir yapÄ±landÄ±rma dosyasÄ±na gÃ¶mÃ¼lÃ¼ bir sabit deÄŸil, sapma alarmÄ± olan bir CloudWatch
   metriÄŸidir.
4. **Her sÄ±ralÄ± satÄ±r kanÄ±t kÄ±rpÄ±mÄ±nÄ± da taÅŸÄ±r**, yani bir insan sÄ±ralamayÄ± tek
   bakÄ±ÅŸta geÃ§ersiz kÄ±labilir.
5. **Yeni gÃ¶rÃ¼ntÃ¼ toplanmaz.** Duvar manzaralarÄ±n kamuya aÃ§Ä±k HPWREN FIgLib
   gÃ¶rÃ¼ntÃ¼lerini yeniden oynatÄ±r; yazdÄ±ÄŸÄ± tek kareler, ekran iÃ§in o gÃ¶rÃ¼ntÃ¼nÃ¼n kÃ¼Ã§Ã¼k
   kanÄ±t kÄ±rpÄ±mlarÄ± ve kÃ¼Ã§Ã¼k resimleridir ve her yeniden oynatmada Ã¼zerine yazÄ±lÄ±r.

---

## Test

```
Yerel olarak 832 geÃ§ti, 1 atlandÄ± (kÃ¼me olmadan CI, 820 koÅŸar ve 13'Ã¼ atlar) Â· plumerank/ Ã¼zerinde %100 satÄ±r kapsamÄ± Â· ruff temiz Â· mypy temiz
```

| Katman | AraÃ§ | Notlar |
|---|---|---|
| Birim + entegrasyon | `pytest` | `make test` â€” Ã§evrimdÄ±ÅŸÄ±, aÄŸ yok, kimlik bilgisi yok, ~2 dk |
| Ã–zellik tabanlÄ± | `hypothesis` | `tests/test_invariants.py` â€” **1.500 Ã¼retilmiÅŸ vaka** (her biri 500) Ã¶zellik testi: birleÅŸmiÅŸ tehlikenin geÃ§erli girdilerde asla `[0, 1]` dÄ±ÅŸÄ±na Ã§Ä±kmamasÄ± ve geÃ§ersiz bir tanesini sessizce geÃ§memesi; Ã¼retilmiÅŸ Ã§aÄŸrÄ± dizileri boyunca alarm bÃ¼tÃ§esinin asla aÅŸilmamasÄ± |
| Regresyon | 690+ Ã¶rnek test, Ã§oÄŸu sabitlediÄŸi belirli davranÄ±ÅŸ veya hatanÄ±n adÄ±nÄ± taÅŸÄ±r (Ã¶r. `test_an_unusable_threshold_degrades_the_wall_instead_of_500ing_health`) | `tests/` |
| Ä°ddia/kod tutarlÄ±lÄ±ÄŸÄ± | `scripts/check_claims.py` (`make test` iÃ§inde koÅŸar) | bu belgenin adÄ±nÄ± andÄ±ÄŸÄ± her OpenCV sembolÃ¼, API rotasÄ±, baÄŸÄ±mlÄ±lÄ±k, Makefile hedefi ve dosya yolu gerÃ§ek kaynakla mekanik olarak doÄŸrulanÄ±r â€” taÅŸÄ±nan bir dosya, yeniden adlandÄ±rÄ±lan bir sembol veya silinen bir rota bir sonraki test koÅŸusunu dÃ¼ÅŸÃ¼rÃ¼r (metne elle yazÄ±lan sayÄ±lar, Ã¶rn. test toplamÄ±, ayrÄ±ÅŸtÄ±rÄ±lmaz) |
| Statik analiz | `ruff check .`, `mypy plumerank api --ignore-missing-imports` | her ikisi de temiz |
| SÄ±r taramasÄ± | `gitleaks` (tam geÃ§miÅŸ, 0 bulgu) + GitHub sÄ±r taramasÄ± ve itme korumasÄ± (etkin) | CI + depo ayarlarÄ± |
| BaÄŸÄ±mlÄ±lÄ±k denetimi | `pip-audit`, `npm audit` | CI, danÄ±ÅŸma amaÃ§lÄ± |
| SAST | CodeQL (`python`, `javascript-typescript`) | CI, her itmede |
| E2E | Playwright, **gerÃ§ek** FastAPI arka ucuna (sahte yok) | `web/e2e/` |

```bash
make test              # Ã§evrimdÄ±ÅŸÄ± paket + kapsam, ~2 dk, aÄŸ yok, kimlik bilgisi yok
make verify             # â‰¤3 dk: .venv'i yeniden kurar, Ã§evrimdÄ±ÅŸÄ± paketi koÅŸar, ardÄ±ndan
                         # mikro-kÃ¼me Ã¶lÃ§Ã¼mlerini yeniden hesaplar ve commit'lenmiÅŸ
                         # results/verify_micro.json ile karÅŸÄ±laÅŸtÄ±rÄ±r. YayÄ±mlanan baÅŸlÄ±k
                         # sonucunu YENÄ°DEN ÃœRETMEZ ve asla Ã¼rettiÄŸini iddia etmez.
make fetch && make bench  # toplam ~95 dk: elde tutulan baÅŸlÄ±k sonucunu yeniden Ã¼reten Ã§ift
                            # (results/bench_2026-10-10.json, commit'lenmiÅŸ)
```

Tam yeniden Ã¼retme yÃ¼rÃ¼yÃ¼ÅŸÃ¼ iÃ§in [`DEMO.md`](DEMO.md), 30 saniyelik yol iÃ§in
[`JUDGE.md`](JUDGE.md) dosyasÄ±na bakÄ±n.

---

## BaÅŸlangÄ±Ã§

### Arka uÃ§ (kÃ¶k)
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.lock
.venv/bin/pip install -e ".[dev]"
.venv/bin/uvicorn api.main:app --reload   # http://127.0.0.1:8000
```

### Ã–n uÃ§ (`web/`)
```bash
cd web && npm install
cp .env.example .env.local   # NEXT_PUBLIC_PLUMERANK_API â€” bkz. web/.env.example
npm run dev                  # http://127.0.0.1:3000
```

Bu yÄ±ÄŸÄ±nda hiÃ§bir yerde API anahtarÄ±, hesap veya oturum aÃ§ma yoktur.

---

## Proje yapÄ±sÄ±

```
plumerank/       # hat: stabilize et, bÃ¶lÃ¼tle, istatistik, yÃ¼kselt, sÄ±rala, Ã¶lÃ§Ã¼mler, tipler
api/             # FastAPI uygulamasÄ± â€” /build-info, /health, /api/watchlist, /api/benchmark
scripts/         # fetch.py, bench.py, check_claims.py, check_submission_readiness.py
web/             # Next.js Ã¶n yÃ¼z â€” Duvar ("/"), "/results", "/judge"
tests/           # 833 test, plumerank/ Ã¼zerinde %100 satÄ±r kapsamÄ±
deploy/          # EC2 Ã¶nyÃ¼kleme (Caddy, systemd, OpenCV 5 yapÄ± sabitlemesi)
results/         # commit'lenmiÅŸ koÅŸu yapÄ±tlarÄ± â€” yayÄ±mlanan her sayÄ± burada yaÅŸar
```

---

## Neyi yanlÄ±ÅŸ yaptÄ±k (tarihli)

KayÄ±ttan yazÄ±ldÄ±, hafÄ±zadan deÄŸil; her satÄ±r yakalayan dosyayÄ± adlandÄ±rÄ±r.

- **2026-09-06: baÅŸlÄ±ÄŸÄ±mÄ±zÄ± biz yanlÄ±ÅŸ aktardÄ±k.** DÃ¶rt belge gÃ¶nderilen DEV sonucu
  olarak 12/29'u aktardÄ±; o, sonuÃ§ dosyasÄ± dÃ¼z `arm_dev_<tarih>.json` adÄ±nÄ± almÄ±ÅŸ
  S4-kapalÄ± ablasyon hÃ¼cresiydi. GÃ¶nderilen yapÄ±landÄ±rma o gÃ¼n 9/29 Ã¶lÃ§tÃ¼. SonuÃ§
  dosya adlarÄ± artÄ±k S4, taban Ã§izgi, arka plan ve bÃ¼tÃ§e hÃ¼crelerini adlandÄ±rÄ±yor
  (`scripts/bench.py` `results_filename`; S4 2026-09-06'da
  `results/artefact_provenance_2026-09-06.md` Â§7 uyarÄ±nca, diÄŸer Ã¼Ã§Ã¼ 2026-10-04'te
  `c977858` iÃ§inde dÃ¼zeltildi).
- **TasarÄ±m Ã¶ncÃ¼lÃ¼ sisti; dÃ¼ÅŸman bulut.** AlÃ§ak bulut ve bulut gÃ¶lgesi, sisin 1,15'ine
  karÅŸÄ± 4,25 ve 3,86 alarm/kamera-gecesi'ne mal oldu (taban Ã§izgi Ã¶ncesi eÅŸik,
  `results/class_split_2026-09-06.json`) ve kÃ¶r bir sayÄ±m, sis pencerelerini seÃ§en
  kuralÄ±n bir deniz katmanÄ±nÄ± rastgelelik sÄ±klÄ±ÄŸÄ±nda seÃ§tiÄŸini buldu
  (`results/fog_census_2026-10-04.md`).
- **TasarÄ±mÄ±n dayandÄ±ÄŸÄ± istatistik tek baÅŸÄ±na ters-Ã¶ngÃ¶rÃ¼mlÃ¼.** S3, yaygÄ±n-sÄ±nÄ±r
  kontrastÄ±, bileÅŸeni olan DEV karelerinde AUC 0,44 skorluyor
  (`results/statistics_confound_2026-09-06.md`). SÄ±ralamayÄ± taÅŸÄ±yan her kameranÄ±n
  kendi taban Ã§izgisidir.
- **SÄ±nÄ±flandÄ±rÄ±cÄ± yÃ¼kseltmesi DEV tespiti kazanmÄ±yor** (sÄ±nÄ±rlama 8,
  `results/ablation_2026-10-10.csv`).
- **2026-10-10: test paketi x86-64'te segfault etti** â€” ilk GitHub CI koÅŸusuna kadar
  gÃ¶rÃ¼lmedi â€” ve belgelenen DIS baÅŸarÄ±sÄ±zlÄ±k sÄ±nÄ±rÄ± ("geniÅŸlik â‰¥ 100") Ã¶zgÃ¼n
  taramanÄ±n test ettiÄŸi ilk geniÅŸlikti, kenar (80) deÄŸil. arm64 Ã¼zerinde bite-Ã¶zdeÅŸ
  olarak dÃ¼zeltildi (`plumerank/statistic.py`, `dis_unmeasurable`).
- **2026-10-10: stabilize edici, kaldÄ±rdÄ±ÄŸÄ± harekete karÅŸÄ± hiÃ§ Ã¶lÃ§Ã¼lmemiÅŸti â€” ve ilk
  Ã¶lÃ§Ã¼mÃ¼mÃ¼z yanlÄ±ÅŸtÄ±.** DÄ±ÅŸ bir mimari incelemesi kaydÄ±n ham Ã¶ncÃ¼le yapÄ±ldÄ±ÄŸÄ±nÄ±
  gÃ¶sterdi. Pencere baÅŸÄ±na tek bir ilkten-sona korelasyon bize kameralarÄ±n zar zor
  hareket ettiÄŸini, bÃ¼yÃ¼k tahminlerin gÃ¼rÃ¼ltÃ¼ olduÄŸunu yayÄ±mlatmÄ±ÅŸtÄ±; o 62
  korelasyonun 8'i stabilize edicinin kendi gÃ¼ven tabanÄ±nÄ±n altÄ±ndaydÄ±
  (`PHASE_RESPONSE_MIN`), iddianÄ±n en Ã§ok yaslandÄ±ÄŸÄ± pencere onlarÄ±n arasÄ±ndaydÄ± ve
  hiÃ§biri hareket edip geri dÃ¶nen bir kamerayÄ± gÃ¶remezdi. Onun yerine zemine karÅŸÄ±
  hakem edildiÄŸinde tek bir kamera gerÃ§ekten sallanÄ±yor ve diÄŸer her bÃ¼yÃ¼k
  kayma hareket eden gÃ¶kyÃ¼zÃ¼nÃ¼ izliyor (sÄ±nÄ±rlama 9,
  `results/registration_2026-10-10.json`).
- **Elde tutulan medyan bir yangÄ±na kadar yok olmaktadÄ±r** (27/54; o daha alt sÄ±ra
  istatistiÄŸidir, `plumerank/metrics.py` `percentiles`; bkz. "GerÃ§ekte neyi Ã¶lÃ§Ã¼yor").

## Kategori seÃ§imi

**YalnÄ±zca Genel.** Ä°ki Ã¶zel Ã¶dÃ¼l kutusu da bilinÃ§li olarak iÅŸaretlenmedi:

- **COOL'un En Ä°yi KullanÄ±mÄ± â€” reddedildi.** Bu hattÄ±n sÄ±cak dÃ¶ngÃ¼sÃ¼ arka plan
  modelleme ve optik akÄ±ÅŸtÄ±r; yeniden boyutlandÄ±rma / uyarlanabilir eÅŸik / kontur
  deÄŸil; bir COOL iddiasÄ±, o Ã¶dÃ¼lÃ¼n kendi Ã¶lÃ§Ã¼len-performans Ã¶lÃ§Ã¼tÃ¼ne karÅŸÄ± dÃ¼z
  kalÄ±rdÄ±.
- **Agentic Vision â€” reddedildi.** Orkestrasyon dÃ¶ngÃ¼sÃ¼ yoktur. Sevk gÃ¶revlisi
  tasarÄ±m gereÄŸi komutadaki insandÄ±r â€” bu *ta kendisi* gÃ¼venlik gerekÃ§esidir
  (yukarÄ±daki "Sorumlu kullanÄ±m"a bakÄ±n).

## Lisans

[Apache License 2.0](LICENSE) Â© 2026 Edy Cu

Bu Ã§eviri, aynÄ± lisansÄ±n izinleri uyarÄ±nca yapÄ±lmÄ±ÅŸtÄ±r; lisans ve telif, orijinal
sahibi Edy Cu'ya aittir.

## TeÅŸekkÃ¼rler

**OpenCV AI Competition 2026, powered by AWS** iÃ§in yapÄ±ldÄ±. GÃ¶rÃ¼ntÃ¼ler: HPWREN
FIgLib (`http://hpwren.ucsd.edu/`). Tek Ã¶ÄŸrenilen bileÅŸenin eÄŸitim verisi:
Pyro-SDIS (Apache-2.0, FransÄ±z yangÄ±n servisleri) â€” asla bir Kaliforniya pikseli
deÄŸil.
