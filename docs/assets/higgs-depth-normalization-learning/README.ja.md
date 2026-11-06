# 9条件の導出図表

[English (canonical)](README.md) | 日本語

[結果と限界](../../higgs-depth-normalization-learning-results.ja.md)。
独立admit済み証拠からの導出物で、raw dataset・予測・checkpoint・logではありません。
単一seed・3epochで、confirmation/testは採点していません。TRAIN曲線はepoch内online平均で、
選択精度表はdevelopmentで選んだ同じcheckpointの全TRAIN/DEV評価です。

| View | PNG | PDF |
|---|---|---|
| All nine cases | [PNG](all-nine-learning-curves.png) | [PDF](all-nine-learning-curves.pdf) |
| Matched normalization controls | [PNG](matched-normalization-learning-curves.png) | [PDF](matched-normalization-learning-curves.pdf) |
| Gain and density | [PNG](gain-density-learning-curves.png) | [PDF](gain-density-learning-curves.pdf) |

- [選択精度CSV：9行](selected-quality.csv)
- [資源量CSV：9行](resources.csv)
- [全学習曲線CSV：27 epoch行](all-curves.csv)
- [元/公開ファイルSHAの由来](provenance.json)

6図は監査済み導出packとbyte完全一致し、合計1,582,930bytesです。CSVは改行だけLFにし、
解析後recordは完全一致します。元と公開版のSHA256を別に記録しています。準備合計と
diagnostics内訳は重なるため、足し合わせません。未記録のparent restore秒は空欄で、0では
ありません。計時は明示した2attemptにまたがり、完全なadoption-timeや一般的throughputは
主張しません。
