# LMの幅スケールと全学習行列のCST置換

[English](lm-width-scaling-plan-20261108.md) | 日本語

状態：設計・準備のみ。ユーザーは4090カーネル比較後のLM幅スケールを選び、出力射影や
他の学習行列も対象に加えた。本書は実行中studyの登録、停止済み確認実験の再開、
必要backendの実装済み宣言ではない。

## 問いと固定する実験族

`A = ceil(c * sqrt(d_in * d_out))`、まずc=8を固定して、幅を増やしたときの有用性を調べる。
Atom最小数やOmegaを求める研究には戻らない。

幅候補は512/1024/2048。4個のpre-LN block、FFN倍率4、context128、batch2、accumulation4
（1024 token/update）、tokenizer/語彙、train/DEV stream、学習token数、cを固定する。
head次元64（head数8/16/32）の共通幅familyを提案する。新しい明示configとfresh Dense対照が
必要で、従来の幅512/head4の観測を同一条件baselineとして継承しない。

既存TinyStories/GPT2 tokenizerのデータ手順と照合済みtoken manifestを使う。
開始候補は通常LR3e-4、全Atom倍率16、AdamW(.9,.95)、decay0、global clip1。
学習前に合格したPolar/初期化recipeを固定する。minimum-001/q0は候補であり、全行列LMで
検証済みという意味ではない。残差出力のscaleも含め、公開CST初期化で各Denseの二次momentに
合わせる。最初はFP32/TF32無効。source・data・LR schedule・実token budget・選択規則を実行前に固定。

## 置換条件

| 条件 | CSTの行列 |
| --- | --- |
| Dense | なし |
| FFN | 全blockのFFN up/down |
| QKV | 全blockのQ/K/V |
| QKV+FFN | 上記の両方 |
| +O | 全blockのQ/K/V/OとFFN up/down |
| +共有語彙 | 上記＋共有token embedding/語彙出力 |
| 全学習行列 | 上記＋学習する位置埋め込み |

最初の4条件でFFNとQKVを切り分ける。追加条件は累積し、幅512でまず成立を確認してから、
成功した固定familyを1024/2048にも加える。未検証条件を最初から大きなmulti-seed gridに広げない。
LayerNorm・softmax・causal mask・活性化・残差加算と、1次元のbias/正規化係数は処理を保つ。
「全行列」は全計算の置換を意味しない。

## 語彙の重み共有とbackend検証

現LMのlogitsは`embedding.weight`を使い、token embeddingと語彙出力は共有されている。
lookup/decoderで同じCST Parameter/AtomStateを使い、optimizer状態・更新は共有Parameterごとに1つ、
両用途の勾配加算とcheckpointの共有関係を保つ。非共有の比較には同条件のDense対照も別に必要で、
初回設計には含めない。位置埋め込みは学習行列として別条件にし、別の位置表現へ黙って変更しない。

現実測operator095684f0が検証しているのは正方N2048/8192/16384と小型FFNである。
幅研究の全長方形、50,257語彙の射影、CST embedding lookupは未検証。
runtime/operator/backendはtorchcstに実装し、本research repoへ複製しない。学習前に：

- 実際の正方・up/down長方形、幾何平均A8予算、R256、両保持方式を許可・独立検証する。
- 学習中に全dense重みやtoken one-hotを作らない語彙対応・直接lookupを実装検証する。
  実語彙数・softmax領域・重み共有を維持。内部paddingも演算意味を保ち、明示・検証する。
- CPU FP64/profile oracleでforward/dX/全Atom勾配を比較。重複token ID・境界・共有encode/decode勾配・
  optimizer状態の重複排除・保存・次更新の厳密resumeを含める。
- 初期化target、行ごとのenergy/coverage、ゼロ行、活性・勾配を記録する。非常に長方形の語彙行列は
  均衡したFFN/Attentionで見えなかった問題を出し得る。kernel/c/radius/LRを変える前に原因を残し、
  変更は別versionにする。

## 実行段階と判断の境界

1. 4090単体比較を完了・監査・回収。A100 MIGと4090は各Dense対照付きで分ける。
2. CPU宣言・GPU演算/optimizerの独立検証後、新familyごとにLM実更新2回、保存後の次更新/DEV再現を確認。
3. 幅512から短期development。初期予算案は各条件約1M token・seed17、既存20M-token LR/data horizon。
   これは提案で、本書によって学習を起動しない。
4. 動く条件と同一token budgetを固定して幅familyを実行。train/DEV CE・PPLをtoken数・wall timeの両方で
   重ねる。実step/token、完全update時間、allocated/reserved、準備時間を記録。
5. 長めの3seed確認は別に予算・protocolを固定してから。停止済み旧確認を再開せず、最終確認scoreで
   選択しない。普遍的最適・非劣性を仮定しない。

既存learner/qualificationの明示config・許可範囲を拡張し、学習loopを別に作らない。
完全updateの時計にzero_grad/LR設定・転送・accumulation・clip・optimizerを含むかを明記し、
過去の時計は元sourceの定義を保つ。geometry/rank/coverage診断、評価、checkpoint IOは主計時外。
失敗・途中停止・OOM・未対応shapeも記録に残す。
