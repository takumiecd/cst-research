# cst-research

[English (canonical)](README.md) | 日本語

CSTをニューラルネットワークで使えるようにするための研究用repoです。
現在は小型言語モデルを入口に、FFNとAttentionでの有効性を確かめることを優先します。
再現条件・学習の挙動・失敗と修正を記録し、観測結果と仮説を分けて残します。
ライセンスは[Apache-2.0](LICENSE)です。

## torchcstとの役割分担

| repo | 役割 |
|---|---|
| [torchcst](https://github.com/takumiecd/torchcst) | 公開Module・演算の数学的契約・backend・autograd・kernelの正しさ |
| cst-research | 研究計画、モデル・タスク、比較条件、学習曲線、統計、実験の来歴と再現手順 |

このrepoはtorchcstのruntimeを複製しません。研究consumerとworkflow controllerをここで管理し、
演算実装は固定した外部sourceから利用します。
公開APIだけでは不足する実験用Graph adapterは、そのprivateな依存先・対象commit・
独立検証範囲を明示し、公開APIへの採用と実験内の利用を別に判断します。

[2026-11-08のworktree整理](docs/worktree-cleanup-20261108.ja.md)では、古いcheckoutの
全payloadとGit履歴を別デバイスへ保全し、独立した復元検証を済ませました。
torchcstに[半径の包絡線](https://github.com/takumiecd/torchcst/pull/156)を実装し、tauの調整と
各Atomの`radius_fraction`の学習に対応しました。CPU・演算とRTX6000 Adaでの検査は成功しました。
[sourceを固定したTinyStories FFNのtau探索](docs/ffn-radius-fraction-tau-20261109.ja.md)を実行中です。
5条件を短期比較し、事前に決めた規則で選ぶ2条件を長め・3seedで追試します。
公開APIはradius_fraction_init / radius_fractionへ揃え、qの別名は置きません。
[固定状態のFFN測定](https://github.com/takumiecd/torchcst/blob/feat/radius-envelope/docs/radius-fraction-speed-20261109.ja.md)では、
幅512・256行でphysical本体に対し21〜22%の追加コストでした。
限定した演算の結果であり、LM throughputやDenseに対する優位性ではありません。
[定数を準備する明示候補](https://github.com/takumiecd/torchcst/pull/157)は、同条件の1ジョブで旧写像より
約3〜4%短縮しました。公開デフォルトには採用していません。任意の`ffn`インストールも新実験のoperatorへ固定しました。

## 現在の方針：FFNとAttentionで使えるかを確かめる

**Atom数の下限・Big-Ωを求める研究は停止し、FFNとAttentionでの実用を確かめることを優先します。**
2026-11-06に採用した研究方針です。FFN・Attentionの開発パイロットを完了し、
4ブロックすべてのQKV同時置換、QKV+FFN同時置換まで確認しました。

[RTX6000 Adaの独立半径FFN比較](docs/ffn-independent-radius-20261108-results.ja.md)も、
初期の物理値と各射影8192Atomを揃えて完了しました。幅512・seed17・999424tokenで、
DEV CEはPolar3.42441、直接の半径学習3.42438、物理表現の半径固定3.42865でした。
8射影すべてで半径勾配が非ゼロとなり、平均半径は0.975→1.048、12.59%は下限0.5に達しました。
短期開発実験の観測であり、非劣性や最適値は主張しません。学習・半径・勾配の重ね描き、
数値検査と検証済み証拠を保存しました。[勾配と更新則の数式](docs/radius-coordinate-gradient-discussion-20261108.ja.md)では、
半径微分を通すPolar案と座標の移動量を使う制御を分けて記録しています。

その後の[完全なPolar連鎖律の同条件比較](docs/ffn-polar-radius-gradient-20261108-results.ja.md)では、
L/Uを保ち、両条件で同じCartesian Adam＋射影を使いました。同じ幅512の短期予算で、
DEV CEはdetach/liveが3.42404/3.42239で、曲線はほぼ重なります。
半径の勾配はs/tの半径方向へ届き、平均半径はなし側2.058に対してあり側1.623でした
（両方とも初期0.975からは広がっています）。既存のphysical Linear Planを使う
検証済み研究用合成であり、公開Polarは変更していません。Python写像の精度形式と
overheadも結果に明記しています。

[確定した関係と内部座標の設計](docs/polar-relations-20261108.ja.md)では、cell-massの恒等式、
観測結果、半径制御の仮説を分けて整理しています。再現可能なCPU検査で質量としての振幅と
Atomのノルムを区別し、エネルギー振幅を使う写像を候補として記録しました。新しい写像は未採用です。

合意した[FFN検証計画](docs/ffn-feasibility.ja.md)は`A=ceil(8*sqrt(d_in*d_out))`を使い、
幅256から始めて128/512も確認します。線形Atom予算での有効性と限定した縦横比の変更を調べ、
最小係数の探索は行いません。

[初期結果と学習曲線](docs/ffn-feasibility-results.ja.md)で、3条件pilot、半径診断、
数値検証、残作業を分けて記録しています。選択設定は幅128/256/512と限定した縦横比の
監査済み1 seed開発実験でFFNなしより改善しました。所有者の判断で主幅の3 seed確認を未完了のまま停止し、
短期のfield別LR実験を優先します。seed29の結果は未完了の確認証拠として保持します。現在のeager FP32はFFN Parameter数を圧縮しますが、速度・GPUメモリの
優位は示していません。
[短期フィールド別LR比較](docs/ffn-field-lr-r12-20261107.ja.md)では、中心LRとPolar外部時間を揃えて振幅LRだけ16倍から1倍へ下げると学習が遅れました。[監査済みの追試](docs/ffn-field-lr-r14-20261107.ja.md)には中心・外部時間16倍に固定した振幅8・32・64倍も含みます。
振幅8倍はAPI実装の工学的基準を通りました。その後、公開field_lrsはCPU25・CUDA8テストと
全8設定の完全LM更新・復元、学習後64投影VJPも通り、[torchcst PR151](https://github.com/takumiecd/torchcst/pull/151)で統合しました。
[検証とAPIの記録](docs/ffn-field-api-r16-20261107.ja.md)にまとめています。
ユーザーが意図した高学習率比較はAtom全体32・64倍と訂正され、その[別のstock optimizer比較](docs/ffn-whole-lr-r15-20261107.ja.md)も監査まで完了しました。
全体32・64倍は途中の点で先に改善する場合もありますが、最終の短期DEVは16倍より改善せず、最適値・収束は主張しません。

[両端で合流する境界のFFN短期比較](docs/ffn-closed-envelope-results-20261107.ja.md)では、
L・Uを振幅の両端で合流させました。新境界の初期q0/q1をRTX4090でそれぞれ999,424トークン学習し、
学習済みcheckpointの監査まで完了しました。Dense・従来q0は再利用しています。
弱いAtomの半径は約50まで広がりましたが、今回のpL4/pU1のDEV損失3.9506/3.9761は
従来q0の3.6735より悪い結果です。形状を検討するための学習曲線・振幅別の半径とqを保存しています。

### 完了した射影・Polarパイロット（2026-11-07）

幅256・seed17・999,424学習tokenの監査済みRTX4090パイロットでは、次のDEV CEになりました。
短期の実用可能性の観察であり、収束・非劣性を主張する結果ではありません。

| CSTへ置換した射影 | DEV CE | 証拠 |
| --- | ---: | --- |
| Dense対照 | 3.581960 | 各Attentionパイロットで取り直した対照 |
| Vのみ | 3.570822 | [結果と曲線](docs/attention-v-20261107-results.ja.md) |
| Qのみ / Kのみ | 3.592582 / 3.582850 | [結果と曲線](docs/attention-qk-20261107-results.ja.md) |
| QKV同時 | 3.580952 | [同時置換の結果](docs/attention-combined-20261107-results.ja.md) |
| QKV + FFN両射影 | 3.695186 | 同レポート、20か所のCST |

O、embedding、語彙出力、正規化はDenseのままで、causal softmaxも変更していません。
QKVはこの条件でDenseに近い一方、FFNを含めると差が残ります。QKV+FFNの対象射影は
2,891,776から237,568パラメータへ減りますが、現在の小幅・eager FP32実装では
速度・peakメモリの利点は示していません。

[A100のminimum基準の下側包絡比較](docs/polar-minimum-a100-20261107-results.ja.md)も完了しました。
400更新でfresh rational対照のCE 4.294037に対し、epsilon .001 / .003は4.248089 / 4.257022でした。
読み取り専用の中心移動量と振幅別の半径・qの曲線を保持していますが、新しい移動依存optimizerを検証したものではありません。
4つのstudyすべてで学習済みcheckpointの監査とcompact evidenceの回収・hash照合が完了し、workerも終了済みです。
新しい半径・更新policyは一緒に数式を検討してから進めます。RTX4090はユーザーの別作業用に返しています。
実装と証拠は[PR31](https://github.com/takumiecd/cst-research/pull/31)に続く
[研究PR32](https://github.com/takumiecd/cst-research/pull/32)にまとめています。
ライブラリの包絡変更は[torchcst PR152](https://github.com/takumiecd/torchcst/pull/152)で
明示指定の機能としてmergeしました。

### 研究の順序

1. **まずFFN。** 長方形の入出力、多トークン、forward/backwardの正しさ、optimizerを含む更新を確認します。
   その後、小型言語モデルのFFNの射影をCSTへ置換し、安定して学習できるか、validationでタスクに意味のある性能が得られるかを調べます。
2. **次にAttention。** まずquery・key・value・outputの射影をCSTへ置換します。
   トークン間のAttention計算とmaskの扱いを明示し、射影の置換だけで新しいAttention演算を実証したとは扱いません。
   最初の用途が動いてから、さらに踏み込んだ変更を検討します。
3. **失敗したら原因を分析して直す。** 失敗条件を残し、初期化、値・勾配のスケール、幾何、実装の正しさを調べます。
   原因を切り分け、担当repoで修正し、影響する検証をやり直してから実験を広げます。

直近の目標は、実際に使える用途を示すことです。Denseはタスクと学習系が正常に動くかを見る対照として残します。
網羅的なDense/DST比較、広範な調整、Atom数の漸近的な研究は後に回します。
学習曲線・validation性能・速度・メモリは有用な観測として記録し、この段階で他手法に勝つことを必須条件にはしません。
具体的なタスク・構成・成功条件は、次の研究で実行前に定めます。学習lossが下がるだけでは、タスクで使える証拠として十分ではありません。

torchcstの公開初期化APIを使い、初期化方式と小文字`q`を明記します。
初期化の変更などの修正案は、対象の条件で確認するまでは仮説です。
成功した実験・失敗・修正を追跡できる形で残し、CSTで何ができるのかをREADMEに積み上げていきます。

## これまでのHIGGS研究

**HIGGS campaignは科学的admission・評価が未完了のまま、新しい研究方針により停止しました。学習jobは完了し、primary/DSTの学習は動いていません。**
以前の実行元はtorchcst [PR #141](https://github.com/takumiecd/torchcst/pull/141)の
固定commit `97580582ed28c4e494bf5c2656bbe94a85c544c3`です。
28→N→N→2の中間層だけをCSTへ置換し、N=1024/2048、
α=0.001/0.002/0.005/0.01を調整済みdenseと比較した計画です。
入力・出力はdenseのまま全層を学習しました。3確認seedを使い、accuracy −0.2ポイント、
ROC-AUC −0.002の許容差を実行前に固定していました。

[機械可読な研究計画](src/cst_research/studies/higgs-alpha-confirmation-20261105.json)と
[再現・証拠の扱い](docs/reproducibility.ja.md)を参照してください。
過去の研究やraw証拠は元のtorchcst履歴・外部アーカイブに残し、ここへ一括コピーしていません。

統合したconsumer実装と測定履歴を以下に示します。元のsource/data identityを保持し、
統合を再実行や再検証とは扱いません。[統合の来歴](docs/integration-provenance.ja.md)を参照してください。

| 研究 | 状態 | 範囲 |
|---|---|---|
| [HIGGS primary](docs/reproducibility.ja.md) | STOPPED | 外部975、最終30モデル評価は未完了 |
| [HIGGS static/RigL比較](docs/higgs-dst-comparison.ja.md) | STOPPED | Stage1/2 consumer、共通期間の78モデルgateは未完了 |
| [A100 MIG比較](docs/higgs-gpu-comparison.ja.md) | COMPLETED | 限定1-epoch環境比較 |
| [座標LR](docs/higgs-coordinate-lr.ja.md) | COMPLETED | 短期pilotと有限2-seed追加検証、stock q1を維持 |
| [深さ・層位置](docs/higgs-depth-position.ja.md) | COMPLETED | developmentのみの13条件、限定的なstall証拠 |
| [固定output gain](docs/higgs-depth-gain.ja.md) | COMPLETED | 探索的3条件、部分改善後もDenseに劣る |
| [Covertype転移](docs/covertype-transfer-results.ja.md) | COMPLETED | 9モデル固定・heldoutを1回評価、CST両密度はDenseに劣る |
| [8更新の正規化診断](docs/higgs-depth-normalization-diagnostic-results.ja.md) | COMPLETED | 9×8 task更新の機構診断。品質選択・既定値採用なし |
| [正規化の3 epoch追試](docs/higgs-depth-normalization-learning.ja.md) | STOPPED | 元ea0はmain300k後にexit1・未完了を維持。保持証拠の再利用は別recovery系譜だけ |
| [正規化load契約のrecovery](docs/higgs-depth-normalization-learning-results.ja.md) | COMPLETED | 探索的9条件を結合完了。A104のRMS/LN救済後もDenseに劣り、正規化の既定採用なし |
| [線形Atom数：Stage B結果](docs/higgs-linear-atom-results.ja.md) | COMPLETED（限定Stage B） | seed17の1 epoch全16終点。H8 CSTはbias-only、Atomオーダ・最小cの主張なし |
| [scaleとLayerNorm](docs/higgs-initialization-results.ja.md) | COMPLETED | GPUごとの6条件を独立検証した2 cohort |
| [幅対応・TRAIN-RMS初期化](docs/higgs-initialization-results.ja.md) | COMPLETED | 旧方式と幅対応は停滞、実測Dense-RMSで救済。1 seed |
| [He／Xavier二乗平均初期化](docs/higgs-initialization-results.ja.md) | COMPLETED | q=0の対応比較と明示公開API。普遍的精度の主張なし |

[観測結果と限界](docs/bounded-validation-results.ja.md)で、完了した座標・深さ・gain結果を
実測commitとraw auditへ結びます。developmentで選んだpilotは最小alphaや正式な非劣性を
確定しません。HIGGS primaryとDSTは最終固定familyのadmission・評価が未完了のままSTOPPEDです。停止は完了を意味しません。
測定jobは元のsource snapshotを保持します。統合をjobの再開や未完了の科学的評価の完了とは扱いません。

[ゼロから徐々に開く境界の比較](docs/ffn-zero-opening-results-20261107.ja.md)は、
RTX4090でtau=.01/.03/.1、初期q0の3条件を976更新/999424トークン学習し、
学習済みcheckpointの監査まで完了しました。DEV CEは3.69173/3.69918/3.75535、
再利用した従来q0は3.67351、Denseは3.58196でした。弱振幅の半径は意図どおり
広がりましたが、この1seed短期比較では対照を改善しませんでした。
学習曲線・振幅別半径/q・初期化診断を保存しています。

## すぐに使う

Python 3.11以上が必要です。レジストリCLIは標準ライブラリだけで動作し、GPUを使いません。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cst-research list
cst-research show higgs-alpha-confirmation-20261105
cst-research validate --json
python -m unittest discover -s tests -v
```

インストールせず確認する場合は `PYTHONPATH=src python -m cst_research list` も使えます。
`validate`は登録内容と固定条件の検査です。学習結果、外部証拠の存在、GPUの正しさを
検証したことにはなりません。停止した正式なHIGGS campaignは固定した外部ソースで実行していました。
独立した数値admission後に変更なしのbenchmarkを呼ぶ
[限定A100 MIG環境比較adapter](docs/higgs-gpu-comparison.ja.md)を別に用意しています。

現在のFFN研究には、研究用ライブラリを固定した専用extraを使えます。

```bash
python -m pip install -e '.[ffn]'
```

ライブラリはmerge済みの[torchcst PR152](https://github.com/takumiecd/torchcst/pull/152)の
`7be68e4faf1813dd4b1a677ee432da5e3a8da9bd`に固定しています。
統合済みの長方形演算・初期化・field別LR APIに加え、明示指定の実験用半径包絡と
Torch2.6のAdamW互換修正を含みます。この固定はconsumerの包絡引数を利用可能にするためで、
これらの実験設定は明示指定の機能のままで、実装のmergeによって公開デフォルトを変更するものではありません。
数値環境の再現や過去runの移行ではありません。品質結果の再現には
[固定ソースの手順](docs/ffn-reproduction.ja.md)を使ってください。

過去HIGGS用のextraは元のstudy commitを保持します。別の環境で導入します。

```bash
python -m pip install -e '.[torchcst]'
```

これはライブラリ依存の導入です。現在のbenchmark runnerを使うには、
[固定torchcst sourceの取得手順](docs/reproducibility.ja.md)に従ってください。

## 外部証拠を参照する

データ・checkpoint・raw logs・pool状態・認証情報はGitへ入れません。
`CST_RESEARCH_EVIDENCE_ROOT`の下にstudy IDのディレクトリを置くか、
Git管理しない `research.local.toml` で保存先を設定します。

```toml
[evidence]
root = "./evidence"
```

```bash
cst-research --config research.local.toml show higgs-alpha-confirmation-20261105 --evidence
```

設定は所在を解決するだけで、コピー・ダウンロード・削除は行いません。
`--evidence`の出力にはローカル保存先が含まれるため、公開用レポートには含めないでください。
環境変数のrootが設定されている場合はTOMLより優先します。

新しい研究・変更は[CONTRIBUTING.ja.md](CONTRIBUTING.ja.md)に沿ってPRで追加します。

[幅を考慮したamplitude初期化実験](docs/higgs-width-amplitude-init.ja.md)では、gain・LayerNormなしの生の`0.1/sqrt(A/N)`を、従来初期化と以前の固定gain対照と比較します。限定したGPU実行と独立検証は完了しました。[結果](docs/higgs-initialization-results.ja.md)を参照してください。

[生のamplitudeをDenseのRMSへ合わせる追加実験](docs/higgs-rms-amplitude-init.ja.md)は、学習前のamplitudeを校正し、実行時gain1・正規化NONEを保ちます。限定したGPU実行と独立検証は完了しました。[結果](docs/higgs-initialization-results.ja.md)を参照してください。

[He／Xavier初期化pilot](docs/higgs-fan-initialization.ja.md)はデータ不要のAtom二乗平均目標と小文字`q`の明示指定を使い、新規の対応するDense／CSTを比較します。CPU準備・native検証・学習結果・公開default採用は区別します。

完了した[初期化の結果](docs/higgs-initialization-results.ja.md)に、旧方式・幅対応・TRAIN-RMS・データ不要のHe/Xavier・正規化対照をまとめます。明示的な公開APIは[torchcst初期化ガイド](https://github.com/takumiecd/torchcst/blob/main/docs/initialization.ja.md)を参照してください。

[実施結果と終了範囲](docs/ffn-campaign-disposition-20261107.ja.md)に、所有者が停止した3seed系列も含め、要件と証拠を対応付けています。

[半径の事後確認](docs/ffn-radius-review-20261107.ja.md)では振幅別・同じAtom IDで調べ、弱い・弱まったAtomの半径が広がることを確認しました。ただし現在の低q・弱Atom追加拡張なしの設定では、弱いAtomも半径4付近にとどまります。
