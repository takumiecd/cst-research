# HIGGS の線形 Atom 数・深層 pilot

[English (canonical)](higgs-linear-atom-scaling.md) | 日本語

**PREPARED、未実行。** 最初の成果物 Stage A は全16条件の新しい native
qualification のみで、タスク学習はしない。予定する Stage B は、別途レビューする
1epoch の探索的学習 consumer である。登録や CPU preflight は GPU admission
でもモデル品質の結果でもない。

## 構造と問い

学習可能な Dense 入出力と `28 → N × H → 2`、hidden H=4/8、ReLU H 個、
bias なし正方写像 L=H−1=3/7 を使う。Dense-stock と、L 写像すべてを
**各写像で厳密に A=cN**、c=1/4/16 の CST に置換したものを比較する。
初期 pilot は N=1024/2048、seed17。Dense4 条件と CST12 条件、合計16条件。
幅4096 と後続8192 は別途の固定 c 検証に留保する。

\[
P_{\mathrm{Dense}}=LN^2+31N+2,\qquad
P_{\mathrm{CST}}=4LA+31N+2=(4cL+31)N+2.
\]

学習スカラー数であり、c **と** L を固定した場合にのみ N に線形である。
L 個の単位 gain buffer は非学習。式だけで実行時間・peak memory・Dense 同等精度の
線形性は示せない。c16 までの失敗は、別の定数や学習 recipe による十分な線形 Atom
budget の存在を否定しない。

## 事前指定 recipe と準備履歴

主 recipe は **正規化 NONE、gain1、calibration なし**、B4、FP32、TF32 off、
eager の通常 complete step、stock q1 CSTAdamW、LR0.0003。
ordinary weight decay と amplitude weight decay は0.01、raw Atom weight decay0、
betas(0.9,0.999)、Adam epsilon1e−8。公開975 の radius-cell-mass helper、
Chebyshev modes8、merge factor2、w_c0.3、spacing1、
`GLOBAL_OWNER_WINDOW_FP32_GEOMETRY_SORT32_FLAT_C` を変更しない。
coordinate-LR adapter、Graph qualification/default 変更、norm fallback、
暗黙の初期化変更は行わない。

入出力 seed は seed+1/+2、内部 Dense は seed+100+j、CST は seed+1000+j。
ordinary 初期化は従来の `torch.nn.Linear` default（内部は一様±1/√N、
He-ReLU 補正ではない）。公開 CST amplitude 初期化の標準偏差は0.1/√A、
中心は balanced-output/uniform-input。したがって c の増加は amplitude scaling により
初期分散の増加を一部相殺する。深層信号・勾配の縮小は初期化や conditioning を反映し得て、
容量の証明ではない。H8 Dense の妥当性は未測定であり、崩壊した Dense 対照は CST 成功の
証拠にならない。旧 H4/N1024 の A104/1048 との厳密な parity は、有理数
c13/128、131/128 を使い旧 factory を変更せず確認する。

暫定 LN20 条件案は、**新しい GPU 結果を得る前**にユーザーの正規化 NONE 指示で置換した。
LN は matched Dense を含む別の後続診断候補に限る。既存9条件学習の証拠
`eefdef7e8b7598232c137a6f8589c74f1a511051d13bcbd8fe0b5a9a01dc8e84`
は recipe 選択の背景であり、新しい固定 c の証拠ではない。
既存30数値 module と10 study registry の bytes は不変。準備履歴は ignored evidence に残す。

## Stage A：新 native admission、タスク学習ではない

各実 N/H/c 条件の full-network Y、dX、全 owned parameter gradient を独立 CPU FP64
reference と比較する。Dense/ReLU を明示的に逆伝播し、変更しない975 chunked CST
reference を一度に256 Atom の graph で呼ぶ。全実 Atom・全幅を使い、full-network
 autograd graph や `[A,N,N]` tensor は作らない。小さい公開モデルで、この手動逆伝播を
独立 full autograd と照合する。実最大条件の CPU preflight RSS/time は外部に保存する。
reference array の見積もりだけを RSS や GPU 効率としない。

通常 cotangent1 と増幅 cotangent1e24 は GPU 実行前、16条件の **合成 CPU** report から選んだ。
決定 SHA は
`b86e9d69f211768239aaeda30b76863d16519f1bb13ee27ce0a25b30f79eeb31`。
増幅勾配は **optimizer に一切適用しない**。

- 通常 elementwise は |error|≤3e−5+3e−4|reference| を必須とする。
- 両 scale で各 gradient field と packed Atom 各列の relative L2≤3e−4、
  max-error/reference-max≤3e−4。FP64 で正規化した norm を使い、厳密にゼロの reference は
  actual も厳密にゼロでなければならない。
- 増幅 dX と packed 各列で、|reference| が3e−5+3e−4|reference| を超える要素と、
  非ゼロ actual が必要。増幅 componentwise の失敗は診断として残し、通常 FP32 の
  rounded-zero tail も報告する。
- 各 N/c を実初期化31 Atom の変更しない48-point polynomial quadrature で独立に照合し、
  全 Y/dX/packed gradients を確認する。深さ間の共有には先頭31 Atom の同一 bytes と
  保存 raw SHA が必要。
- fresh 合成32入力による通常 CE の連続8更新と、4更新＋native save/load＋4更新が一致すること。
  実入力・ラベル・明示的な8個の B4 順序、initial0、normal4/8、resumed8、全勾配、
  m/v/clock、モデル側 optimizer store、scalar-Adam 証拠を保存する。

owned parameter tensor は H+3（7/11）個。実 CST owner は index2/4/6 の3個、または
2/4/6/8/10/12/14 の7個に厳密に限る。native load 境界だけで generation は
`max(before+1,serialized)` になり得る。他の全 state bytes/metadata は literal に一致させる。
後続確認は実 post-load baseline を使い counter を書き換えない。
proof は合成更新の範囲を記録し、学習統計とはしない。
Stage A はタスク NPZ を decode せず、development/heldout も評価しない。

## source・data・実行・証拠

外部公開 runtime は `97580582ed28c4e494bf5c2656bbe94a85c544c3`、
manifest `17dad4ab1c0385a42e32fc7add37a724bc9e7a7a68f63ff3fd188ea3bcda1931`、
archive `edbb585de56e304887b118069b7c7ecd9e45ae5fe2e08a5afca828ad8544eb5f`。
既存 train100k/development50k artifact
`58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3` と
manifest `530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526` を使う。
development は既に探索的 recipe 選択に使っており、新しい正式 confirmation sample ではない。
future confirmation、original confirmation、official test は stage しない。
過去の aggregate feature exposure は不変の [data protocol](higgs-depth-position.ja.md) に記載する。

Stage A の `research-inputs/` は以下8個の regular file に厳密に限定する。
`consumer-source-manifest.json`、`original-source-manifest.json`、
`original-source.tar.gz`、`train-development.npz`、`data-manifest.json`、
`runtime-pin.json`、`recipe-selection-audit.json`、`cotangent-decision.json`。
worker 前に全 bytes hash と厳密な inventory を確認する。
RTX4090 runtime pin は
`1e7ecf23d0b4fa74f823178406a8a1bcda7020f504ce34fe62e0e7d63bdfe23b`：
Torch2.11.0+cu126/CUDA12.6/Triton3.6.0、Python3.12.9、CC8.9、128SM、
GPU25262096384 bytes。承認済み Hub profile は root launcher が別途確認する。
install・provision・retry・background process は行わない。

```bash
python -B -m cst_research.experiments.higgs_linear_atom_driver \
  --consumer-commit FULL_REVIEWED_COMMIT \
  --cache-root /path/to/verified-linear-cache \
  --output /path/to/fresh-qualification-output --timeout 7200
```

厳密で clean な consumer snapshot から実行する。共有 external source と compilation
path は consumer commit に依存せず公開 source/runtime identity で固定し、再利用ごとに
source bytes と binding を確認する。consumer 自体は commit ごとに別の immutable copy。
foreground timeout は owned process group を回収し、失敗証拠を保存して nonzero を伝播する。
recovery index は保存証拠であり、再利用 campaign state や合格証明ではない。
Stage B は Stage A の全 source bytes を保持し、実際に合格した16 proof/raw initial state と
同一 source/runtime/data を bind したうえで fresh0 から開始する。
qualification step8 をタスク学習開始に流用せず、qualification 完了を予定400k更新と呼ばない。

予定 Stage B は各条件 full100k の1epoch（25000更新、合計400000）、full50k development と
同一 selected-state の train/development predictions を学習時計外で保存する。
HPO、3seed の主張、自動 advance、正式 min-alpha 結論はない。最終 Dense 同等品質の主張には、
別途の適切な Dense/CST LR 調整、共通 horizon、paired seeds、事前固定 margin/family が必要。

raw 証拠の計画上限は8GiBであり、最終容量の保証ではない。必要な oracle/resume/checkpoint
arrays は保持する。全 SHA-bound compressed evidence を local に、immutable files を remote に
保存し、1条件ずつ owned temporary directory で独立 admission して永続的な展開済み全体コピーを
避ける。実行前に実 local free から保守的 compressed 総量・最大 scratch・primary reserve を引いて
8GiB以上を残し、owner が具体的 storage/runtime 条件を seal する。
既存証拠の削除は承認しない。証拠 locator は
`$CST_RESEARCH_EVIDENCE_ROOT/higgs-linear-atom-scaling-20261106/`。
