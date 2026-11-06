# 線形 Atom 数 pilot：Stage B 学習案

[English (canonical)](higgs-linear-atom-learning.md) | 日本語

**CPU 準備のみ。未凍結・未実行・未 admission。** 同じ登録済み線形 Atom 研究の2段階目の
実装成果物である。不変の [Stage A protocol](higgs-linear-atom-scaling.ja.md)、Stage A の
全114 file、registry、履歴上の qualification 専用 entrypoint は変更しない。
別の Stage B entrypoint は、実 native16 qualification・全 file release・独立した科学的 CPU
admission に合格し、owner がその厳密な hash と新 source/runtime/storage/launch を seal するまで
使えない。

## 変更された研究目標：オーダーの特定

ユーザーの最新目標は、タスク・学習 recipe・budget・品質基準を明示した条件付きの
必要 Atom 数 A*(N,L) を実験で特定し、
**A∝N を仮定せず検証する**ことである。選んだ cN budget で学習できることは上界の一例であり、
最小数やそのスケーリングオーダーの特定ではない。変更しない16条件1epoch pilot は最初の screen
に限る。

今後の別途レビューする研究では、深さ・タスク・horizon・margin を固定して、学習可能性の境界と
調整済み Dense 品質の境界を区別する。各幅・深さで両境界を挟み込み、可能な範囲で
幅1024/2048/4096/8192 を調べ、適切な独立 LR 調整と paired seeds を用いる。
A/N と有効指数を確認し、定数や劣線形の候補も比較する。

\[
p_{\mathrm{eff}}=\frac{\log(A^*(N_2,L)/A^*(N_1,L))}{\log(N_2/N_1)}.
\]

有限で不確かな bracket、少数の幅、短い horizon、学習失敗だけで漸近則は確定しない。
初期 scale/conditioning による stall と density/capacity の限界を分け、失敗した学習 recipe を
A* の内在的な下界としない。この文言変更は追加の幅・Atom budget・正規化・更新や自動追試を
承認しない。後続の範囲・protocol は、実 pilot 証拠の admission 後に別途指定する。

## 固定範囲

Stage A と同じ順序の全16条件を実行する。N1024/2048、hidden H4/H8、Dense-stock、
CST c1/4/16。各 H−1=3/7 個の内部正方写像で厳密に A=cN とする。
入出力 Dense は学習可能。正規化 NONE、gain1、calibration なし、明示的な従来初期化、
q1、LR0.0003、B4、FP32/TF32 off/eager を維持する。
LN/gain fallback、density/LR sweep、初期化修正はしない。

各条件は fresh0 から full100k training の1epoch、真の B4 更新25000回、合計400000タスク更新。
epoch1 順序は PCG に依存しない PyTorch `randperm(100000)`、CPU generator seed117。
実 tensor bytes を保存する。品質による early stop や自動的な条件延長はしない。
Dense4対照で同じ recipe における H8 の妥当性も測る。失敗した Dense 対照から CST parity は
示せない。探索的 screen であり、正式非劣性・最小の十分な c・実測線形コストの主張ではない。

## 継承 admission と新しい初期状態 gate

Stage A producer は `446f25f1ec339f1e3acb4cf84d96ac5825306ab7`、source manifest
`717bcbcb88dde83fe619cad547bd9ffbc9092e35be5787fdcad7f1eda2859db4`、protocol
`9a34b7fc5a30d58b47c36db25ed15b148dae7c347fbd4898beb0fda8e7bc7b28`。
Stage B 自身の数値 qualification に producer を付け替えない。

タスク array の decode 前に、owner が seal した独立 report と root file-release の hash、
実際の正常 exit0/completed outer receipt、raw 全 inventory と全 bytes、厳密な16条件の
qualification/initial0 SHA binding、旧114 source files、runtime/data を確認する。
準備中や一部だけ合格した report は拒否する。必要な report status は
`passed-independent-StageA16-qualification-only-NOT-task-training`。
準備段階で report や native 合格を捏造しない。

各タスク条件で新 model/optimizer を構築し、完全な initial core が実際に qualification を通った
initial0 と literal に一致することを確認する。epoch0/updates0、空の optimizer history、
全 gradient None を要求する。保存 state を main model へ load せず、qualification state8 を
学習開始に使えない。initial0 は重複 checkpoint を避ける SHA-bound producer reference とする。
継承する16×8 normal、16×4 resume の合成 qualification 更新は192回で、タスク学習ではない。
Stage B に新しい qualification/replay 更新はない。

## 証拠・選択 state・測定

epoch1 の full model/optimizer/gradients/moments/clocks、厳密な順序、full50k DEV
logits/labels/raw row IDs、選択 checkpoint の SHA reference を保存する。
候補は固定 epoch1 のみ。**同一 selected state** を full100k TRAIN と50k DEV で学習時計外に
評価する。2回の独立 native full-state restore で TRAIN array bytes が一致し、restore 後の
DEV array bytes・metrics が cached epoch1 と一致することを要求する。
selected TRAIN arrays、DEV SHA reference、load proof を保存する。
実際の3/7 owner のみ native `max(before+1,serialized)` generation 規則を適用し、他の全 state は
literal に確認する。評価中の mutation は実 post-load baseline と比較する。

変更しない裸の `ordinary_step` epoch loop を使い、更新ごとの観測コピーは入れない。
同期した学習区間の直前に memory peak を reset し、checkpoint/評価前に取得する。
resident arrays/model/optimizer/native buffers と残存 allocator reservation の範囲を明記する。
preparation、full raw admission、diagnostics、snapshot、prediction/checkpoint I/O、restore、
evaluation は別々に報告し、inclusive totals とその成分は加算しない。
固定256 TRAIN features の initial/end activation RMS・ReLU retention と、最後の実 B4 の
勾配・採用 m/v/clocks は時計外の診断で、新しい backward probe ではない。
endpoint の差を1更新の変位とは呼ばない。

## source・data・実行境界

公開975 source と source/runtime compilation path は Stage A と同一にする。
RTX4090 runtime pin は
`1e7ecf23d0b4fa74f823178406a8a1bcda7020f504ce34fe62e0e7d63bdfe23b`。
train/development artifact は
`58127b3aefd643d209ee22045dc9df83399f0d043100ddb583b733e193ac15c3`、
manifest は `530db950794c74b665d79251d9353f7cb9f354ae8c16901f50f7b66450488526`。
original/future confirmation と official test は stage しない。
development は既に以前の recipe 選択に使っており、新しい confirmation ではなく探索的である。

stage する厳密な10入力は、元の8個（consumer manifest は新しいもの）と
`stage-a-admission.json`、`stage-a-file-release.json`。
実 Stage A raw root はその場で read-only に読み、複数 GiB の証拠コピーは作らない。

```bash
python -B -m cst_research.experiments.higgs_linear_atom_learning_driver \
  --consumer-commit NEW_REVIEWED_STAGE_B_COMMIT \
  --stage-a-root /path/to/actual-readonly-StageA-attempt \
  --stage-a-admission-sha256 OWNER_SEALED_ACTUAL_REPORT_SHA \
  --stage-a-file-release-sha256 OWNER_SEALED_ACTUAL_RELEASE_SHA \
  --cache-root /path/to/same-verified-linear-cache \
  --output /path/to/fresh-learning-output --timeout 7200
```

これは案であり、実行承認や完成した execution bundle ではない。
Stage A qualified raw と Stage B new raw を合わせた計画上限は8GiB。
launch 前に実 local free から保守的 compressed 総量・最大 owned scratch・primary-study reserve を
引いて8GiB以上を残す。全 compressed raw を local に、不変 file を remote に保持し、
条件ごとの owned temporary CPU admission で永続的な全展開コピーを避ける。
owner が実 runtime/profile/storage を確認し有限 launch を seal する。
foreground timeout/失敗では証拠を保持し owned process を回収する。
retry、自動 resume、追加研究、既存証拠の削除は承認しない。
