# タスクLMの出力・勾配診断

[English](lm-signal-diagnostics-20261109.md) | 日本語

所有者の依頼によりタスク性能差を層ごとの出力・勾配から調べる。
site-countと分散配置のDense/CST/DST計17モデルを対象にした事後read-only診断であり、
Dense教師・重み再現の学習はしない。既存の固定TRAIN観測では全8CSTのblock出力は
初期Dense近傍から後で約2倍、cotangentは約0.5–0.7倍になった。
これだけで原因特定・勾配が死んだという証明にはしない。

初期・976・4882stepで元の128-target TRAIN probeを測り、最終状態はoffset0/4096/8192/12288を別々の128行呼び出しで評価する（計512 targets）。
検証済みnativeのR<=256行の範囲を超えない。全block・norm・QKV・O・FFN上下・FFN残差出力の
入出力分布と入出力cotangentを記録する。mean/std/RMS/max・完全ゼロ/相対近傍ゼロ率・
最大8192要素のsample絶対値分位を使う。共分散participation/stable rankは最大128token行を中心化した
値であり、重み行列全体のrankや表現能力の下限ではない。因果Attentionのlogit分散・
利用可能key数で正規化したentropy・最大確率・Q/K/V RMSを計算するが、native SDPA本体は変更しない。
計51モデル状態・102probeを測る。

学習checkpointと本来のtask labelsは変更しない。完了・監査済みcohort、GPU/runtime/data一致、
checkpointハッシュ、ParameterとCPU/CUDA RNG不変を要求する。元128probeのlossは1e-6以内、
FFN生勾配RMSは絶対1e-10/相対1e-5以内で再現する。診断整合性の検査であり、native微分の許容差変更ではない。
CPUで一様Attention/中心化rankの既知解と、hookでタスク勾配を変えないことを確認する。

同じ位置のactivation cotangentは単位を揃えて比較できるが、Dense重みとCST内部座標の生勾配は
直接同等とはみなさず、既に記録した実関数の更新量と合わせて判断する。
出力scale・勾配減衰・更新scale・feature多様性は別の仮説。GELUのゼロ率はReLUの死んだunitの証明ではない。
1 seed・固定TRAIN probe・事後観測だけで因果は断定しない。optimizer更新・confirmation・追加HPOはなし。
evaluator・identity・ログ・統計をNFS保存し、compact回収を検証する。

過去QKVは幅256・旧Polar、QKVOは幅512・別operator/runtimeであり、同一cohortのO単独切り分けではない。
O/FFNの本質的な弱点と主張する前に、現recipe・同条件のQ/K/V/O単独とQKV/QKVO比較が必要。
実モデルではQKVが参照/値の表現を作り、O/FFNが変換したfeatureを残差streamへ戻す。
残るDense・正規化・入力統計・予算・同時最適化は検証すべき仮説で、確定した説明ではない。
今回のFFN cohort観測は、その新しいAttention役割別学習実験そのものではない。

証拠：CST_RESEARCH_EVIDENCE_ROOT/lm-signal-diagnostics-4090-20261109-01。

現radius-envelope physical native adapterの検証範囲は幅512・拡張4の長方形、1..256行に限定される。
現envelopeによるAttention役割比較の前には、正方形のbackend検証が別途必要で、FFNの証拠だけで
shape gateを緩和しない。追加TRAIN文脈は別々の128行呼び出しとし、既存の契約範囲を守る。
