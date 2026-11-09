# 新M16：CST数を揃えたFFN配置比較

[English](lm-m16-placement-20261109.md) | **日本語**

実行前に固定した探索計画。ユーザーの依頼は配置と挙動の分析であり、新しい
optimizer、カーネル、正規化、学習率の改善実験ではない。

主比較8条件はCST4層、up/down各2層、有効FFNスカラー数を一致させる。
両方CSTにするブロックの全6組（01,02,03,12,13,23）と、全ブロックに1層ずつ
置くup/down/up/down、down/up/down/up。ブロック番号は0始まり。
pair12は従来良かった中央2ブロックを新M16で比較する配置。
別枠の対照としてDense、同じ中央4箇所・同じ有効スカラー数のDST-pair12、
全8FFN層CSTも新規に学習する。Denseと全8CSTは同数比較には含めない。
attention、正規化、埋め込み、語彙出力はDenseのまま。全CST Transformerではない。

前回検証済みと同じtorchcst95f647f97749f61f5f5c15ac29932f908fb334ceの
connected ordered M16 envelopeを使用し、runtimeは変更しない。
既存の物語分離済みTinyStories/GPT2データ、manifest SHA256は
5b995103ca95d3c9b62a4e860039683dc734b0a0bb20e232db1b2e45aa690d01。
幅512、pre-norm4ブロック、8head、文脈128、FFN拡張4、
置換層のAtom数8192（A=8sqrt(d_in*d_out)）。名前別seed17の初期値を使い、
M16には旧初期化のraw行をコピーする。RMS再調整はしない。
共通のCST行・通常パラメータ・残すDense行列の初期byteを一致させる。
係数・形状・深さ・CST各fieldのLRを配置間で固定する。

全条件を新規に4882更新/4,999,168ターゲットまで学習する。
2文脈×4蓄積、1024ターゲット/更新。20MトークンのLR/topology期間、shuffle905。
AdamW通常LR.0003、CST全field32倍、tau.003、fraction0、半径[.5,50]、
振幅上限32、betas(.9,.95)、eps1e-8、decay0、clip1を固定。
DSTは既存の行平衡RigL型・Dense保存の実装と選定済み疎重みLR.003を使う。
250更新ごとのtopologyイベントではAdamを進めない。追加HPOなし。
DEV8192ターゲットを初期、200更新ごと、256,976,2441,4882で評価し、
CE、perplexity、次トークンtop1を記録。confirmation/testには触れない。
単一seedの探索であり、同等性・収束・Denseによる因果的補償は主張しない。

0,976,2441,4882で、同じTRAINオフセット0,4096,8192,12288の各128トークンを
元ラベルで観測する。勾配は1文脈CEの非蓄積・非clip値。
全FFN up/down、FFN枝、norm2、attention出力、ブロックについて入出力RMSと
各cotangent RMS、Parameter/field勾配RMSを記録する。
異なるパラメータ化のraw勾配の大きさは同等な関数更新量ではない。
中心化したtoken特徴Gramの固有値からstable rank、entropy rank、最大成分比率を
計算する。これは128文脈行の指標であり行列全体のrankではない。
オフセット0のtensorを保存し、同じトークンで条件間の表現/勾配cosineと
centered linear CKAを計算する。類似度は正しさでもDense教師への適合でもない。
同文脈でFFN/残差RMS比、両者cosine、FFN/負cotangent cosineも記録する。
最後は局所感度であり、optimizer更新が損失を下げる証拠ではない。
各probeでParameter/buffer/RNG不変を確認し、観測あり/なしの次更新について
シリアライズ復元後のモデル・optimizer完全一致を実行前に確認する。
学習更新則は変えない。

学習前にnative envelope/coalesced/回帰テストと独立求積を再実行する。
代表pair12のup/downについて初期/最終の全8192Atom FP64因子oracleで
Y/dX/raw-dPを既存atol1e-4/rtol4e-3で検証する。
全11条件の復元・次更新完全一致を確認。3時点checkpointのSHA、
FP32有限moment、更新clock、最終digest、最終DEV再現を監査する。
ソース/データ/設定を固定してhashを保存。大きいcheckpoint/tensorはremoteに残し、
検証可能な小さい結果を取得する。同期した学習全step時間は入力準備・DEV・
probe・保存を除外し、最初10更新を除く。allocated/reservedは別記し、
観測の割当履歴がreservedに影響し得ることを明記する。
DST時間はtopologyイベントを含み、Dense保存なので疎カーネル速度ではない。

予約済み4090で逐次実行し、他のジョブに干渉しない。
配置でタスク損失が変わるか、どこでスケール・特徴多様性・勾配伝播が異なるかを
分析する。良いhybridがあっても、観察だけでDenseの因果的補償を断定したり、
最適化と容量のあらゆる説を切り分けたりはできない。
この計画から推測的改善や追加探索は開始しない。

## 検査失敗後の数値受入基準の追記

job01ではDense、DST-pair12、pair12の4882学習更新が終了した。
pair12最終検査のdown層radius_fraction勾配1要素（raw全32768要素中）が
元の要素別基準で失敗した。native -5.71988e-5、FP64因子 -1.74586e-4、
絶対差1.17387e-4。半径field全体の相対L2誤差は1.54e-5、
Y/dX最大差は約1.2e-6。該当半径.5079038、中心501.9923/1916.0001。
native/referenceの物理行は同一で、envelope chain適用前の1Atom物理計算でも
差が再現する。誤差はFP32物理計算で生じており、古いdecodeの利用や
fraction chainの誤りではない。元job/ソース/結果は変更せず、
元の検査が通ったとは言い換えない。

残り配置の結果を見る前に、独立64点Gauss-Legendre積分witnessと別の数値受入
検査を追加する。全AtomのFP64因子との比較で、Y、dX、raw各fieldのそれぞれに
最大絶対差<=3e-4かつ相対L2<=1e-4の両方を課す。絶対差は既存の独立primitive
テストと同じ基準、相対差は同テストの3e-4より厳しくし、raw全列をまとめず
fieldごとに確認する。これは明示的な事後の基準変更であり、独立した事前規定
ではない。旧要素別基準の失敗数も引き続き記録する。発端のpair12だけでなく、
完了した全CST条件の代表up/down各1層をこの検査で確認する。

runtime、初期値、学習コード、更新則、データ順は変更しない。
残り8条件は同じ固定済み学習ソースで新規に実行する。完了済み3軌跡は
元JSONのhashとcheckpoint参照を付けて別証拠jobに再利用し、
pair12の完了状態は独立再検証とoptimizer clock確認後にのみ復旧する。
全最終checkpointを監査し、最終DEVを再現する。
探索的な配置分析であり、数式やdefaultの採用判断でも、元の厳しい要素別検査を
全条件が通ったという主張でもない。

## 事後の特徴順序分析（読み取りのみ）

pair12最終値と全CSTの976更新probeを確認したところ、単純なRMSや特徴rankには
一律の崩壊が見られなかった。残りの配置を評価する前に、保存済みオフセット0の
tensorの分析だけを追加する。特徴ごとのtoken平均と、tokenごとの特徴平均を引き、
特徴軸トーラス上のlag1/2/4/8/16 cosine、低周波エネルギー（mode1..N/16）、
高周波エネルギー（mode>N/4）、Nyquist正規化spectral centroidを計算する。
FP64 rFFTで片側スペクトルの重複度を補正し、全条件・全時点・全保存箇所の
x,y,dx,dyに同じ定義を適用する。これは順序・基底依存の指標であり、
回転不変なCKA/rankと合わせて座標幾何への偏りの可能性を見る。
滑らかさやrankの高低そのものを良し悪しとは扱わない。
事後の記述的追加であり、学習条件、Dense教師、因果介入、固定runtimeや
学習軌跡の変更ではない。

全CSTの5M時点を確認した後、出力/cotangent RMSを特徴軸の行平均成分と
行中心化成分にも分解する。pre-norm残差にはLayerNormで除かれる特徴一定成分が
含まれ得るためである。全CST条件の保存済み3時点から、物理半径、振幅、fraction
境界、中心と格子点の距離も記述する。保存されたsemantic optimizer-field
manifestでParameterとmomentを対応させ、全Atom、|a|>=.01、|a|<.003の集合で、
bias補正Adam scaleのepsに対する大きさ、保持momentからの提案量、方向別FP32
半ULPとの比較を出す。保存momentの診断であり、実際の次更新の再現やoptimizer
変更ではない。fraction境界で外向きの保持momentも記録する。
閾値は探索的な記述用binであり、全checkpointに同じ分析を適用する。
モデル・optimizer・GPU学習状態は変更せず、正確な分析スクリプトを保存する。
