# Compact profileの境界診断

[English (canonical)](compact-boundary-20261109.md) | 日本語

所有者が依頼したCPU分析のみです。固定torchcst0d277ae6dc0ae14fe515c7559f39535ed0238950を
用い、モデル・optimizer・GPU・本番backend・速度/品質実験は変更・実施しません。
主な問いは、public Triweightが支持・座標・幅の境界で連続か、離散L2正規化を含めた場合に
どうなるかです。

周期128・間隔1のgridで1組の[幅,中心]を評価します。public TorchのTriweight点評価と
離散L2(floor1e-6)、関連形状controlのpublic raw BiweightとWendlandC2を使います。
独立controlは前研究のGaussian固定cell積分と、正規化Triweight密度の固定cell積分です。
後者は数学参照のみで、新public cell_mass bindingではありません。密度は
35/(32R)*max(0,1-((x-mu)/R)^2)^3で、正確な多項式原始関数と端の裾を安定計算する式で
積分します。compact積分の周期imageは-2..2で固定します。異なる契約・正規化・幅を
タスク品質のために揃えた比較ではありません。

24条件を固定します。中心7.3で半径0.5/1/2を両方向、中心7で支持半径1、中心7.5で
空支持から生まれる半径0.5、半径1で支持中心7、半径0.5で中心7.5をそれぞれ両方向。
以前の区間境界の中心7.5・半径0.5001、枝内(0.75,7.3)の幅・中心、空支持内(0.4,7.5)の
幅・中心、半径0.75の周期境界中心0、Triweight L2 floorの幅・中心の両方向も含めます。
floor位置は1 siteのみ有効なときの(1-(d/R)^2)^3=1e-6から求めます。
変位はGaussian研究と同じ1e-2..1e-12の8尺度。FP64の6profile、FP32の主4profileで
240条件・1920変位です。1軸factorの診断であり、LM全体や行列の時間測定ではありません。

有限・予測ノルム、絶対/相対Taylor残差、cosine、符号化上の無変化、raw支持site数、raw
Triweightノルムを記録します。compact支持端では予測方向がゼロに近づくため、滑らかな
関数でもcosine・相対比が未定義・非収束になり得ます。絶対残差の次数も評価します。
L2 floorは別に調べ、傾きの折れと値の有限の跳びを区別します。独立テストでpublic多項式・
解析微分・支持端の一次/二次微分ゼロ、正確な積分の求積・mass・周期同一性、L2 floorの
連続性と両側の異なる傾きを確認します。値とJVPが有限であることを要求します。
source/runtime/result hashと正負の観察を残し、新たなタスク許容差やkernel採用は決めません。

runnerはscripts/inspect_compact_boundaries.py、証拠は
CST_RESEARCH_EVIDENCE_ROOT/compact-boundary-cpu-20261109-01です。
