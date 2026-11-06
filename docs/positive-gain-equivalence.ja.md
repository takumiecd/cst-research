# 正のgainに関する関数クラスの補足

[English (canonical)](positive-gain-equivalence.md) | 日本語

これは実行済みの凍結69 pilotについての解析的な補足であり、新しい学習結果でも、
封印済みソース・protocolの変更でもない。

実モデルは**biasのない**3つの正方CST写像、それぞれの後のReLU、そのReLU直前に
適用する正のscalar gain、hardな重み上限のない学習可能な最終affine写像を持つ。
途中にnormalization、dropout、隠れ層の加算項はない。次のように置く。

\[
h_0=\operatorname{ReLU}(W_{\rm in}x+b_{\rm in}),\quad
h_\ell=\operatorname{ReLU}(W_\ell h_{\ell-1}),\quad
\widetilde h_\ell=\operatorname{ReLU}(g_\ell W_\ell\widetilde h_{\ell-1}),
\qquad g_\ell>0.
\]

正の斉次性から帰納的に、

\[
\widetilde h_\ell=\Bigl(\prod_{j=1}^{\ell}g_j\Bigr)h_\ell,
\quad
F_g(x)=W_{\rm out}\widetilde h_3+b_{\rm out}
=\bigl(GW_{\rm out}\bigr)h_3+b_{\rm out},
\quad G=g_1g_2g_3>0.
\]

したがって、可逆なパラメータ写像は\(W'_{\rm out}=GW_{\rm out}\)であり、
他のパラメータと出力biasは変えない。逆写像はその重みをGで割る。
**固定された正のgainに対し、理想的な実数演算でのネットワーク全体の関数クラスは
stockと同一である。** 各siteの実効的な振幅範囲は変わるが、制限のない出力Denseが
積を吸収するため、全体の関数クラスは拡大しない。固定されたパラメータの組に対し、
理想的なforwardはgainの積だけに依存する。各siteにRMS合わせを分配する固有の
利点を、これだけで示したことにはならない。

ただし、介入が初期化だけの変更になるわけではない。同一の保存済み学習可能tensor
でも、gainとstockは一般に異なる関数から開始する。パラメータ座標、勾配scale、
適応的更新・epsilonの挙動が異なる。変換後の出力座標で表した実効的な正則化にも
注意が必要である。特に、decoupled weight decayの乗算部分そのものはこの定数の
線形rescaleと可換だが、それだけでAdam全体の軌道が一致するわけではない。
FP32の有限精度、overflow/underflow、中間activationのscaleについては、厳密な
bitwise可換性が成り立つとは限らない。

実行されたgainは521.2747192382812、258.95751953125、945.0846557617188で、
G≈127575095.34510244である。`analysis.json`は実際の初期保存状態2つ、合格済み
生データ監査03、全manifest検証済み69/975ソースcheckoutに結び付く。
固定された**合成入力**4×28を2種類（linspaceとゼロ）使う公開CPU-FP64 comparatorで、
出力重みを変換したstockとの最大絶対差は8.67×10⁻¹⁹および0だった。
dataset、ラベル、タスク評価、GPU操作、新しいqualification gateは使っていない。

同一の全体関数クラスで部分的な学習回復が観測されたため、stockのAUC=.5という停滞を、
表現可能な関数の絶対的不足や不変の初期ゼロ列パターンだけで説明することはできない。
ただし、conditioningと初期化、optimizer座標、正則化、有限精度の効果を切り分けた
わけではなく、Denseとの残る差も解決していない。既存の3epoch・seed17探索という
制限と、保存済みの全amplified componentwise診断の不合格はそのまま維持する。
追加arm、学習、seed、密度を示唆しない。

[外部証拠の規約](reproducibility.ja.md)に従い、解析reportは
`higgs-depth-gain-20261106/stats-positive-gain-equivalence-01/analysis.json`、
SHA256 `a90fa2323c706095b4975e8b81bc29c745322ae13a56fbb467dd8c8f8c05f5c7`である。
reader SHAは `9d8d201eece3618feb1a89f15ea143a111637569d3f65dafe11df774fe5653b3`。
実測consumer69・外部975・先行raw auditのidentityは変更せず、後続の解析的補足を
それらの学習qualificationとは分離する。
