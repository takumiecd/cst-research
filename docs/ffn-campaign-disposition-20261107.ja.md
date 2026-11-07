# FFN実験系列の実施結果と終了範囲

[English](ffn-campaign-disposition-20261107.md) | 日本語

今回の限定実験では、記録した小型言語モデルのタスクで、線形Atom予算を保ったCST FFNの
有用性を示しました。Dense超越・収束・3seed再現性・最小係数・漸近的実行時間・速度や
GPUメモリの優位を示したわけではありません。所有者の変更指示を反映した科学計算は終了し、
研究repoの統合はこのPRのmergeで完了します。

| 要件 | 実施結果と証拠 |
| --- | --- |
| 長方形・多トークンの数学 | 独立polynomial quadrature、CPU FP64/nativeのY・dX・全Atom VJP、明示的2048経路の拡大12条件を検証しました。R3の小型タスクfixtureは2ブロック・CST4投影であり、実4ブロックタスクではありません。[Backend記録](https://github.com/takumiecd/torchcst/blob/c3b68dcf6275f96d29a21ff2aa6e989d42de2cff/docs/research-history/cuda-linear/ffn-rectangular-2048-20261107.ja.md)。 |
| 因果LMの全FFN | pre-LN4ブロック・4heads・context128・CST8投影、attention/embed/outputはDenseです。共通初期化・データ・順序を固定し、因果maskも確認しました。[モデル・protocol](ffn-feasibility.ja.md)。 |
| タスク学習と対照 | 新規Dense・FFNなし・CSTで、選択N256/seed17の5Mと幅・縦横比系列を終了監査しました。保全した全形状でCSTはFFNなしより改善し、Denseには届きません。[R8](ffn-feasibility-results.ja.md)、[R10形状](ffn-feasibility-r10-shapes-20261107.ja.md)。 |
| 幅に対する線形Atom予算 | A=ceil(c*sqrt(d_in*d_out))のc8を固定し、N128/256/512、expansion1/4/16を観測しました。Atom・FFNパラメータ数の線形性は構成上の性質で、タスクの有用性はこの有限点だけの実証です。Dense attentionの二次項は残ります。 |
| 合意した長期seed系列 | seed29の全幅は5Mを完了し、学習後24投影VJP・9復元監査を通りました。未開始のseed43/71の6条件は所有者の判断で停止し、計画した3seed確認は未完了です。[保全report](ffn-feasibility-r11-seed29-20261107.ja.md)。中止条件は再開しません。 |
| 曲線・生成・速度・メモリ | 採用runごとに固定2promptの生成、train/DEV曲線、準備・経過時間、同期した完全更新sample、allocated/reserved peak、パラメータ数を保持しています。現eager CSTはDenseより遅く、実測allocationも大きい値です。 |
| LR機構の追試 | R12–R14の振幅だけの介入と、訂正版R15のAtom全体32・64倍を分けて監査・描画しました。全体高LRは短期最終DEVを16倍より改善しませんでした。[振幅だけ](ffn-field-lr-r14-20261107.ja.md)、[Atom全体](ffn-whole-lr-r15-20261107.ja.md)。 |
| 公開API | 絶対値field_lrs、意味的宣言、共通scheduler係数、checkpoint復元を実装しました。remote CPU25・native CUDA8、実4ブロック8設定・64 VJPと更新・復元の検証を通り、torchcst PR151で統合しました。記録した実タスク更新ではParameter・損失が試作と完全一致しました。[R16](ffn-field-api-r16-20261107.ja.md)。 |
| ソース・証拠保全 | 元ソース・データハッシュ・学習前の失敗・負の結果・所有者の停止・raw archive・remote checkpointを保持し、派生PNG/SVGと同等の英日記録をcommitしました。[固定ソースの再現手順](ffn-reproduction.ja.md)。 |
| 資源・範囲 | 所有者の実行先限定後は許可済みRTX4090だけで科学計算の子プロセスを逐次実行しました。検証workerは終了し、ユーザーのHubは起動したままです。A100/Colab再確保、HIGGS/DST再開、Atom下限・Big-Omega探索、Attention置換は行っていません。 |

終了時の証拠再監査ではR8/R10/R11の8archive・raw88ファイル、学習後64投影VJP・24復元監査、
観測24runそれぞれの2生成例・最終トークン予算・曲線/時間/メモリ・Atom式を確認しました。
R12–R16も別々の終了archiveと監査をハッシュ照合しています。commitした派生図58ファイルも
manifestと一致しました。report・検証器が裏付けるのはこの契約範囲であり、任意の第三者データの
真正性や、より強い品質主張は保証しません。

所有者の停止記録と正式readerの条件は保持しています。元の3seed計画は未完了のまま見える形で
保全し、成功へ名称変更していません。現FFN用の別extraが統合済み公開runtimeを固定しますが、
旧HIGGSのpinと計測ソースは変えず、過去の学習結果を新APIへ黙って移行していません。

Denseとの差が残る理由と、幅を固定してAtomを増やしたら差が縮むかは研究課題です。
seed29の幅128/256/512でDEV差は0.1062/0.1069/0.1126であり、幅だけでは解消していません。
高LRで最終投影RMSが増えても最終損失は改善せず、低い出力の大きさだけでは説明できません。
Atom容量、特徴の多様さ、座標系の条件、有限の学習時間はまだ仮説です。
固定幅のc8/16/32比較は別の限定容量実験であり、完了した結果や下限探索の再開ではありません。
この系列ではその新実験やAttention研究は開始しません。
