# タスク更新の方向と干渉の検証プロトコル

[English](lm-update-interaction-20261109.md) | 日本語

ユーザーの指示により、CST学習の複数の疑わしい挙動を検証します。監査済みの不変checkpoint、
本来のTinyStoriesラベル、幅512・4ブロックの同一モデル、固定operator0d277aeを使用。
Denseは比較対照です。探索的診断・DEV screeningであり、confirmationではありません。

Stage A：Dense、CST単独up・中央連続4・分散交互4・全8、DST全8の6モデルについて、
step976/4882から各3組の1024対象TRAIN batchを使い、独立に復元した36回の1更新反実仮想を
測定します。offset0は実際の次のshuffle batch、offset32/64は同じoptimizer clockで別batchを
使う診断です。20M schedule・Adam moment・clip・公開field LR scheduler・DSTの通常更新を
保持し、選択stepにDST topology eventはありません。

実際に保存した入力上で写像出力の変化を測り、タスクcotangentで重み付けした-sum(g*delta_y)、
下降方向cosine、相対出力変化を記録。cotangentにはaccumulation1/4を含みます。局所の一次近似で
あり、Dense再構成や異なるパラメータ化の同等性の主張ではありません。各写像だけ・Dense側の
残りだけ・全FFN・全up・全down・全Parameterを更新したときの同一batchのCEを評価します。
共同改善量と個別改善量の和との差は有限stepの非線形干渉で、その符号だけでは長期学習の悪い
協調を証明できません。CSTの振幅・半径割合・出力座標・入力座標の更新も別々に適用し、
境界・不変割合とfield方向を記録。周期座標の接線変化はwrapを考慮します。Dense側の小さい
方向変化を有限差分で照合（epsilon.01、絶対2e-4/相対12%の診断整合基準、native微分の基準は
変更なし）。各offset0では計測あり・なしの次stepのParameter/moment/counterが完全一致すること。
各仮更新後にParameter・buffer・optimizer・CPU/CUDA RNGを復元し、元checkpointは上書きしません。

Stage B：監査済みstep976から512更新の限定継続screening。Dense・DST対照と全8CSTの8設定を
同じ順序の学習データ・従来20M horizonで比較します。各手法の同一checkpoint/momentから開始。
CST倍率（振幅、割合、出力座標、入力座標）は(32,32,32,32)、(8,8,8,8)、(16,16,16,16)、
(64,64,64,64)、(8,32,32,32)、(32,8,8,8)、(32,8,32,32)、(32,32,8,8)。通常LRは固定。
公開の絶対field_lrsとscheduler anchorを使い、勾配・momentを再スケールしません。controlの
継続は元のstep1200/1400のDEV CE/top1を再現すること。全候補・TRAIN曲線・1200/1400/1488の
DEV CE/top1・最終checkpoint・moment/counter監査・geometry・全CST写像の独立oracleを保持。
同一状態からの継続介入であり、初期からのLR選定・最適recipeの証明ではありません。
長期の再学習は自動追加しません。

予約済みRTX4090で実GPU/runtime/source/data/orderを確認し、native68回帰testとcheckpointの
独立oracleを検証。直列実行し、新たな資源確保はしません。この診断の時間・peak memoryは
計測追加処理を含みbenchmarkの主張には使いません。永続NFSへ保存し、compactのhashを照合して
回収。Hub状態が曖昧・同じ失敗が反復した場合は再試行を止め、sourceや許容差を変えず部分結果を
保全。否定的・結論不明の結果も残し、公開runtime/既定値は変更しません。

証拠:CST_RESEARCH_EVIDENCE_ROOT/lm-update-interaction-4090-20261109-01。

## 実行後のユーザーによる目的の明確化

目的は改善探索ではなくCSTの挙動の分析であると明確化されました。訂正時の確認では、限定10条件のLR比較は既に終了していました。raw結果は別途保全し、recipeの選択・採用や追加の最適化実験はしません。[挙動分析レポート](lm-task-update-behavior-20261109-results.ja.md)では36組の復元型タスク更新probeと既存信号の証拠を分析します。上のStage B計画は実行履歴として保持します。
