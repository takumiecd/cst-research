# ゼロから徐々に開く境界のFFN短期比較

[English (canonical)](ffn-zero-opening-20261107.md) | 日本語

許可された3条件は tau=.01,.03,.1、初期 q=0 です。従来の有効な有理式境界
L0,U0 に対し `g=a²/(a²+tau²)`, `L=U0-g*(U0-L0)`, `U=U0` とし、
q の幾何補間を保ちます。ゼロで L=U、|a|=tau で相対的な隙間が半分です。
強振幅では従来区間へ近づき、amplitude_max 側の合流は強制しません。
minimum .5, birth4, maximum50, upper_floor1, a_max1, w_c.3 を保ちます。

[前回の短期条件](ffn-closed-envelope-20261107.ja.md)を揃えます。幅256、
pre-LN 4ブロック、FFN 8射影、FFN4N、c8、seed17、context128、batch2/累積4、
FP32/TF32無効、AdamW(.9,.95)、decay0、clip1、baseLR3e-4、Atom全体のLR16倍。
976更新/999424トークンで止め、LRとデータ順序は20Mトークン予定を維持します。
既存のハッシュ固定TinyStoriesとDEVのみを使用します。各条件の初期化を同じDense
二次モーメント（up std.02、down std.02/sqrt8）へ誤差.5%以内で再校正し、
clippingも保存します。到達不能なら棄却し、目標を黙って変えません。

GPU実行前にソース・データ・設定を固定します。初期Parameterのbyteとレポートの
一致を確認し、監査済みR12 Denseと従来q0の曲線を再利用、再学習しません。
各tauで両方向nativeのY/dX/全Atom勾配、独立求積、optimizerと実タスク再開を
検証してから学習します。学習済みcheckpoint監査、DEV再計算、取得検証も行います。

初期・更新1–5・20更新ごと・最終976の計55時点で、全8射影について現在振幅binの
個数、半径/qの平均・分位点、L/Uを保存します。up/down別の層統合は個数重みを使い、
bin所属は変化しえます。学習曲線と全体/振幅別の半径・qを重ねます。
従来q0には途中のbin履歴がないため、保存済み全体履歴とbin半径の終点だけを表示し、
欠測を明示します。過去の時間値は今回の同時対照測定ではありません。
1seed短期DEVだけで収束・最適tau・Atomスケールは主張しません。
confirmation/HPO、長期延長、自動での条件追加は行いません。
