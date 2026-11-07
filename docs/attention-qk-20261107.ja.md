# CSTのQuery/Key射影への適用を調べる短期比較

[English (canonical)](attention-qk-20261107.md) | 日本語

CST-V比較の完了後、ユーザーがQとKの実験を許可しました。確保済みRTX4090で
新しいDense対照、QだけCST、KだけCSTを比較します。選んだ射影を全4ブロックで
置き換え、残る2射影、Attention出力、FFN、埋め込み、語彙出力はDenseです。
因果的scaled-dot-product/softmax演算はそのままにします。Q/Kそれぞれc8/各2048Atom、
256×256、有理式Polar、初期q0、Atom全体LR16倍です。

[V比較のタスクと予算](attention-v-20261107.ja.md)を揃えます。幅256、pre-LN4ブロック、
4head、context128、FFN4N、seed17、batch2/累積4、eager FP32/TF32無効、AdamW(.9,.95)、
decay0、clip1、baseLR3e-4、LR/データ順序20Mトークン予定、976更新/999424トークン、
同じハッシュ固定TinyStoriesとDEVです。confirmation/生成による選択、自動LR探索、
QKVを同時に置き換える実行は行いません。

残すDense射影の行は従来のpacked・名前ごとの乱数系列から厳密にコピーし、共有
Parameterを保持します。CST射影は同じDense std.02の二次モーメントへ誤差.5%以内で
再校正し、clippingも保存します。Q/Kごとに独立求積、実正方形nativeのY/dX/全Atom VJP、
明示的masked-softmaxとの統合勾配、未来token勾配ゼロ、optimizer、4ブロック実タスク/再開を
検証します。Denseも同じ凍結ソース/runtimeで新しく実行します。以前のV結果は元の
ソースに紐付けた過去の参考値です。

train/DEV曲線、射影ごとの振幅別半径/q、準備、同期した更新全体時間、allocated/reserved
ピークを保存します。診断は更新時間の外、計測付き総経過時間の中です。学習後の全対象射影、
DEV厳密再計算、最終geometry、次更新の再開を監査し、compact結果の取得・ハッシュを検証、
大きいcheckpointはリモートに保持します。1seed短期観測で収束・非劣性や普遍的改善を主張しません。
