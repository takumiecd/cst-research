# CSTのQKV・FFN同時置換の適用可能性を調べる短期比較

[English (canonical)](attention-combined-20261107.md) | 日本語

Q/K/V個別条件で学習が成立した後、ユーザーがQKV、続いてQKV＋FFNの実験を許可しました。
RTX4090で全Dense、全ブロックのQ/K/Vを独立CSTにしてFFNはDense、さらにFFNの両射影も
全ブロックでCSTにした3条件を新しく比較します。Attention出力O、LayerNorm、埋め込み、
語彙出力、因果的scaled-dot-product/softmaxはDenseまたは従来演算を保ちます。

[個別射影プロトコル](attention-qk-20261107.ja.md)を揃えます。幅256、pre-LN4ブロック、
4head、context128、FFN4N、c8、初期q0、seed17、batch2/累積4、FP32/TF32無効、
AdamW(.9,.95)、decay0、clip1、baseLR3e-4、Atom全体LR16倍、976更新/999424トークン、
LR/データ順序20Mトークン予定です。同じハッシュ固定TinyStoriesと8192-token DEVを使用し、
confirmation/生成による選択、LR探索、出力Oの置換、自動長期延長は行いません。

正方形Q/K/Vは各2048Atom、長方形FFNは各4096Atomです。QKVのみは12射影、
QKV＋FFNは20射影をCSTにします。従来の有理式Polar境界とDense二次モーメントを合わせる
初期化を使い、Q/K/VとFFN upはstd.02、FFN downはstd.02/sqrt8、誤差.5%以内とし、
clippingを保存します。共有の通常重みと射影名ごとのAtom乱数系列を厳密に保持します。

各モデル全体を別々に検証します。12射影のreceiptで20射影を許可せず、明示的な
(attention_kind,ffn_kind)と長方形・正方形の両ケースの一致を要求します。
学習前に独立求積、3つの正方形seedとFFN両方向のnative/全Atom VJP、明示的masked-softmax
とのY/dX/全QKV Atom勾配、未来token勾配ゼロ、optimizer state、全12/20射影の有限かつ
非ゼロの勾配を含む4ブロック実タスク/再開を確認します。初期化・正しさに失敗したら
棄却し、目標を黙って変更しません。

更新全体時間、準備、allocated/reservedピーク、train/DEV、射影ごとの振幅bin別半径/qを
保存し、診断は更新時間の外で行います。学習後の全射影、DEV厳密再計算、最終geometry、
次更新Parameter/optimizer stateの再開を監査し、compact証拠を取得・ハッシュ検証、
checkpointはリモートに保持します。以前の個別Q/K/Vは過去の参考値で、新しい速度対照では
ありません。1seed短期の適用可能性の観測であり、収束・正式な非劣性・Atom下限や普遍的な
速度/メモリ優位の主張ではありません。
