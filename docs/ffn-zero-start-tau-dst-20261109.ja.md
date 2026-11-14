# fraction0のtau探索とTinyStoriesのDST対照

[English (canonical)](ffn-zero-start-tau-dst-20261109.md) | 日本語

ユーザーの指示で小さいtauを探索し、今後の比較図にDSTを含めます。過去のDSTはHIGGSで、
TinyStoriesの結果ではないため異なるタスクの曲線は混ぜません。tau={.0003,.001,.003,.01}、
初期fraction0、全CST field LR32倍、振幅上限32、幅512、4ブロック、FFN8行列各8192Atoms。
直前の0スタート計画のseed17・token順・976step/999424tokens・256step時点・data
manifest・公開moment初期化・RMS校正なし・AdamW betas(.9,.95)/eps1e-8/WD0/clip1・
warmup200・20M-token LR予定長・基準LR3e-4・FP32/TF32off・Dense Attention/
embedding/共有headを維持。Denseとtau.003も新規実行。CSTとDSTは別source固定job。

過去のHIGGS行数保存RigL-styleを長方形の研究consumerへ移植します(torchcst runtimeでは
ありません)。各FFNでK=4A=32768接続スロット、upは行あたり16、downは64。
bias/norm/外側はDense。名前付きDense LM初期値と独立seedの行別maskから開始し、
有効normal重みを1/sqrt(行密度)倍してDenseの目標二次momentを維持します。HIGGSの
uniform初期化を既存LM初期化へ合わせる変更です。Dense保存/行列積/mask/休眠座標の
Dense勾配を使う精度比較用で、疎kernelの速度/メモリは主張しません。新規成長のゼロも
スロットとして数え、Tensor.numelとは区別。有効FFN scalar+biasはCST272384と一致。

全体の絶対重み閾値でpruneし、各行に旧接続を最低1個残します。勾配で行別予算まで再成長し、
同点は小さい列indexを優先。新規重みは0、prune/成長(即時再成長を含む)のAdam momentは
リセットし、真の生存重み/momentは保存。cadence250、初期drop.3、cosine終点は
floor(.75*976)=732より手前で、eventは250/500。旧参照同様、eventでは全Parameterの
Adam更新を置き換えるため、976学習step/999424tokens、Adam更新974回です。
成長にはclip前の蓄積Dense勾配を使用。通常更新は無効勾配をmask後にglobal clip。Graphなし。

疎重みの最大LR={.0003,.001,.003,.01}のみ調整し、他は最大3e-4。4候補の初期model/mask
hashを一致させ、seed/データ/976step予算を揃え、Denseも新規実行。最終DEV CE最小、
同点なら低LRを選択。全候補を別図に残し、主図は選択DSTをDense/CSTへ重ねます。
開発用調整と明示し、held-out主張にはしません。confirmation・追加seed・自動延長なし。
今後の図でもsourceに結びついたDST対照と条件を保持します。

学習前のCPU/実GPUgate：独立した長方形Y/dX/有効dW/成長用生dW、無効重み/moment0、
行/全体予算、同点、prune/再成長resetと生存保存、eventの全Parameter optimizer時計停止、
通常/event次stepのserialize一致、全modelの有限loss/勾配/通常初期値一致、source/data
identity。CSTはbyte/GPU/runtime一致のoperator証拠だけ再利用し、全tauを境界0のresumeも
含め新規検査。学習後に最終/256checkpointとobserverを監査し、失敗とsource/result/
checkpoint hashを保全。compact取得検証と所有PID終了を確認。失敗再調整・gate緩和なし。

時間は転送・蓄積・forward/backward・clip・optimizerまたはtopology処理を含み、
probe/評価/observer/zero_grad/LR設定を除きます。topologyの時間は全step集計に含めつつ
別にも記録。準備/wall時間、PyTorch allocated/reserved peakを保存。逐次/別job速度比較は
探索的で、疎DST効率・収束・非劣性・普遍最適・default採用は主張しません。
