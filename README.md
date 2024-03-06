論文に書かれていたBitNet1.58bのBitLinear部の実装です。<br>
https://arxiv.org/abs/2310.11453<br>
https://arxiv.org/abs/2402.17764<br>
<br>
論文を読み自力で書いたところもありますが、以下の方々のコードが非常に参考になりました。<br>
https://github.com/Beomi/BitNet-Transformers<br>
https://github.com/frodo821/BitNet-Transformers<br>
https://github.com/kyegomez/BitNet<br>
<br>
参考にさせてもらったコードから大きな変更点としては、activationの順番とdeqauntizeの部分を直しています。<br>
また、Beomiさんのコードのllamaに追加してみたところ、epoch60000でlossは3.7付近でした。<br>
<br>
<br>
補足1:torch.compile<br>
自作したGPTではtorch.compileはmatmal後に2の累乗以外をかけるとlossがnanになりました。ところがllamaだと問題ないようです。<br>
<br>
補足2:モデルサイズと学習率について<br>
モデルサイズを大きくするとともに、論文通りに、あるいはそれよりさらに学習率を下げたほうが良さそうです。<br>
<br>
補足3:上記のllamaのweightの保存時でエラーが出る<br>
epoch67892でBeomiさんのコードとの相性で止まってしまいますが、bitnetの感覚はつかめると思います。<br>
line 893あたりの保存設定を変えればなんとかなりそうですが、llamaに詳しくないため現在対処が難しいです。<br>
気が向いたらその部分にも挑戦します。<br>
<br>
補足4:epsの値<br>
この値は判断が難しく、nanにならなければ小さくしたほうがいいのですが、0付近の誤差は調べても判断がつかなかったので小さく出たabs_meanの10000分の1ぐらいにしました。nanが出るようなら少し大きくしてください(具体的にはfloat16なら1e-6にすると良いです。bfloatだとそのままで大丈夫だと思います)<br>
<br>
補足5:abs_max_x_valueにepsを足している<br>
正直この値が0になることはないと思うのですが、念の為足しておきました。消しても大丈夫だと思います。<br>
<br>
補足6:quantize_weightsのtorch.sign()<br>
そのままだとquantize_weightsに-0が入ってしまう形だったので、安定を取るためにsignを入れました。何もなければ外して構いません。<br>
<br>
補足7:scaled_xにtorch.round()がない<br>
・clamp時のepsの効果がどうなるのかが自信がなかった<br>
・丸めた際に前と後ろの接続がどうなっているのか判断がつかなかった<br>
・そもそもの入力が分からず、結局matmul(int8,bit1.58)で何をしているのか分からなくなってしまった(int8を普通に越える)<br>
3つ目が特に大きな理由で、実際にroundしてみたり、関係ないと思いつつSTEらしきことをしたのですが結果がまるで変わらなかったので外しています。追記:Q_bを変えれば意味があったので実際にやってみたものを実験3に入れました。<br>
<br>
<br>
実験1:matmulの前でactivationとternizaed_weightのint8への変換<br>
matmul時に"addmm_cuda" not implemented for 'Char'となってしまうため実装できません。CPUで動かしてみましたが、あまりの遅さで実験できませんでした。(後にint8ではmatmul時に桁がはみ出ることに気づきましたが、このエラーは知っておいて損はないので書いておきます)<br>
<br>
実験2:matmul時のactivationとternizaed_weightのbf16を四捨五入し整数にする<br>
最適化に期待して、torch.roundによりbf16の値を四捨五入してみましたが、メモリ使用量に変化ありませんでした。<br>
<br>
実験3:roundの有無、Q_bの値の変化による検証<br>
かなり早めのiterで取りました。個人的にはGPTの場合はlossの値がきれいなのでこれでも参考になると思います。気になる方は検証してみてください。今回はlossが9.0を割ったiterで止めています。(bf16,param=35.66M)<br>
<br>
activationにround無し<br>
Q_b=32 iter 18: loss 8.9687<br>
Q_b=16 iter 18: loss 8.9691<br>
Q_b=8 iter 18: loss 8.9687<br>
Q_b=4 iter 18: loss 8.9688<br>
Q_b=2 iter 18: loss 8.9689<br>
Q_b=1 iter 18: loss 8.9694<br>
<br>
activationにround追加<br>
Q_b=32 iter 18: loss 8.9855<br>
Q_b=16 iter 18: loss 8.9834<br>
Q_b=8 iter 18: loss 8.9839<br>
Q_b=4 iter 18: loss 8.9871<br>
Q_b=2 iter 18: loss 9.0197<br>
Q_b=1 iter 18: loss 9.3447<br>
