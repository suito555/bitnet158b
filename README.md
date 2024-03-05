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
実験1:matmal時のactivationとternizaed_weightのint8への変換<br>
少し試してみましたが、matmal時に"addmm_cuda" not implemented for 'Char'となってしまうため実装できません。CPUで動かしてみましたが、あまりの遅さで実験できませんでした。仕方ないのでroundをして四捨五入してみましたが、メモリ使用量は変化ありませんでした。(後にint8ではmatmal時に桁がはみ出ることに気づきましたが、このエラーは知っておいて損はないので書いておきます)<br>
<br><br>
補足1:torch.compile<br>
自作したGPTではtorch.compileはmatmal後に2の累乗以外をかけるとlossがnanになりました。ところがllamaだと問題ないようです。<br>
<br>
補足2:モデルサイズと学習率について<br>
モデルサイズを大きくするとともに、論文通りに、あるいはそれよりさらに学習率を下げたほうが良さそうです。<br>
<br>
補足3:上記のllamaのweightの保存時でエラーが出る<br>
epoch67892でBeomiさんのコードとの相性で止まってしまいますが、bitnetの感覚はつかめると思います。<br>
line 893あたりの保存設定を変えればなんとかなりそうですが、llamaに詳しくないため現在対処が難しいです。<br>
時間があったらその部分にも挑戦します。
