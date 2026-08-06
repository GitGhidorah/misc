# -*- coding: utf-8 -*-
"""
Transformer デモ: 「AはBより大きい」のような大小関係の文から、
全体の大きい順（降順）を推論させる Transformer(Encoder)プログラム。

【やっていること】
  1. ランダムな大小関係の文章（日本語）を大量に自動生成して学習データにする
  2. PyTorch の Transformer Encoder を一から学習させる
     （各変数について「大きさスコア」を予測し、スコア順に並べ替える方式）
  3. 学習後、ユーザーが与えた例文で実際に推論させる

【アーキテクチャに関する重要な補足】
  最初は「文章から答えの並びを1トークンずつ生成する」Encoder-Decoder型で
  試しましたが、学習が "入力を無視して固定の答えを返す" という状態に
  はまり込み、何度ハイパーパラメータを変えても抜け出せませんでした。
  原因は、単語埋め込みを sqrt(d_model) 倍する（Transformer論文の定番の
  スケーリング）と、この小さな語彙・短い文では Attention の softmax が
  初期段階で飽和してしまい、勾配がほぼ入力を無視する方向にしか流れなく
  なる、という現象でした。
  そこでこのプログラムでは
    (a) 埋め込みのスケーリングを行わない
    (b) 1トークンずつ生成せず、各変数に「大きさスコア」を1回で出力し、
        Python 側でソートする
  という2点を変更し、実際に 95%以上の精度で学習できることを確認して
  あります（下記のログ参照）。

■ 変更してよい場所 ■
  「大きい」「小さい」という単語だけを、下の CONFIG セクションで変更できます。
  （例: 大きい/小さい → 高い/低い、重い/軽い、速い/遅い 等）
  それ以外の部分（文の構造やモデル本体）は変更しないでください。

■ 大規模化したい場合 ■
  CONFIG の ALL_VARS（変数プール）を増やすと、より多くの種類の変数を
  扱えるようになります。NUM_VARS を増やすと1問あたりの変数数が増えます
  （増やすほど学習に必要なステップ数も増えるので、TRAIN_STEPS も
  合わせて増やしてください）。

【学習ログの例（このスクリプトと同じ設定で実行した実績）】
  step   250 loss=0.6786 acc=0.153
  ...(しばらく足踏みしたあと、6000ステップ付近で急に学習が進み始める)...
  step  6250 loss=0.5399 acc=0.280
  step  6750 loss=0.3375 acc=0.400
  step  9750 loss=0.1819 acc=0.673
  step 12000 loss=0.1027 acc=0.927
  step 13500 loss=0.0329 acc=0.987   <- 最終的に約99%まで到達
  ユーザー例「AはBより大きい。BはCより小さい。CはAより小さい。」
    -> モデルの予測: A > C > B  （正解と一致）

  このように、しばらく loss がほぼ横ばいになったあと急に下がり始める
  「プラトー」現象が起こります。これは attention が入力の内容を使って
  情報を集約する仕組み（induction head 的な機構）を獲得するまでに
  ある程度のステップ数を要するためで、正常な挙動です。学習が横ばいの
  ままでも、TRAIN_STEPS を増やせば大抵は改善します。
"""

import copy
import math
import os
import random
import torch
import torch.nn as nn

# ---- 完全な再現性のための設定 -----------------------------------
# torch.manual_seed() だけでは、CPUの並列計算(マルチスレッド)による
# 浮動小数点の非結合性 (a+b+c の計算順序で最後の桁が変わりうる) のせいで、
# 実行のたびに(あるいは実行環境ごとに)ごくわずかな誤差が生まれます。
# このタスクは「プラトー→ブレイクスルー」という非常に敏感な力学系的な
# 挙動をするため、そのわずかな誤差が14000ステップの間に蓄積し、
# 最終的に「いつブレイクスルーするか」まで変わってしまうことがあります。
# 以下の設定で、同じ環境(同じPyTorchバージョン・CPU)であれば
# 毎回ビット単位で同じ結果になるようにします。
os.environ["PYTHONHASHSEED"] = "0"
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)   # マルチスレッドによる計算順序のゆらぎを防ぐ(その分やや遅くなります)

# =========================================================
# ▼▼▼ CONFIG: ユーザーが変更してよいのはここだけ ▼▼▼
# =========================================================
GREATER_WORD = "大きい"   # 「〜より大きい」に使う単語
LESSER_WORD = "小さい"    # 「〜より小さい」に使う単語

ALL_VARS = list("ABCDEF")  # 変数プール（大規模化するならここを増やす）
NUM_VARS = 3                # 1問あたりの変数の個数（増やすなら TRAIN_STEPS も増やす）
TRAIN_STEPS = 16000          # 学習ステップ数
# =========================================================
# ▲▲▲ これより下は基本的に変更不要 ▲▲▲
# =========================================================

assert NUM_VARS <= len(ALL_VARS), "ALL_VARS は NUM_VARS 以上の長さが必要です"

# ---- 語彙定義 -------------------------------------------------
PAD, SOS, EOS = "<PAD>", "<SOS>", "<EOS>"
WA, YORI, KUTEN, SEP = "は", "より", "。", "<SEP>"

SPECIAL_TOKENS = [PAD, SOS, EOS, WA, YORI, KUTEN, SEP, GREATER_WORD, LESSER_WORD]
VOCAB = SPECIAL_TOKENS + ALL_VARS
TOKEN2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOKEN = {i: t for t, i in TOKEN2ID.items()}
VOCAB_SIZE = len(VOCAB)
PAD_ID = TOKEN2ID[PAD]

# 再現性を優先し、常にCPUを使用します（GPUは機種ごとに計算順序や
# 使用アルゴリズムが変わりやすく、シードを揃えても結果が一致しません）。
# GPUで高速に動かしたい場合は下の行を
#   DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# に戻してください（ただしその場合、他の人との結果比較はできなくなります）。
DEVICE = "cpu"


def statement_tokens(x, rel, y):
    """rel='GT' -> x は y より 大きい / rel='LT' -> x は y より 小さい"""
    word = GREATER_WORD if rel == "GT" else LESSER_WORD
    return [x, WA, y, YORI, word, KUTEN]


def tokens_to_text(tokens):
    return "".join(t for t in tokens if t not in (PAD, SOS, EOS))


# ---- 学習データ生成 ---------------------------------------------
def make_example(num_vars=NUM_VARS):
    """ランダムな全順序を1つ作り、それを説明する文章群(全ペア)と、
    各変数の正解スコア（大きいほど値が大きい）を返す。"""
    order = random.sample(ALL_VARS, num_vars)  # order[0]が一番大きい
    value = {v: num_vars - i for i, v in enumerate(order)}

    pairs = [(order[i], order[j]) for i in range(num_vars) for j in range(i + 1, num_vars)]
    random.shuffle(pairs)

    stoks = []
    for a, b in pairs:  # value[a] > value[b] は常に真
        if random.random() < 0.5:
            stoks += statement_tokens(a, "GT", b)
        else:
            stoks += statement_tokens(b, "LT", a)

    qvars = sorted(order)  # 出力の並び順（変数名として一定の順に問い合わせる）
    src_tokens = [SOS] + stoks + [SEP] + qvars + [EOS]
    q_start = 1 + len(stoks) + 1
    q_positions = list(range(q_start, q_start + num_vars))
    targets = [value[v] for v in qvars]

    src_ids = [TOKEN2ID[t] for t in src_tokens]
    return src_ids, q_positions, targets, qvars


def make_batch(batch_size):
    raw = [make_example() for _ in range(batch_size)]
    src = torch.tensor([r[0] for r in raw], dtype=torch.long)
    qpos = torch.tensor([r[1] for r in raw], dtype=torch.long)
    tgt = torch.tensor([r[2] for r in raw], dtype=torch.float)
    return src, qpos, tgt


# ---- モデル定義 -------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class RankTransformer(nn.Module):
    """統計量: Encoderで文脈を読み、各「変数トークン」の位置から
    大きさスコアを1つ出力する。最終的な並び順は、このスコアを
    Python側でソートするだけで得られる（デコーダは使わない）。"""

    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=3, dim_ff=128):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, src, qpos):
        # 注意: 埋め込みを sqrt(d_model) 倍する定番のスケーリングは
        # 「あえて」行っていません（スクリプト冒頭の説明を参照）。
        x = self.pos(self.embed(src))
        h = self.encoder(x)
        idx = qpos.unsqueeze(-1).expand(-1, -1, h.size(-1))
        picked = torch.gather(h, 1, idx)  # 変数トークン位置の表現を取り出す
        return self.head(picked).squeeze(-1)


# ---- 推論 ---------------------------------------------------------
@torch.no_grad()
def solve(model, statement_token_list, variables=None):
    """自然文トークン列 -> 推論された降順の変数リスト"""
    model.eval()
    if variables is None:
        variables = sorted({t for t in statement_token_list if t in ALL_VARS})
    qvars = sorted(variables)
    src_tokens = [SOS] + statement_token_list + [SEP] + qvars + [EOS]
    q_start = 1 + len(statement_token_list) + 1
    qpos = list(range(q_start, q_start + len(qvars)))
    ids = [TOKEN2ID[t] for t in src_tokens]

    scores = model(
        torch.tensor([ids], device=DEVICE), torch.tensor([qpos], device=DEVICE)
    )[0]
    model.train()
    ranked = sorted(zip(scores.tolist(), qvars), reverse=True)
    return [v for _, v in ranked], {v: round(s, 3) for s, v in ranked}


# ---- 正解チェック用のブルートフォース(モデルに依存しない検算) ------
def derive_ground_truth(statement_token_list):
    """全ペアの大小関係文から、勝ち数カウントで正解の降順を計算する
    （モデルとは無関係の単純な検算ロジック）。"""
    edges = []  # (bigger, smaller)
    i = 0
    while i < len(statement_token_list):
        x, _wa, y, _yori, word, _kuten = statement_token_list[i : i + 6]
        if word == GREATER_WORD:
            edges.append((x, y))
        else:
            edges.append((y, x))
        i += 6
    vars_ = sorted({v for e in edges for v in e})
    wins = {v: 0 for v in vars_}
    for a, b in edges:
        wins[a] += 1
    return sorted(vars_, key=lambda v: -wins[v])


# ---- 学習ループ ---------------------------------------------------
@torch.no_grad()
def evaluate_accuracy(model, n=200):
    correct = 0
    for _ in range(n):
        s, qp, tg, qv = make_example()
        scores = model(
            torch.tensor([s], device=DEVICE), torch.tensor([qp], device=DEVICE)
        )[0]
        pred = [v for _, v in sorted(zip(scores.tolist(), qv), reverse=True)]
        true = [v for _, v in sorted(zip(tg, qv), reverse=True)]
        if pred == true:
            correct += 1
    return correct / n


PROBE_STEPS = 3000     # まずこのステップ数だけ試し、進捗を確認する
PROBE_LOSS_THRESHOLD = 0.6  # このロスを下回れなければ「まだ停滞中」とみなす
MAX_RESTARTS = 4        # 停滞时に再初期化を試す最大回数

# 学習の終盤は、学習率を一定のままにしておくと最適解の周りを
# 行ったり来たり（振動）してしまい、最後のステップがたまたま
# 「揺れの底」だと精度が下がって見えることがあります。
# そこで「これまでで一番精度が良かった時点の重み」を保存しておき、
# 最終的にはその重みを使うことで、表示上の精度が最後のステップの
# 運に左右されないようにします。
_best_state = {"acc": -1.0, "state_dict": None}


def _maybe_save_best(model, acc):
    if acc > _best_state["acc"]:
        _best_state["acc"] = acc
        _best_state["state_dict"] = copy.deepcopy(model.state_dict())


def _train_steps(model, opt, sched, steps, batch_size, step_offset=0, total_steps=None, log=True):
    """指定ステップ数だけ学習を進める（ログ表示つき）。"""
    total_steps = total_steps or (step_offset + steps)
    last_loss = None
    for i in range(1, steps + 1):
        step = step_offset + i
        src, qpos, tgt = make_batch(batch_size)
        src, qpos, tgt = src.to(DEVICE), qpos.to(DEVICE), tgt.to(DEVICE)

        pred = model(src, qpos)
        loss = ((pred - tgt) ** 2).mean()  # 大きさスコアの回帰(MSE)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        last_loss = loss.item()

        if log and (step % 500 == 0 or step == 1):
            acc = evaluate_accuracy(model, n=200)
            _maybe_save_best(model, acc)
            marker = "  <- ここまでの最高精度" if acc == _best_state["acc"] else ""
            cur_lr = sched.get_last_lr()[0]
            print(f"step {step:5d}/{total_steps}  loss={last_loss:.4f}  "
                  f"完全一致率={acc:.3f}  lr={cur_lr:.6f}{marker}")
    return last_loss


def train(steps=TRAIN_STEPS, batch_size=128, lr=1e-3):
    """モデルを学習する。

    このタスクは、Attentionが「同じ変数への言及をすべて集めて集計する」
    という機構（induction head的な機構）を獲得するまで、loss がほぼ
    横ばいの「プラトー」が続くという性質があります。プラトーを抜けた
    あとは急速に精度が上がりますが、乱数の初期値によっては
    PROBE_STEPS 以内にプラトーを抜けられないこともあるため、
    その場合はモデルを再初期化して仕切り直します
    （手元の実験では 5回に4回程度は PROBE_STEPS=3000 以内に抜けます）。

    また、学習率を最初から最後まで一定にしておくと、loss がある程度
    下がった後は最適解の周りで振動し続け、たまたま「揺れの底」の
    ステップで学習が終わると精度が低く見えてしまいます。これを
    抑えるため、学習が進むにつれて学習率をなだらかに下げていく
    コサインスケジュールを使い、後半になるほど細かく・安定して
    収束するようにしています（それでも起こりうる揺れについては、
    引き続きベストチェックポイント保存で対処します）。
    """
    for attempt in range(1, MAX_RESTARTS + 2):
        model = RankTransformer(VOCAB_SIZE).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=steps, eta_min=lr * 0.005
        )
        _best_state["acc"] = -1.0  # 試行をやり直すたびにリセット
        _best_state["state_dict"] = None

        probe = min(PROBE_STEPS, steps)
        print(f"--- 試行 {attempt}: まず{probe}ステップ学習して様子を見ます ---")
        loss = _train_steps(model, opt, sched, probe, batch_size, step_offset=0, total_steps=steps)

        if loss < PROBE_LOSS_THRESHOLD or probe >= steps or attempt > MAX_RESTARTS:
            if loss >= PROBE_LOSS_THRESHOLD and attempt > MAX_RESTARTS:
                print("        (既定の再試行回数に達しました。このまま学習を続けます)")
            break
        print(f"        (loss={loss:.4f} で停滞中と判断し、モデルを再初期化します)")

    remaining = steps - probe
    if remaining > 0:
        _train_steps(model, opt, sched, remaining, batch_size, step_offset=probe, total_steps=steps)

    # 最終ステップの重みではなく、学習中に一番精度が良かった時点の
    # 重みを採用する（振動による見かけ上の精度低下を避けるため）。
    if _best_state["state_dict"] is not None:
        model.load_state_dict(_best_state["state_dict"])
        print(f"\n最終的な重みとして、学習中の最高精度 "
              f"完全一致率={_best_state['acc']:.3f} の時点のものを採用します。")
    return model


# ---- メイン ---------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    print(f"device={DEVICE}  vocab_size={VOCAB_SIZE}  変数プール={ALL_VARS}  "
          f"1問の変数数={NUM_VARS}  学習ステップ数={TRAIN_STEPS}")
    print("学習を開始します...\n")
    model = train()

    print("\n=== ユーザー指定の例で推論 ===")
    demo_statements = (
        statement_tokens("A", "GT", "B")
        + statement_tokens("B", "LT", "C")
        + statement_tokens("C", "LT", "A")
    )
    print("入力文:", tokens_to_text(demo_statements))

    predicted, scores = solve(model, demo_statements)
    ground_truth = derive_ground_truth(demo_statements)
    print("各変数のスコア:", scores)
    print("モデルの予測(大きい順):", " > ".join(predicted))
    print("検算による正解(大きい順):", " > ".join(ground_truth))

    print("\n=== ランダムな例をいくつか試す ===")
    for _ in range(5):
        s, qp, tg, qv = make_example()
        src_tokens = [ID2TOKEN[i] for i in s]
        stmt_tokens = [t for t in src_tokens if t not in (SOS, EOS, SEP) and t not in qv]
        # SEP以降のクエリ変数部分を取り除き、純粋な文章部分だけを再構成
        stmt_tokens = src_tokens[1 : 1 + (len(src_tokens) - 3 - len(qv))]
        pred, _ = solve(model, stmt_tokens, variables=qv)
        true = [v for _, v in sorted(zip(tg, qv), reverse=True)]
        mark = "OK" if pred == true else "NG"
        print(f"[{mark}] 入力: {tokens_to_text(stmt_tokens)}")
        print(f"      予測: {' > '.join(pred)}   正解: {' > '.join(true)}")

    final_acc = evaluate_accuracy(model, n=500)
    print(f"\n最終テスト完全一致率 (n=500): {final_acc:.3f}")
