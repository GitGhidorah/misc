"""
多段階情報伝播による照応解決デモ (v4: FFNに意味的機能を持たせた版)

【v3との違い】
v3のFFN: 単なる増幅器 (scale=3.5倍するだけ)

v4のFFN: Key-Value記憶として機能
  FFN1: 「前者フラグが立っている → りんご情報を書き込め」
        「後者フラグが立っている → オレンジ情報を書き込め」
        というパターン→変換の知識を保持

  FFN2: 「記号フラグが立っていて、りんご情報が来た → りんごスコアを強化」
        「記号フラグが立っていて、オレンジ情報が来た → オレンジスコアを強化」
        という条件付き変換を実行

【Key-Value記憶としてのFFN (Geva et al. 2021)】
  W1の各行 = キー   (どんな入力パターンを検出するか)
  W2の各列 = バリュー (検出したら何を出力するか)
  ReLU     = パターンが一致したニューロンだけ発火
"""

import numpy as np
np.set_printoptions(precision=3, suppress=True)

tokens = ["りんご", "オレンジ", "。", "前者", "を", "A", "後者", "を", "B", "。"]
T = len(tokens)
D = 8

print("=" * 65)
print("トークン列:", " / ".join(f"{i}:{t}" for i, t in enumerate(tokens)))

# ============================================================
# 埋め込み (v3と同じ)
# dim0: りんごフラグ
# dim1: オレンジフラグ
# dim2: 前者フラグ
# dim3: 後者フラグ
# dim4: 記号フラグ (A, B)
# dim5: 文境界
# dim6: 位置
# dim7: 指示語フラグ
# ============================================================
positions = np.arange(T) * 0.1
X0 = np.zeros((T, D))
X0[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, positions[0], 0.0]  # りんご
X0[1] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, positions[1], 0.0]  # オレンジ
X0[2] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, positions[2], 0.0]  # 。
X0[3] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, positions[3], 1.0]  # 前者
X0[4] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, positions[4], 0.0]  # を
X0[5] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, positions[5], 0.0]  # A
X0[6] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, positions[6], 1.0]  # 後者
X0[7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, positions[7], 0.0]  # を
X0[8] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, positions[8], 0.0]  # B
X0[9] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, positions[9], 0.0]  # 。

print("\n【初期埋め込み X0】")
print("  次元:[りんご, オレンジ, 前者語, 後者語, 記号, 文境界, 位置, 指示語]")
for i, t in enumerate(tokens):
    print(f"  {t:6s}(pos{i}): {X0[i]}")

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def attention_with_mask(X, Wq, Wk, Wv, mask=None):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores += mask
    W = softmax(scores)
    return W @ V, W

def ffn_kv(X, W1, b1, W2):
    """
    Key-Value記憶としてのFFN
    W1: (D, H) - キー行列  (H個のニューロン=H個のパターン検出器)
    W2: (H, D) - バリュー行列 (各ニューロンが発火したとき何を出力するか)
    処理: hidden = ReLU(X @ W1 + b1)  ← パターンマッチ
          out    = hidden @ W2         ← バリュー読み出し
    """
    hidden = np.maximum(0, X @ W1 + b1)
    out = hidden @ W2
    return out, hidden  # hiddenも返して可視化に使う

def show_attn(W, label, rows):
    print(f"\n  [{label}]")
    print(f"  {'':8s}", end="")
    for t in tokens: print(f"{t:8s}", end="")
    print()
    for i in rows:
        t = tokens[i]
        print(f"  {t:8s}", end="")
        for j in range(T):
            w = W[i, j]
            s = f"{w:.3f}"
            if   w > 0.50: s = f"\033[1;32m{w:.3f}\033[0m"
            elif w > 0.10: s = f"\033[33m{w:.3f}\033[0m"
            print(f"{s:8s}", end="")
        print()

# ============================================================
# LAYER 1
# アテンション: 「前者/後者」が「りんご/オレンジ」の位置情報を収集
#   → ただしV投影はdim2/dim3(指示語フラグ)のみ渡す
#   → 「誰に注目したか」だけをattn出力に乗せる
#
# FFN1: アテンション出力を受け取り、
#   「前者フラグが来た + 果物情報あり → りんごスコアを書き込む」
#   「後者フラグが来た + 果物情報あり → オレンジスコアを書き込む」
#   というKey-Value変換を実行
# ============================================================
print("\n" + "=" * 65)
print("【LAYER 1】")
print("  アテンション: 前者/後者が りんご/オレンジ の存在を検出")
print("  FFN1 (Key-Value記憶): 検出結果を果物スコアに変換して書き込む")

# --- アテンション ---
Wq1 = np.zeros((D, 2))
Wq1[2, 0] = 8.0   # 前者フラグ → Q[0]
Wq1[3, 1] = 8.0   # 後者フラグ → Q[1]

Wk1 = np.zeros((D, 2))
Wk1[0, 0] = 8.0   # りんごフラグ → K[0]
Wk1[1, 1] = 8.0   # オレンジフラグ → K[1]

# V: dim2(前者フラグ), dim3(後者フラグ) を渡す
# → attn出力には「どの指示語がどの果物に注目したか」の情報が入る
Wv1 = np.zeros((D, D))
Wv1[2, 2] = 1.0   # 前者フラグをV経由で渡す
Wv1[3, 3] = 1.0   # 後者フラグをV経由で渡す
Wv1[0, 0] = 0.5   # 果物情報も少し渡す（FFNへの入力に使う）
Wv1[1, 1] = 0.5

out_attn1, W1 = attention_with_mask(X0, Wq1, Wk1, Wv1)
show_attn(W1, "Layer1 アテンション重み", rows=[3, 6])

# アテンション後に元の埋め込みと合算（残差）
X_after_attn1 = X0 + out_attn1

print("\n  【アテンション後 (残差込み) - 前者・後者の状態】")
print("  次元:[りんご, オレンジ, 前者語, 後者語, 記号, 文境界, 位置, 指示語]")
for i in [3, 6]:
    print(f"  {tokens[i]:6s}: {X_after_attn1[i]}")

# --- FFN1: Key-Value記憶 ---
# ニューロン(隠れ層)の設計:
#   H=4 のニューロンを使う
#
#   ニューロン0 (キー): 「前者フラグ(dim2) が大きい」パターンを検出
#                (バリュー): dim0(りんごスコア) に大きな値を出力
#
#   ニューロン1 (キー): 「後者フラグ(dim3) が大きい」パターンを検出
#                (バリュー): dim1(オレンジスコア) に大きな値を出力
#
#   ニューロン2,3: 他のトークン用（今回はほぼ不活性）

H1 = 4
# W1_ffn: キー行列 (D, H) - 各ニューロンが「何を検出するか」
W1_ffn1 = np.zeros((D, H1))
W1_ffn1[2, 0] = 6.0   # ニューロン0のキー: dim2(前者フラグ)に強く反応
W1_ffn1[0, 0] = 2.0   # ニューロン0のキー: dim0(果物情報)も補助的に使う
W1_ffn1[3, 1] = 6.0   # ニューロン1のキー: dim3(後者フラグ)に強く反応
W1_ffn1[1, 1] = 2.0   # ニューロン1のキー: dim1(果物情報)も補助的に使う

# バイアス: 閾値として機能（これ以上のスコアでないと発火しない）
b1_ffn1 = np.array([-3.0, -3.0, -1.0, -1.0])

# W2_ffn: バリュー行列 (H, D) - 各ニューロンが発火したとき何を出力するか
W2_ffn1 = np.zeros((H1, D))
W2_ffn1[0, 0] = 4.0   # ニューロン0発火 → dim0(りんごスコア)を出力
W2_ffn1[1, 1] = 4.0   # ニューロン1発火 → dim1(オレンジスコア)を出力

ffn1_out, ffn1_hidden = ffn_kv(X_after_attn1, W1_ffn1, b1_ffn1, W2_ffn1)

print("\n  【FFN1 内部状態（Key-Value記憶の動作）】")
print("  各トークンのニューロン発火パターン:")
print(f"  {'':8s}  ニューロン0      ニューロン1      ニューロン2      ニューロン3")
print(f"  {'':8s}  (前者→りんご検出) (後者→オレンジ検出)")
for i, t in enumerate(tokens):
    h = ffn1_hidden[i]
    if np.any(h > 0.01):
        bars = []
        for val in h:
            bar = "█" * int(val * 2) if val > 0 else "─"
            bars.append(f"{val:5.2f}{bar:6s}")
        print(f"  {t:8s}: {'  '.join(bars)}")
    else:
        print(f"  {t:8s}: (不活性)")

X1 = X_after_attn1 + ffn1_out

print("\n  【Layer1後 X1 (前者・後者の変化)】")
for i in [0, 1, 3, 6]:
    t = tokens[i]
    v = X1[i]
    note = ""
    if t == "前者": note = f"  ← FFN1がdim0(りんご)={v[0]:.2f}を書き込んだ"
    if t == "後者": note = f"  ← FFN1がdim1(オレンジ)={v[1]:.2f}を書き込んだ"
    print(f"  {t:6s}: {v}{note}")

# ============================================================
# LAYER 2
# アテンション: A/Bが直前の指示語（前者/後者）に注目
# FFN2: 「記号フラグ + りんご情報 → りんごスコア強化」
#        「記号フラグ + オレンジ情報 → オレンジスコア強化」
# ============================================================
print("\n" + "=" * 65)
print("【LAYER 2】")
print("  アテンション: A/Bが直前の指示語(前者/後者)の情報を収集")
print("  FFN2 (Key-Value記憶): 記号+果物情報の組み合わせを最終スコアに変換")

# --- アテンション ---
Wq2 = np.zeros((D, 2))
Wq2[4, 0] = 5.0   # 記号フラグ → Q[0]
Wq2[6, 1] = 1.0   # 位置 → Q[1]

Wk2 = np.zeros((D, 2))
Wk2[7, 0] = 5.0   # 指示語フラグ → K[0]
Wk2[6, 1] = 1.0   # 位置 → K[1]

Wv2 = np.zeros((D, D))
Wv2[0, 0] = 1.0   # りんご情報を転写
Wv2[1, 1] = 1.0   # オレンジ情報を転写
Wv2[4, 4] = 0.5   # 記号フラグも保持

pos = np.arange(T) * 1.0
dist_mask = np.zeros((T, T))
for i in range(T):
    for j in range(T):
        dist = i - j
        if dist <= 0:
            dist_mask[i, j] = -1e9
        else:
            dist_mask[i, j] = -dist * 3.0

out_attn2, W2 = attention_with_mask(X1, Wq2, Wk2, Wv2, mask=dist_mask)
show_attn(W2, "Layer2 アテンション重み", rows=[5, 8])

X_after_attn2 = X1 + out_attn2

print("\n  【アテンション後 (残差込み) - A・Bの状態】")
for i in [5, 8]:
    print(f"  {tokens[i]:6s}: {X_after_attn2[i]}")

# --- FFN2: Key-Value記憶 ---
# ニューロン設計:
#   ニューロン0: キー「記号フラグ(dim4)が高い AND りんごスコア(dim0)が高い」
#               バリュー: dim0(りんごスコア)をさらに増幅
#
#   ニューロン1: キー「記号フラグ(dim4)が高い AND オレンジスコア(dim1)が高い」
#               バリュー: dim1(オレンジスコア)をさらに増幅
#
#   これが「前者をA」パターンの知識 = FFNの記憶

H2 = 4
W1_ffn2 = np.zeros((D, H2))
W1_ffn2[4, 0] = 3.0   # ニューロン0キー: 記号フラグ
W1_ffn2[0, 0] = 3.0   # ニューロン0キー: りんごスコア (AND条件)
W1_ffn2[4, 1] = 3.0   # ニューロン1キー: 記号フラグ
W1_ffn2[1, 1] = 3.0   # ニューロン1キー: オレンジスコア (AND条件)

b1_ffn2 = np.array([-4.0, -4.0, -1.0, -1.0])

W2_ffn2 = np.zeros((H2, D))
W2_ffn2[0, 0] = 5.0   # ニューロン0発火 → りんごスコアを大きく出力
W2_ffn2[1, 1] = 5.0   # ニューロン1発火 → オレンジスコアを大きく出力

ffn2_out, ffn2_hidden = ffn_kv(X_after_attn2, W1_ffn2, b1_ffn2, W2_ffn2)

print("\n  【FFN2 内部状態（Key-Value記憶の動作）】")
print("  各トークンのニューロン発火パターン:")
print(f"  {'':8s}  ニューロン0          ニューロン1")
print(f"  {'':8s}  (記号+りんご→強化)   (記号+オレンジ→強化)")
for i, t in enumerate(tokens):
    h = ffn2_hidden[i]
    if np.any(h > 0.01):
        n0 = h[0]; n1 = h[1]
        bar0 = "█" * int(n0 * 1.5)
        bar1 = "█" * int(n1 * 1.5)
        print(f"  {t:8s}: ニューロン0={n0:6.2f} {bar0:12s}  ニューロン1={n1:6.2f} {bar1}")
    else:
        print(f"  {t:8s}: (不活性)")

X2 = X_after_attn2 + ffn2_out

print("\n  【Layer2後 最終隠れ状態 X2】")
for i in [3, 5, 6, 8]:
    t = tokens[i]
    v = X2[i]
    note = ""
    if t == "A": note = f"  ← りんご={v[0]:.2f}, オレンジ={v[1]:.2f}"
    if t == "B": note = f"  ← りんご={v[0]:.2f}, オレンジ={v[1]:.2f}"
    print(f"  {t:6s}: {v}{note}")

# ============================================================
# 最終検証
# ============================================================
print("\n" + "=" * 65)
print("【最終検証】\n")

checks = [
    ("前者", 3, 0, 1, "りんご",   "オレンジ"),
    ("後者", 6, 1, 0, "オレンジ", "りんご"),
    ("A",    5, 0, 1, "りんご",   "オレンジ"),
    ("B",    8, 1, 0, "オレンジ", "りんご"),
]

all_ok = True
for name, idx, dim_t, dim_o, target, other in checks:
    v = X2[idx]
    s_t, s_o = v[dim_t], v[dim_o]
    ok = s_t > s_o
    bar_t = "█" * min(int(s_t * 3), 40)
    bar_o = "█" * min(int(s_o * 3), 40)
    status = "✅ OK" if ok else "❌ NG"
    print(f"  {status}  「{name}」")
    print(f"         {target:6s}スコア = {s_t:6.3f}  {bar_t}")
    print(f"         {other:6s}スコア = {s_o:6.3f}  {bar_o}")
    print()
    if not ok: all_ok = False

print("=" * 65)
if all_ok:
    print("✅ 全照応解決 成功！")
else:
    print("❌ 一部失敗")

print("""
【FFNのKey-Value記憶としての役割まとめ】

  FFN1 (Layer1):
    ニューロン0: キー「dim2(前者フラグ)が高い」を検出
                バリュー「dim0(りんごスコア)を出力」
    ニューロン1: キー「dim3(後者フラグ)が高い」を検出
                バリュー「dim1(オレンジスコア)を出力」
    → 前者の隠れ状態にりんごスコアが書き込まれる
    → 後者の隠れ状態にオレンジスコアが書き込まれる

  FFN2 (Layer2):
    ニューロン0: キー「記号フラグ AND りんごスコアが高い」を検出
                バリュー「りんごスコアをさらに増幅」
    ニューロン1: キー「記号フラグ AND オレンジスコアが高い」を検出
                バリュー「オレンジスコアをさらに増幅」
    → Aの「記号+りんご情報」という組み合わせパターンに反応
    → Bの「記号+オレンジ情報」という組み合わせパターンに反応

  アテンションとFFNの分業:
    アテンション → 「どのトークンの情報を集めるか」(動的)
    FFN         → 「集めた情報をどう変換するか」(学習済み知識)

  実際のLLMでは:
    FFNのキーには「前者をXとする」という構文パターン全体が
    数百〜数千次元のベクトルとして圧縮されており、
    バリューには「Xは列挙の先頭要素を指す」という
    意味的変換が記憶されている
""")
print("=" * 65)
