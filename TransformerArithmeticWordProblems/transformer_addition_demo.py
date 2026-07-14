# -*- coding: utf-8 -*-
"""
「２つの数字を足す」文章をTransformerがどう処理するかを
模式的に(あくまで概念デモとして)再現するスクリプト。

★ 注意 ★
これは本物の学習済みTransformerではありません。
実際のモデルは数百億パラメータの分散表現とAttentionの学習によって
以下のような処理を"創発的に"実現していますが、
ここでは各ステージ(トークン化 → Attention → FFN → 自己回帰生成)の
「役割」をわかりやすく模式化したシミュレーションです。
"""

import re
import random
import numpy as np

random.seed(0)
np.random.seed(0)

TEXT = "いまから２つの数字を足して下さい。最初の数字は、３と６と４を足したもの。たせ。次の数字は、５と７と９と２をたしたもの。たせ。では、足して下さい。"


# ============================================================
# Stage 1: トークン化
# ============================================================
def tokenize(text: str):
    """
    簡略化トークナイザ。
    実際のBPE/SentencePieceの代わりに、既知の単語・記号・数字を
    素朴な正規表現で切り出す(デモ用)。
    """
    pattern = r"数字|足して|下さい|最初|次|では|たせ|たした|いまから|２つ|もの|[０-９0-9]|[のをはとで、。]"
    tokens = re.findall(pattern, text)
    return tokens


tokens = tokenize(TEXT)

print("=" * 60)
print("Stage 1: トークン化")
print("=" * 60)
print(tokens)
print(f"トークン数: {len(tokens)}\n")


# ============================================================
# Stage 2: 埋め込み (Embedding)
# ============================================================
DIM = 16  # デモ用の小さい埋め込み次元

def embed(token: str):
    """
    トークン文字列をハッシュして固定の疑似埋め込みベクトルを作る。
    数字トークンには「数値らしさ」を表す特別な次元も追加する。
    """
    rng = np.random.RandomState(abs(hash(token)) % (2**32))
    vec = rng.normal(size=DIM)
    if token.isdigit():
        # 先頭の次元に実際の数値情報を"埋め込む"(実モデルでは学習で獲得される)
        vec[0] = float(token)
        vec[1] = 1.0  # 「これは数字トークンである」フラグ的な次元
    else:
        vec[1] = 0.0
    return vec


embeddings = [embed(t) for t in tokens]

print("=" * 60)
print("Stage 2: 埋め込み (最初の3トークンの例)")
print("=" * 60)
for t, e in list(zip(tokens, embeddings))[:3]:
    print(f"  '{t}': {np.round(e[:5], 2)} ...(以下{DIM-5}次元省略)")
print()


# ============================================================
# Stage 3: Attention によるグループ分け
# ============================================================
def find_groups(tokens):
    """
    実際のAttentionは学習された重みで「どの数字が
    どの"最初の/次の"に属するか」を紐付けるが、
    ここでは「。」で区切られた文単位をセグメントとみなし、
    各セグメント内の数字トークンをグループ化することで模式的に再現する。

    → これは実モデルにおける「直前の文脈にある同種トークンへ
       注意を向けるヘッド(coreference resolution的な機構)」の
       働きを簡略化したものに相当する。
    """
    groups = []
    current_group = []
    current_label = None

    for t in tokens:
        if t in ("最初", "次"):
            current_label = t
        elif t.isdigit():
            current_group.append(t)
        elif t == "。":
            if current_group:
                groups.append((current_label, current_group))
            current_group = []
            current_label = None
    return groups


groups = find_groups(tokens)

print("=" * 60)
print("Stage 3: Attentionによるグループ紐付け(模式化)")
print("=" * 60)
for label, digits in groups:
    print(f"  ラベル『{label}』 ← 注意が向けられた数字トークン: {digits}")
print()


# ============================================================
# Stage 4: FFN による「近似計算」のシミュレーション
# ============================================================
def ffn_addition_circuit(numbers):
    """
    Anthropicの回路解析(attribution graph)で報告されている
    足し算回路の考え方を簡略化して再現する:

      (a) 大きさ概算ヒューリスティック: log空間でのおおよその和の推定
          → ノイズを含む「だいたいこのくらい」という見積もり
      (b) 下一桁パターンマッチ回路: 1の位だけを見た正確な計算
      (c) (a)と(b)を突き合わせて、最も辻褄の合う整数解に"スナップ"する

    実際のモデルはこれをベクトル空間上の重ね合わせで並列に行うが、
    ここでは概念を明示的な2つの推定値の統合として書く。
    """
    exact_sum = sum(int(n) for n in numbers)

    # (a) 大きさの概算: 真の和にわざとノイズを乗せた"おおよその感覚"
    magnitude_estimate = exact_sum + np.random.normal(scale=1.5)

    # (b) 下一桁だけを見た正確なパターンマッチ(mod 10)
    last_digit_lookup = sum(int(n) for n in numbers) % 10

    # (c) 概算値の中から、下一桁が(b)と一致する最も近い整数を選ぶ
    #     = 「だいたいの大きさ」と「正確な下一桁」の統合によるスナップ
    candidates = range(int(magnitude_estimate) - 10, int(magnitude_estimate) + 10)
    best = min(candidates, key=lambda c: (abs(c - magnitude_estimate) if c % 10 == last_digit_lookup else 1e9))

    return {
        "numbers": numbers,
        "magnitude_estimate": round(magnitude_estimate, 2),
        "last_digit_lookup": last_digit_lookup,
        "snapped_result": best,
        "true_result": exact_sum,
    }


print("=" * 60)
print("Stage 4: FFNによる近似計算のシミュレーション")
print("=" * 60)

sub_results = []
for label, digits in groups:
    result = ffn_addition_circuit(digits)
    sub_results.append(result)
    print(f"  グループ『{label}』 {'+'.join(digits)}")
    print(f"    大きさ概算: {result['magnitude_estimate']}")
    print(f"    下一桁パターン: {result['last_digit_lookup']}")
    print(f"    → スナップ後の結果: {result['snapped_result']} (正解: {result['true_result']})")
print()


# ============================================================
# Stage 5: 自己回帰生成(中間結果をトークン化して系列に戻す)
# ============================================================
print("=" * 60)
print("Stage 5: 自己回帰生成 (Chain of Thought)")
print("=" * 60)

generated_sequence = list(tokens)  # 元の系列
intermediate_tokens = []

for i, result in enumerate(sub_results):
    new_token = str(result["snapped_result"])
    intermediate_tokens.append(new_token)
    generated_sequence.append(new_token)
    print(f"  ステップ{i+1}: 中間結果トークン '{new_token}' を生成 → 系列に追加")
    print(f"    (この時点でモデルは元の3〜4個の数字ではなく、")
    print(f"     この1個のトークン'{new_token}'にAttentionを向けるだけで済む)")

# 最終ステップ: 生成済みの中間トークン同士を足す
final_result = ffn_addition_circuit(intermediate_tokens)
generated_sequence.append(str(final_result["snapped_result"]))

print(f"\n  最終ステップ: '{intermediate_tokens[0]}' + '{intermediate_tokens[1]}' を計算")
print(f"    → 最終トークン '{final_result['snapped_result']}' を生成")

print("\n" + "=" * 60)
print("生成された最終系列(末尾)")
print("=" * 60)
print(" ".join(generated_sequence[-6:]))
print(f"\n>>> 最終出力: {final_result['snapped_result']}")
