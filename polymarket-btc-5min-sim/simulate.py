#!/usr/bin/env python3
"""
Polymarket "BTC 5 分钟涨跌" 策略模拟器（纯模拟，不连接任何交易所，不碰真钱）

复刻推文里的四条"策略"：
  1. 时间偏差：在训练期里找"胜率 >= 65%"的时间窗口（星期几 × 一天中的第几个 5 分钟）
  2. Kelly 资金管理：f* = (p*b - q) / b
  3. 波动率过滤：开盘前 30 分钟波动太大就跳过
  4. 贝叶斯更新：每个窗口的胜率用 Beta 分布估计，每出一个结果就更新

然后把"训练期里挑出来的窗口"拿到之后从没见过的数据上去跑（样本外），
对比"回测截图"和"实盘"的差别，并把价差、手续费、跟单延迟都算进去。

数据：
  - 默认：模拟的 BTC 1 分钟价格（GARCH 波动聚集 + 厚尾 + 美股开盘时段波动放大，
    年化波动率约 55%）。按构造，涨跌方向本身没有任何可预测性。
  - 也可以用 --csv 读入真实的 Binance 1 分钟 K 线（见 README）。
"""

import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

SLOTS_PER_DAY = 1440 // 5


# ---------------------------------------------------------------- 数据

def simulate_prices(days, seed, annual_vol=0.55):
    """模拟 BTC 1 分钟收盘价。方向随机（无漂移），只有波动率有结构。"""
    rng = np.random.default_rng(seed)
    n = days * 1440
    sigma = annual_vol / np.sqrt(365 * 1440)
    a, b = 0.05, 0.94                      # GARCH(1,1)
    w = sigma ** 2 * (1 - a - b)
    nu = 4                                 # t 分布，厚尾
    z = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2))

    minute = np.arange(n) % 1440
    season = 1 + 0.6 * np.exp(-((minute - 14.5 * 60) / 90) ** 2)   # 美股开盘附近波动更大
    season /= np.sqrt(np.mean(season ** 2))

    r = np.empty(n)
    var = sigma ** 2
    for t in range(n):
        r[t] = np.sqrt(var) * z[t]
        var = w + a * r[t] ** 2 + b * var
    r *= season

    ts = pd.date_range("2025-01-06", periods=n, freq="min", tz="UTC")   # 从周一开始
    close = 60000 * np.exp(np.cumsum(r))
    return ts, close


def load_binance_csv(pattern):
    """读 Binance 1 分钟 K 线 CSV（data.binance.vision 下载的格式，无表头）。"""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"找不到文件：{pattern}")
    frames = []
    for f in files:
        df = pd.read_csv(f, header=None)
        if not str(df.iloc[0, 0]).replace(".", "").isdigit():   # 有表头的版本
            df = df.iloc[1:]
        frames.append(df.iloc[:, [0, 4]].astype(float))
    df = pd.concat(frames).drop_duplicates(subset=0).sort_values(0)
    t = df[0].to_numpy()
    unit = "us" if t[0] > 1e14 else "ms"    # 2025 年起现货数据改成了微秒
    ts = pd.to_datetime(t.astype("int64"), unit=unit, utc=True)
    return pd.DatetimeIndex(ts), df[4].to_numpy()


def build_windows(ts, close, vol_lookback=30):
    """把 1 分钟价格切成整点对齐的 5 分钟窗口。结束价 >= 开始价 记为"涨"。"""
    mod = (ts.hour * 60 + ts.minute).to_numpy()
    start = np.where(mod % 5 == 0)[0]
    start = start[(start > vol_lookback) & (start + 4 < len(close))]

    # 真实数据可能有缺口：只保留时间连续的窗口
    tt = ts.to_numpy()
    minute = np.timedelta64(1, "m")
    ok = (tt[start + 4] - tt[start - 1] == 5 * minute) & \
         (tt[start - 1] - tt[start - 1 - vol_lookback] == vol_lookback * minute)
    start = start[ok]

    p0, p1 = close[start - 1], close[start + 4]
    lr = np.diff(np.log(close), prepend=np.log(close[0]))
    c2 = np.concatenate([[0.0], np.cumsum(lr ** 2)])
    vol = np.sqrt((c2[start] - c2[start - vol_lookback]) / vol_lookback)   # 开盘前 30 分钟

    t0 = ts[start]
    return pd.DataFrame({
        "time": t0,
        "slot": t0.weekday * SLOTS_PER_DAY + mod[start] // 5,
        "up": p1 >= p0,
        "vol": vol,
    })


# ---------------------------------------------------------------- 策略

def find_time_windows(train, threshold, min_n):
    """策略 1：在训练期找胜率 >= threshold 的时间窗口（押涨或押跌都算）。"""
    g = train.groupby("slot")["up"].agg(["sum", "count"])
    rate = g["sum"] / g["count"]
    pick = (g["count"] >= min_n) & ((rate >= threshold) | (rate <= 1 - threshold))
    g = g[pick]
    bet_up = (g["sum"] / g["count"]) >= 0.5
    wins = np.where(bet_up, g["sum"], g["count"] - g["sum"])
    return pd.DataFrame({"bet_up": bet_up, "wins": wins, "n": g["count"]}, index=g.index)


def backtest(df, windows, *, bankroll=20.0, price=0.50, spread=0.01, fee=0.015,
             latency=0.0, vol_cap=None, kelly_mult=1.0, fixed_stake=None, bayes=True,
             min_bet=1.0, max_stake=1000.0):
    """
    在 df 上按时间顺序交易 windows 里的时间窗口。

    每份合约成交价 c = 0.50（公平价）+ 价差 + 跟单延迟滑点，另付 fee 比例的手续费。
    押对每份拿回 1 美元，押错归零。
    p 用 Beta(1+胜, 1+负) 的均值估计（策略 4）；仓位用 Kelly（策略 2）；
    波动率超过 vol_cap 的窗口跳过（策略 3）。
    fixed_stake 不为空时不用 Kelly，每笔固定下这么多。
    """
    c = price + spread + latency
    cost = c * (1 + fee)
    b = (1 - cost) / cost                      # 押对时每 1 美元本金的净赚

    a_ = {s: 1 + r.wins for s, r in windows.iterrows()}
    b_ = {s: 1 + r.n - r.wins for s, r in windows.iterrows()}
    side = windows["bet_up"].to_dict()

    sub = df[df["slot"].isin(windows.index)]
    if vol_cap is not None:
        sub = sub[sub["vol"] <= vol_cap]

    eq_t, eq_v = [], []
    trades = wins = 0
    ruined = False
    for t, s, up in zip(sub["time"].to_numpy(), sub["slot"].to_numpy(), sub["up"].to_numpy()):
        won = bool(up) == bool(side[s])
        p = a_[s] / (a_[s] + b_[s])
        f = (p * b - (1 - p)) / b * kelly_mult
        if fixed_stake is not None or f > 0:
            stake = fixed_stake if fixed_stake is not None else min(f * bankroll, max_stake)
            if bankroll < max(min_bet, fixed_stake or 0):
                ruined = True
                break
            stake = max(stake, min_bet)
            bankroll += stake * b if won else -stake
            trades += 1
            wins += won
            eq_t.append(t)
            eq_v.append(bankroll)
        if bayes:
            if won:
                a_[s] += 1
            else:
                b_[s] += 1
    return {
        "final": bankroll, "trades": trades, "wins": wins, "ruined": ruined,
        "breakeven": cost, "curve": (np.array(eq_t), np.array(eq_v)),
    }


REAL = "现实：1¢ 价差 + 1.5% 手续费，全 Kelly"
SCENARIOS = {
    "理想：无价差、无手续费，全 Kelly": dict(spread=0.0, fee=0.0, latency=0.0),
    REAL: dict(spread=0.01, fee=0.015, latency=0.0),
    "跟单：再加 2¢ 延迟滑点，全 Kelly": dict(spread=0.01, fee=0.015, latency=0.02),
    "现实成本，不用 Kelly，每笔固定 $1": dict(spread=0.01, fee=0.015, latency=0.0, fixed_stake=1.0),
}


def hit_rate(df, windows, vol_cap):
    """选中窗口在这段数据里的实际命中率（不管有没有下单、本金还在不在）。"""
    sub = df[df["slot"].isin(windows.index) & (df["vol"] <= vol_cap)]
    return float((sub["up"].to_numpy() == windows.loc[sub["slot"], "bet_up"].to_numpy()).mean())


def run_once(df, args):
    cut = df["time"].iloc[0] + pd.Timedelta(days=args.train_days)
    train, test = df[df["time"] < cut], df[df["time"] >= cut]
    windows = find_time_windows(train, args.threshold, args.min_n)
    vol_cap = train["vol"].quantile(args.vol_quantile)
    common = dict(bankroll=args.bankroll, vol_cap=vol_cap, max_stake=args.max_stake)

    # "回测截图"：在挑窗口用的同一段数据上回测，p 直接用训练期统计（偷看了答案）
    insample = backtest(train, windows, bayes=False, kelly_mult=args.kelly, **SCENARIOS[REAL], **common)
    out = {"n_windows": len(windows), "insample": insample,
           "hit_train": hit_rate(train, windows, vol_cap), "hit_test": hit_rate(test, windows, vol_cap)}
    for name, cfg in SCENARIOS.items():
        out[name] = backtest(test, windows, kelly_mult=args.kelly, **cfg, **common)
    return out


def _mc_worker(job):
    seed, args = job
    ts, close = simulate_prices(args.days, seed)
    r = run_once(build_windows(ts, close), args)
    for v in r.values():          # 曲线太大，蒙特卡洛只留统计量
        if isinstance(v, dict):
            v.pop("curve", None)
    return r


# ---------------------------------------------------------------- 输出

def pct(x):
    return f"{100 * x:.1f}%"


def plot(first, mc, args, path):
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fixed = list(SCENARIOS)[-1]
    fig, axes = plt.subplots(3, 1, figsize=(9, 15))

    ax = axes[0]
    for key, color, label in [
        ("insample", "#2a9d5c", "回测：在挑窗口的同一段数据上跑"),
        (REAL, "#d1495b", "实盘：全 Kelly"),
        (fixed, "#e09f3e", "实盘：每笔固定 $1"),
    ]:
        t, v = first[key]["curve"]
        if len(t):
            ax.plot(t, np.maximum(v, 0.01), color=color, label=label)
    ax.axhline(args.bankroll, color="gray", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_ylabel("资金（美元，对数刻度）")
    ax.set_title(f"同一套策略：回测 vs 实盘（本金 ${args.bankroll:.0f}）")
    t, v = first[REAL]["curve"]
    if len(t):
        ax.annotate("实盘从这里开始", xy=(t[0], args.bankroll), xytext=(-150, -60),
                    textcoords="offset points", arrowprops=dict(arrowstyle="->", color="#d1495b"),
                    color="#d1495b")
    ax.legend(loc="upper left")
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)

    ax = axes[1]
    vals = [np.mean([r["hit_train"] for r in mc]), np.mean([r["hit_test"] for r in mc])]
    be = first[REAL]["breakeven"]
    ax.bar(["回测期", "实盘期"], vals, color=["#2a9d5c", "#d1495b"], width=0.5)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.axhline(be, color="black", ls="--", lw=1)
    ax.text(1.28, be, f" 保本线 {pct(be)}", va="bottom")
    ax.text(1.28, 0.5, " 抛硬币 50%", va="top", color="gray")
    for i, val in enumerate(vals):
        ax.text(i, val + 0.005, pct(val), ha="center", fontsize=13)
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylim(0.4, 0.75)
    ax.set_title(f"\"胜率 ≥ {pct(args.threshold)} 的时间窗口\"实际命中率"
                 + (f"（{len(mc)} 次模拟平均）" if len(mc) > 1 else ""))

    ax = axes[2]
    names = list(SCENARIOS)
    loss = [np.mean([r[n]["final"] < args.bankroll for r in mc]) for n in names]
    ruin = [np.mean([r[n]["ruined"] for r in mc]) for n in names]
    y = np.arange(len(names))
    ax.barh(y - 0.2, loss, height=0.4, color="#e07a5f", label="最终亏钱")
    ax.barh(y + 0.2, ruin, height=0.4, color="#3d405b", label="本金亏光")
    for i in range(len(names)):
        ax.text(loss[i] + 0.01, i - 0.2, pct(loss[i]), va="center")
        ax.text(ruin[i] + 0.01, i + 0.2, pct(ruin[i]), va="center")
    ax.set_yticks(y, [n.replace("，", "\n", 1) for n in names])
    ax.set_xlim(0, 1.2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.set_title(f"实盘 {args.days - args.train_days} 天后" + (f"（{len(mc)} 次模拟）" if len(mc) > 1 else ""))

    fig.tight_layout()
    fig.savefig(path, dpi=130)


def report(mc, args):
    lines = [
        f"数据：{'真实 CSV ' + args.csv if args.csv else f'模拟 BTC 价格 × {len(mc)} 次'}",
        f"训练期 {args.train_days} 天，实盘期 {args.days - args.train_days} 天，本金 ${args.bankroll:.0f}，"
        f"Kelly 倍数 {args.kelly}，单笔上限 ${args.max_stake:.0f}",
        f"训练期里\"胜率 ≥ {pct(args.threshold)}\"的时间窗口：平均 {np.mean([r['n_windows'] for r in mc]):.0f} 个"
        f"（共 {7 * SLOTS_PER_DAY} 个）",
        f"这些窗口的命中率：回测期 {pct(np.mean([r['hit_train'] for r in mc]))}，"
        f"实盘期 {pct(np.mean([r['hit_test'] for r in mc]))}",
        "",
        f"回测（偷看答案）：最终资金中位数 ${np.median([r['insample']['final'] for r in mc]):,.0f}",
        "",
        "实盘（样本外）：",
    ]
    for name in SCENARIOS:
        rs = [r[name] for r in mc]
        lines.append(
            f"  {name}\n"
            f"    保本需要命中率 {pct(rs[0]['breakeven'])}，"
            f"最终资金中位数 ${np.median([x['final'] for x in rs]):,.2f}，"
            f"亏钱 {pct(np.mean([x['final'] < args.bankroll for x in rs]))}，"
            f"亏光 {pct(np.mean([x['ruined'] for x in rs]))}，"
            f"平均交易 {np.mean([x['trades'] for x in rs]):.0f} 笔"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="Binance 1 分钟 K 线 CSV，支持通配符，如 'data/BTCUSDT-1m-*.csv'")
    ap.add_argument("--days", type=int, default=364, help="模拟天数（默认 52 周）")
    ap.add_argument("--train-days", type=int, default=182, help="训练期天数（默认前 26 周）")
    ap.add_argument("--runs", type=int, default=100, help="蒙特卡洛次数（仅模拟数据）")
    ap.add_argument("--threshold", type=float, default=0.65, help="挑窗口的胜率门槛")
    ap.add_argument("--min-n", type=int, default=10, help="窗口至少要有几个样本")
    ap.add_argument("--vol-quantile", type=float, default=0.7, help="波动率超过训练期这个分位数就跳过")
    ap.add_argument("--kelly", type=float, default=1.0, help="Kelly 倍数（1 = 全 Kelly，0.5 = 半 Kelly）")
    ap.add_argument("--bankroll", type=float, default=20.0)
    ap.add_argument("--max-stake", type=float, default=1000.0, help="单笔上限（5 分钟市场盘口有限）")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.csv:
        ts, close = load_binance_csv(args.csv)
        df = build_windows(ts, close)
        args.days = (df["time"].iloc[-1] - df["time"].iloc[0]).days + 1
        first = run_once(df, args)
        if first["n_windows"] == 0:
            raise SystemExit(f"训练期里没有样本数 >= {args.min_n} 的窗口：请多下载几个月的数据，"
                             f"或调低 --min-n（每个窗口一周只出现一次）")
        mc = [first]
    else:
        with Pool() as pool:
            mc = pool.map(_mc_worker, [(s, args) for s in range(args.runs)])
        ts, close = simulate_prices(args.days, 0)        # 第一次的完整曲线用于画图
        first = run_once(build_windows(ts, close), args)

    text = report(mc, args)
    print(text)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    plot(first, mc, args, os.path.join(args.out, "summary.png"))
    print(f"\n图表：{os.path.join(args.out, 'summary.png')}")


if __name__ == "__main__":
    main()
