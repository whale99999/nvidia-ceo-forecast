#!/usr/bin/env python3
"""Forecaster for make.fun polymarket-nvidia-ceo-forecast-01.

Model (fitted offline on the kit's practice cases p00-p05 and on sliding
checkpoints over the same recorded price history; the program itself only
reads the case it is given):

  - The median forecast is persistence: each market's Yes price is expected
    to stay at its checkpoint price. Mean-reversion and trend variants were
    backtested and lost to persistence on every held-out fold.
  - The spread around the median is per market: an EWMA (half-life 360
    minutes) of the market's absolute one-minute price changes, shrunk 30%
    toward the case-wide median of that statistic, floored and capped.
  - The spread grows like h**0.45 with the horizon h in minutes and is
    damped near the 0/1 boundary by (4x(1-x))**0.3.
  - Quantile offsets are empirical multipliers on that spread, one set per
    horizon bucket (1-10, 11-30, 31-90, 91-240 minutes) and per market,
    fitted as quantiles of the model's own standardised residuals over the
    practice history and shrunk 70/30 toward the pooled table. The inner
    multipliers come out at or near zero: most minutes the price does not
    move at all, so the honest 25-75% band is a point until horizons grow.

Everything is deterministic: two runs on the same case write the same bytes.
Standard library only, no network.
"""

import json
import os
from pathlib import Path

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

VOL_HALF_LIFE = 360.0
VOL_SHRINK = 0.3
VOL_FLOOR = 0.0004
VOL_CAP = 0.03
GAMMA = 0.45
BETA = 0.3
BUCKET_EDGES = (10, 30, 90)

TABLE_JSON = """{"pooled":[[-2.9348,0.0,0.0,0.0,2.5123],[-4.2838,-0.245,0.0,0.0,3.7985],[-4.6242,-0.5683,0.0,0.1564,5.9039],[-4.2014,-0.8341,0.0,0.3997,5.5069]],"per_market":{"0x9913e01540453df27e7ca2182c0b2a9ce4de099d7cf7d4397bd80a3832916400":[[-2.9034,0.0,0.0,0.0,0.7537],[-3.1593,-0.1221,0.0,0.0,2.4257],[-3.0771,-0.3274,0.0,0.0469,4.1657],[-2.4463,-0.5069,0.0,0.1838,5.1872]],"0xa82699004214aa3f755ede5d9e6d5c4850e3f69ec821a3b15433a1bd6a3a19aa":[[-2.3474,-0.0795,0.0,0.2262,3.5136],[-3.2804,-0.407,0.0,0.3064,4.898],[-3.278,-0.6904,0.0,0.4295,5.5499],[-2.939,-0.9588,0.0,0.7503,9.8992]],"0x3c6918f6944f05883477c01f0f4380a63096e8e11f00c11595ac1aa1294c3358":[[-1.4452,0.0,0.0,0.0,1.2017],[-2.6756,-0.0735,0.0,0.0,1.7063],[-2.8669,-0.5549,0.0,0.0469,6.0151],[-2.4796,-0.9276,0.0,0.1199,4.7919]],"0xf74c2c32764c1c30ae9fec8ff310af9f43b6bd3683dac58b9b9bb913ad7fffea":[[-3.1288,0.0,0.0,0.0,2.4199],[-6.0072,-0.505,0.0,0.0,4.0419],[-4.9206,-0.8379,0.0,0.1768,5.4685],[-3.9005,-1.0351,-0.0944,0.3634,4.6279]],"0xf282298236c2b9a66d6f7ca7cd17a23969ebba9d93b818210090d7add5f4a83c":[[-2.6594,0.0,0.0,0.0,1.5278],[-3.0835,-0.3253,0.0,0.0,1.8148],[-2.6117,-0.5345,0.0,0.1068,3.2882],[-2.4705,-0.6213,0.0,0.2959,5.119]],"0xa0afeccf72356d88deb62e63eb21ca44949d55f9c4d0a0d7bc1ff604751caf1e":[[-2.3596,0.0,0.0,0.0,1.4539],[-4.0046,-0.2063,0.0,0.1329,2.0574],[-3.7735,-0.6355,0.0,0.2288,2.2428],[-5.5328,-1.4442,-0.0649,0.3249,2.0717]],"0x497f156c5795aa461dda8818084550277bcf85e6686d5ad354f025a6034fe83a":[[-4.4413,0.0,0.0,0.0,1.5936],[-6.2989,-0.4956,0.0,0.0,3.1279],[-4.8925,-0.609,0.0,0.0469,5.0119],[-3.983,-0.8983,-0.2066,0.4418,4.2072]],"0xd54c5a8a7a36512ce3b310cc97ac146e0f3ea74b8eadfd3f823459a39f3e2819":[[-2.8243,0.0,0.0,0.0,6.3448],[-6.2462,-0.642,0.0,0.0,7.7251],[-4.59,-0.5197,0.0,0.7354,7.6974],[-3.6891,-0.5846,0.1461,2.4877,8.2228]],"0x1d956daa1a57a0fde801ec275307881fa580e32453378629980dc9850b453a95":[[-0.8804,0.0,0.0,0.0,0.7537],[-1.6431,-0.0735,0.0,0.0,3.183],[-1.6852,-0.1705,0.0,0.0469,2.3397],[-1.5691,-0.2502,0.0,0.1199,1.9809]],"0xd940515f5526f103b1aa4d3b297792d22ee82c24953c5e2ad3ae9687403f7f00":[[-4.2085,0.0,0.0,0.0,3.2057],[-7.5719,-1.244,0.0,0.0,7.382],[-5.7826,-1.7742,0.0,0.8159,7.1853],[-5.2159,-2.4254,-0.2446,1.877,5.9239]],"0xfe28d93c0729d1e099a0359caaf1277e4e6560034c2e3900d65921515f467282":[[-2.9612,-0.6632,0.0,0.0,4.5404],[-5.1937,-0.6712,0.0,0.159,6.7899],[-6.8879,-0.6734,-0.1069,0.4814,7.5078],[-7.7,-0.9435,-0.1503,0.7484,7.7534]],"0x4c12a732cd77165089070051dcecced7be15feaf8aa284d908744dbbf88af357":[[-2.1194,0.0,0.0,0.0,1.989],[-3.6101,-0.6067,0.0,0.0,3.8474],[-4.3266,-1.263,0.0,0.0469,6.3375],[-9.3942,-1.8799,0.0,0.2939,11.0088]],"0xddab1a21931237345f33cdf4ac90351be6c864bce02f1d39a1f8e30630e7b615":[[-1.9713,0.0,0.0,0.0,1.8816],[-2.5573,-0.0735,0.0,0.0,2.2198],[-2.5773,-0.4161,0.0,0.0469,2.9699],[-2.4414,-0.6328,0.0,0.2717,6.3678]],"0xea85b51786537d4b608e7420ec9938f640f03a75b0564c2d78e82c5ffa0f91e8":[[-5.5048,0.0,0.0,0.0,8.2568],[-4.776,-0.2511,0.0,0.3568,5.8881],[-6.125,-0.5091,0.0,1.6092,10.3907],[-4.8342,-1.0365,0.0,1.1374,6.0192]],"0x874b4a0617d6a44e28e57a513987cb63d8eb5d8d50105059749ba3118fbe7f37":[[-5.6152,0.0,0.0,0.0,4.7497],[-4.7048,-0.0735,0.0,0.0,6.2419],[-3.7732,-0.3635,0.0,0.0469,7.4532],[-3.2798,-0.6272,0.0,1.3874,5.3141]],"0x475196360677530c2d57cc95e1cf3276fa4afc0ce5dcdd14ad0ec23b1fc6ff7f":[[-3.1881,0.0,0.0,0.0,3.2099],[-2.8245,-0.0735,0.0,0.0,3.0355],[-2.5029,-0.1705,0.0,0.0469,2.9059],[-1.9657,-0.2502,0.0,0.1199,2.4093]],"0xf24ee57d6ca0020518692e737764f8553f110554fb6c634614ba694c5510dcc4":[[-4.3785,-0.425,0.0,0.0,3.3228],[-5.7916,-0.7952,0.0,0.4996,4.0243],[-5.9616,-1.3946,0.0637,1.9317,7.8543],[-5.2742,-1.321,1.1327,2.4428,8.5117]],"0xa162fa01bb8ef9e25614f7d0a04e64cbf15f0c70bd3d0fc5d15b2fa892e0f470":[[-1.4679,0.0,0.0,0.0,0.7537],[-3.0749,-0.0735,0.0,0.0,2.7909],[-2.9358,-0.3301,0.0,0.0469,4.1149],[-2.3806,-0.7874,0.0,0.6486,3.9055]],"0x9ae0d7b8addd0acab7a762583f19b84869e707a32b52192216ad3cf5897d5eac":[[-1.881,0.0,0.0,0.0,0.7537],[-2.4203,-0.0735,0.0,0.0,1.9671],[-2.5495,-0.4842,0.0,0.0469,6.8119],[-2.5922,-0.7428,0.0,0.1199,5.7441]],"0x4c3844f8716850ebdad01ff02ca7cf798a99e99c468d6bdd1aff19b42dd56401":[[-14.2313,0.0,0.0,0.0,5.4022],[-8.1659,-0.6571,0.0,0.0,6.8705],[-5.2543,-2.8623,0.0,0.0469,6.4585],[-4.721,-2.2827,0.0,0.1199,4.6567]],"0x65345fc617268fc8a42937c1eea92c18bf8bdce76e9e503c34fe94f18d67acac":[[-7.3676,0.0,0.5987,1.3473,39.6346],[-7.7441,-0.5902,0.2798,1.9649,27.1013],[-5.4186,-0.9428,0.2997,4.254,37.7495],[-4.3376,-1.3262,-0.6215,2.48,27.4754]],"0x9f4d7751fc43e881bb4f90e85a29fba5b58633aad4e2b3d1e98e867e7ff41932":[[-2.8263,0.0,0.0,0.0,0.7537],[-9.6457,-0.0735,0.0,0.0,1.1395],[-8.8813,-0.1705,0.0,0.0469,2.3332],[-6.3676,-3.8078,0.0,0.1199,2.0197]],"0x770f4fb78a642e47e0df450503f5150623ad4a54ab308bc9ae3b149d9b43759c":[[-11.886,0.0,0.0,0.0,0.7537],[-19.4035,-0.0735,0.0,0.0,1.1395],[-18.1585,-0.1705,0.0,0.0469,1.7712],[-12.4848,-4.1256,0.0,0.1199,1.6521]],"0x0ddf9ce5b763637a84564146d912355d3703c91c5d01e8de1ea84b133d3064ec":[[-0.8804,0.0,0.0,0.0,0.7537],[-9.3482,-0.0735,0.0,0.0,1.1395],[-7.5124,-0.1705,0.0,0.0469,5.8968],[-5.2027,-2.6914,0.0,0.1199,4.2969]]}}"""
TABLES = json.loads(TABLE_JSON)
POOLED = TABLES["pooled"]
PER_MARKET = TABLES["per_market"]


def ewma_abs_diff(prices):
    """EWMA of |price changes| with the fitted half-life.

    The running value starts at 0.0 (the no-change minute that precedes a
    market's first change), exactly as in the offline fit.
    """
    alpha = 1.0 - 2.0 ** (-1.0 / VOL_HALF_LIFE)
    s = 0.0
    prev = prices[0]
    for p in prices[1:]:
        d = p - prev
        if d < 0.0:
            d = -d
        s = alpha * d + (1.0 - alpha) * s
        prev = p
    return s


def median(values):
    v = sorted(values)
    n = len(v)
    mid = n // 2
    if n % 2:
        return v[mid]
    return (v[mid - 1] + v[mid]) / 2.0


def bucket_of(h):
    if h <= BUCKET_EDGES[0]:
        return 0
    if h <= BUCKET_EDGES[1]:
        return 1
    if h <= BUCKET_EDGES[2]:
        return 2
    return 3


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (never -0)."""
    if x < 0.0:
        x = 0.0
    elif x > 1.0:
        x = 1.0
    return round(x * 1e6) / 1e6 + 0.0


def forecast_market(market, vol, vol_median, horizon):
    prices = market["prices"]
    if not prices:
        raise ValueError("market %s has no prices" % market["id"])
    x = prices[-1]
    s = (1.0 - VOL_SHRINK) * vol + VOL_SHRINK * vol_median
    if s < VOL_FLOOR:
        s = VOL_FLOOR
    elif s > VOL_CAP:
        s = VOL_CAP
    mult = PER_MARKET.get(market["id"], POOLED)
    boundary = (4.0 * x * (1.0 - x)) ** BETA if 0.0 < x < 1.0 else 0.0
    rows = []
    for h in range(1, horizon + 1):
        scale = s * (h ** GAMMA) * boundary
        m = mult[bucket_of(h)]
        row = []
        prev = 0.0
        for tau in range(5):
            v = price6(x + m[tau] * scale)
            if v < prev:
                v = prev
            row.append(v)
            prev = v
        rows.append(row)
    return {"id": market["id"], "q": rows}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in input_dir.iterdir() if p.name.endswith(".in"))
    for name in files:
        case = json.loads((input_dir / name).read_text(encoding="utf-8"))
        if list(case["quantiles"]) != QUANTILES:
            raise ValueError("%s asks for quantiles %r" % (name, case["quantiles"]))
        horizon = int(case["horizon_minutes"])
        markets = case["markets"]
        vols = [ewma_abs_diff(m["prices"]) for m in markets]
        vol_median = median(vols)
        forecasts = [
            forecast_market(m, v, vol_median, horizon)
            for m, v in zip(markets, vols)
        ]
        out = json.dumps({"forecasts": forecasts}, separators=(",", ":"))
        (output_dir / (name[:-3] + ".out")).write_text(out + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
