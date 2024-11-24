import sys

def gen_alpha(n_long: int, n_short: int, chunk_size_level_0: int, 
              chunk_size_level_1: int, quality_0: float, quality_1: float, bandwidth_0: float,
              bandwidth_1: float):
    assert n_long > n_short
    assert quality_0 > quality_1
    assert chunk_size_level_0 > chunk_size_level_1
    q_c_ratio = (quality_0 - quality_1) / (chunk_size_level_0 - chunk_size_level_1)
    assert bandwidth_0 > bandwidth_1
    upper_one = q_c_ratio * (bandwidth_0 / n_short)
    lower_one = q_c_ratio * (bandwidth_0 / n_long)
    upper_two = (quality_0 - quality_1) / (((n_short * chunk_size_level_0)/bandwidth_1 - (n_short * chunk_size_level_1)/bandwidth_0))
    upper = min(upper_one, upper_two)
    lower = lower_one
    assert upper > lower, f"upper_one: {upper_one}, upper_two: {upper_two}, lower_one: {lower_one}"
    mid = (upper + lower) / 2
    print(f"upper: {upper}, lower: {lower}, mid: {mid}")
    return mid
if __name__ == "__main__":
    # 131072 per token per PP for Llama-3-8B.
    # 33554432 per chunk.
    # bandwidth_0 = 12GB = 12 * 1024 * 1024 * 1024 = 12884901888
    # bandwidth_1 = 1GB = 1073741824
    # 0.4-0.992488959547783-0.0-0.0,0.3-0.9878210563504681-0.0-0.0,0.2-0.98-0.0-0.0,0.1-0.95-0.0-0.0,0.05-0.9-0.0-0.0
    # For 0.1-0.95, 12GB 1GB 131072 per token --> 0.39
    if len(sys.argv) != 9:
        print("Usage: python gen_alpha.py n_long n_short token_size_level_0 token_size_level_1 quality_0 quality_1 bandwidth_0 bandwidth_1")
        sys.exit(1)
    n_long = int(sys.argv[1])
    n_short = int(sys.argv[2])
    chunk_size_level_0 = int(sys.argv[3])
    chunk_size_level_1 = int(sys.argv[4])
    quality_0 = float(sys.argv[5])
    quality_1 = float(sys.argv[6])
    bandwidth_0 = float(sys.argv[7])
    bandwidth_1 = float(sys.argv[8])
    print(gen_alpha(n_long, n_short, chunk_size_level_0, chunk_size_level_1, quality_0, quality_1, bandwidth_0, bandwidth_1))