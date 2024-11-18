import sys
def extract():
    alpha = None
    throughput = None
    avg_ttft = None
    avg_quality = None
    # Read line by line from stdin
    for line in sys.stdin:
        # Split the line into words
        words = line.split()
        # Find the index of a word
        for widx, word in enumerate(words):
            if word == "alpha":
                if widx + 3 >= len(words):
                    continue
                if words[widx+1] != "in":
                    continue
                alpha = float(words[widx+3])
                break
            if word == "Throughput:":
                throughput = float(words[widx+1])
                break
            if word == "Average":
                if words[widx+1] == "TTFT:":
                    avg_ttft = float(words[widx+2])
                if words[widx+1] == "quality:":
                    avg_quality = float(words[widx+2])
    assert throughput is not None
    assert avg_ttft is not None
    assert avg_quality is not None
    if alpha is None:
        alpha = "_"
    alpha = str(alpha)
    throughput = str(throughput)
    avg_ttft = str(avg_ttft)
    avg_quality = str(avg_quality)
    return "-".join([alpha, throughput, avg_ttft, avg_quality])

if __name__ == "__main__":
    print(extract())
