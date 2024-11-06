from matplotlib import pyplot as plt

# Three subplots.
# x-axis are ['no-cache', 'lru', 'lfu', 'oursv1']
# y-axis are TTFT, throughput, quality
'''
INFO 11-04 05:07:06 simulator.py:99] Starting simulation with cluster: Cluster({'id': 0, 'num_replicas': 1}) and 4965 requests
INFO 11-04 05:07:07 simulator.py:108] Simulation ended at: 13925.522135958885s
INFO 11-04 05:07:07 simulator.py:112] Throughput: 0.3565395933829464 req/s
INFO 11-04 05:07:07 simulator.py:115] Average TTFT: 6961.571086590569s
INFO 11-04 05:07:07 simulator.py:117] Average quality: 1.0
INFO 11-04 05:07:07 simulator.py:119] Average hit length: 0.0
INFO 11-04 05:07:07 simulator.py:122] Replica 0 accumulated execution time: 13925.022135958885
INFO 11-04 05:07:07 simulator.py:126] Writing output
INFO 11-04 05:07:17 simulator.py:129] Metrics written
INFO 11-04 05:07:17 simulator.py:137] Chrome event trace written


INFO 11-04 05:07:34 simulator.py:99] Starting simulation with cluster: Cluster({'id': 0, 'num_replicas': 1}) and 4965 requests
INFO 11-04 05:07:44 simulator.py:108] Simulation ended at: 4940.137559227552s
INFO 11-04 05:07:44 simulator.py:112] Throughput: 1.0050327426057213 req/s
INFO 11-04 05:07:44 simulator.py:115] Average TTFT: 2447.6007269162587s
INFO 11-04 05:07:44 simulator.py:117] Average quality: 1.0
INFO 11-04 05:07:44 simulator.py:119] Average hit length: 9873.814300100705
INFO 11-04 05:07:44 simulator.py:122] Replica 0 accumulated execution time: 3897.69834613073
INFO 11-04 05:07:44 simulator.py:126] Writing output
INFO 11-04 05:07:54 simulator.py:129] Metrics written
INFO 11-04 05:07:54 simulator.py:137] Chrome event trace written


INFO 11-04 05:08:12 simulator.py:99] Starting simulation with cluster: Cluster({'id': 0, 'num_replicas': 1}) and 4965 requests
INFO 11-04 05:08:24 simulator.py:108] Simulation ended at: 4939.9971628753065s
INFO 11-04 05:08:24 simulator.py:112] Throughput: 1.0050613059684717 req/s
INFO 11-04 05:08:24 simulator.py:115] Average TTFT: 2447.5293160473757s
INFO 11-04 05:08:24 simulator.py:117] Average quality: 1.0
INFO 11-04 05:08:24 simulator.py:119] Average hit length: 9873.814300100705
INFO 11-04 05:08:24 simulator.py:122] Replica 0 accumulated execution time: 3897.69834613073
INFO 11-04 05:08:24 simulator.py:126] Writing output
INFO 11-04 05:08:33 simulator.py:129] Metrics written
INFO 11-04 05:08:33 simulator.py:137] Chrome event trace written


INFO 11-04 05:08:51 simulator.py:99] Starting simulation with cluster: Cluster({'id': 0, 'num_replicas': 1}) and 4965 requests
INFO 11-04 05:09:04 simulator.py:108] Simulation ended at: 4328.1661319223995s
INFO 11-04 05:09:04 simulator.py:112] Throughput: 1.1471371127324874 req/s
INFO 11-04 05:09:04 simulator.py:115] Average TTFT: 2149.6451782869835s
INFO 11-04 05:09:04 simulator.py:117] Average quality: 0.9684189325277145
INFO 11-04 05:09:04 simulator.py:119] Average hit length: 9873.814300100705
INFO 11-04 05:09:04 simulator.py:122] Replica 0 accumulated execution time: 3897.69834613073
INFO 11-04 05:09:04 simulator.py:126] Writing output
INFO 11-04 05:09:16 simulator.py:129] Metrics written
INFO 11-04 05:09:16 simulator.py:137] Chrome event trace written
'''
x = ['no-cache', 'lru', 'lfu', 'oursv1']
y1 = [6961.571086590569, 2975, 2626, 2161.682010005244]
y2 = [0.3565395933829464, 0.724467, 0.8706188, 1.125]
y3 = [1.0, 1.0, 1.0, 0.98420]
# plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.bar(x, y1)
plt.title('average TTFT(second)')
plt.subplot(1, 3, 2)
plt.bar(x, y2)
plt.title('average throughput(req/s)')
plt.subplot(1, 3, 3)
plt.bar(x, y3)
plt.title('average quality')
plt.savefig("comparison.pdf", format='pdf')
'''
from matplotlib import pyplot as plt

# Three subplots.
# x-axis are ['no-cache', 'lru', 'lfu', 'oursv1']
# y-axis are TTFT, throughput, quality

x = ['lru', 'oursv1']
y1 = [2530.267875264543, 2486.1282510151145]
y2 = [0.9871328487261307, 0.9992193051445627]
y3 = [1.0, 0.9685800604229813]
# plot

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
# Make the different bigger.
# Starting from not 0.
plt.bar(x, y1)
plt.title('average TTFT(second)')
plt.subplot(1, 3, 2)
plt.bar(x, y2)
plt.title('average throughput(req/s)')
plt.subplot(1, 3, 3)
plt.bar(x, y3)
plt.title('average quality')
plt.show()

'''