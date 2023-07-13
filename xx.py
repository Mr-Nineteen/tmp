import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM


def print_latency(latency_set, title, warmup=3):
    # trim warmup queries
    latency_set = list(latency_set)
    latency_set = latency_set[warmup:]
    count = len(latency_set)
    if count > 0:
        latency_set.sort()
        n50 = (count - 1) * 0.5 + 1
        n90 = (count - 1) * 0.9 + 1
        n95 = (count - 1) * 0.95 + 1
        n99 = (count - 1) * 0.99 + 1
        n999 = (count - 1) * 0.999 + 1
        avg = sum(latency_set) / count
        p50 = latency_set[int(n50) - 1]
        p90 = latency_set[int(n90) - 1]
        p95 = latency_set[int(n95) - 1]
        p99 = latency_set[int(n99) - 1]
        p999 = latency_set[int(n999) - 1]
        print(f"====== latency stats {title} ======")
        print("\tAvg Latency: {0:8.2f} ms".format(avg * 1000))
        print("\tP50 Latency: {0:8.2f} ms".format(p50 * 1000))
        print("\tP90 Latency: {0:8.2f} ms".format(p90 * 1000))
        print("\tP95 Latency: {0:8.2f} ms".format(p95 * 1000))
        print("\tP99 Latency: {0:8.2f} ms".format(p99 * 1000))
        print("\t999 Latency: {0:8.2f} ms".format(p999 * 1000))


device = torch.device("cuda:0")

print("start")
t1 = time.time()

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")

model = model.to(device)
t2 = time.time()
print("load model, using time: ", t2 - t1, " s")
t0 = time.time()
print("compile using time: ", t0 - t2, " s")
model.eval()

time_list = []
for _ in range(30):
    t3 = time.time()
    input_text = "The quick brown fox is"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(device)
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    t4 = time.time()
    time_list.append(t4 - t3)
    print("Input text: ", input_text)
    print("Generated text: ", generated_text)

print(print_latency(time_list, "predict_time"))
