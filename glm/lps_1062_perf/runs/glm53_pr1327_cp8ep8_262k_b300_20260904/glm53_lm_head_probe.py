import time

import torch


def main() -> None:
    device = torch.device("cuda:0")
    rows = 4_096
    hidden = 6_144
    vocab = 151_552

    activations = torch.zeros((rows, hidden), device=device, dtype=torch.bfloat16).float()
    weight = torch.zeros((vocab, hidden), device=device, dtype=torch.bfloat16).float()
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = activations @ weight.T
    torch.cuda.synchronize()
    print(f"forward_ms={(time.perf_counter() - started) * 1e3:.3f} shape={tuple(output.shape)}")


if __name__ == "__main__":
    main()
