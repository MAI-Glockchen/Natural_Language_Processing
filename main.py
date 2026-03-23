from pipeline import run

if __name__ == "__main__":
    print("starting pipeline...", flush=True)
    stats = run()
    print(stats, flush=True)