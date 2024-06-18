import stream_v3_9
for delay in range(0, 33, 3):  # Genera i valori 0, 3, 6, ..., 30
    print("Simulazione a delay:",delay)
    stream_v3_9.run_script(delay)