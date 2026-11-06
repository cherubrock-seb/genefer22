from check_register_prepare_tail import compare
for rns in (2,3):
    for wide in (False,True):
        for wi in (False,True):
            print(f"n=16 rns={rns} wide={wide} use_wi={wi}", flush=True)
            compare(1<<16, rns, wide, wi)
for n in (12,14,18,20,22):
    print(f"n={n} rns=3 wide=True use_wi=False", flush=True)
    compare(1<<n, 3, True, False)
