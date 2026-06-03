import numpy as np
import matplotlib.pyplot as plt
import math


def opdr1():  
    Lc = 7e-6
    kBT_Lp = 0.0911e-12

    # WLC curve
    x = np.linspace(0, 0.99 * Lc, 1000)
    y = kBT_Lp * (1 / (4 * (1 - x/Lc)**2) - 1/4 + x/Lc)

    # Bisectie zonder scipy
    def vind_Lext(F_N):
        lo, hi = 0.001 * Lc, 0.9999 * Lc
        for _ in range(80):
            mid = (lo + hi) / 2
            f_mid = kBT_Lp * (1 / (4*(1 - mid/Lc)**2) - 0.25 + mid/Lc)
            if f_mid < F_N:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    forces_pN = [0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 40]
    Lext_punten = [vind_Lext(F * 1e-12) for F in forces_pN]

    print(f"{'F (pN)':<10} {'L_ext (µm)'}")
    print("-" * 22)
    for F, L in zip(forces_pN, Lext_punten):
        print(f"{F:<10} {L*1e6:.3f}")

    plt.figure(figsize=(7, 5))
    plt.plot(x * 1e6, y * 1e12, label="WLC")
    plt.plot([L * 1e6 for L in Lext_punten], forces_pN, "ro", label="kalibratiepunten")
    plt.axvline(Lc * 1e6, color="gray", linestyle="--", label="$L_c$ = 7.00 µm")
    plt.xlabel("$L_{ext}$ (µm)")
    plt.ylabel("$F$ (pN)")
    plt.title("WLC extensiecurve — 20.6 kbp DNA")
    plt.xlim(0, 7.5)
    plt.ylim(0, 8)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def opdr2():
    Lc = 7e-6
    kBT_Lp = 0.0911e-12
    gamma = 9.4248e-9
    fac = 58

    def vind_Lext(F_N):
        lo, hi = 0.001 * Lc, 0.9999 * Lc
        for _ in range(80):
            mid = (lo + hi) / 2
            f_mid = kBT_Lp * (1 / (4*(1 - mid/Lc)**2) - 0.25 + mid/Lc)
            if f_mid < F_N:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    forces_pN = [0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 40]

    print(f"{'F (pN)':<10} {'t_c,x (ms)':<15} {'t_c,x / (1/fac)':<20} {'Overschatting (%)'}")
    print("-" * 62)

    tc_vals = []
    for F_pN in forces_pN:
        F = F_pN * 1e-12
        Lext = vind_Lext(F)
        tc = gamma * Lext / F * 1000
        fc = 1 / (2 * math.pi * tc / 1000)
        ratio = tc / (1000 / fac)
        ov = (1 / ((2 / math.pi) * math.atan(fac / (2 * fc))) - 1) * 100
        tc_vals.append(tc)
        print(f"{F_pN:<10} {tc:<15.1f} {ratio:<20.1f} {ov:.1f}")

    plt.figure(figsize=(7, 5))
    plt.plot(forces_pN, tc_vals, "bo-", label="$t_{c,x}$")
    plt.axhline(1000 / fac, color="red",   linestyle="--", label="$1/f_{ac}$ = 17.2 ms")
    plt.axhline(38,         color="green", linestyle="--", label="Grens <10% (38 ms)")
    plt.xscale("linear")
    plt.yscale("linear")
    plt.xlabel("$F$ (pN)")
    plt.ylabel("$t_{c,x}$ (ms)")
    plt.title("Karakteristieke tijd vs. kracht")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()

import math

def opdr3():
    Lc    = 7e-6         # m
    kBT   = 4.1e-21      # J
    kBT_Lp = 0.0911e-12  # N
    R     = 1.4e-6       # m  (M270 bead radius)

    def vind_Lext(F_N):
        lo, hi = 0.001 * Lc, 0.9999 * Lc
        for _ in range(80):
            mid = (lo + hi) / 2
            f_mid = kBT_Lp * (1 / (4*(1 - mid/Lc)**2) - 0.25 + mid/Lc)
            if f_mid < F_N:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    forces_pN = [0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 40]

    print(f"{'F (pN)':<10} {'L_ext (µm)':<14} {'<x²> (nm²)':<16} {'σ_x (nm)':<14} {'<y²> (nm²)':<16} {'σ_y (nm)'}")
    print("-" * 82)

    for F_pN in forces_pN:
        F    = F_pN * 1e-12
        Lext = vind_Lext(F)

        var_x = kBT * Lext / F           # m²
        var_y = kBT * (Lext + R) / F     # m²  (y-as: inclusief straal bead)

        print(f"{F_pN:<10} "
              f"{Lext*1e6:<14.3f} "
              f"{var_x*1e18:<16.0f} "
              f"{math.sqrt(var_x)*1e9:<14.1f} "
              f"{var_y*1e18:<16.0f} "
              f"{math.sqrt(var_y)*1e9:.1f}")


def opdr4():

    def magneet_F(z):
        return 22.3811 * math.exp(-1.4578 * z) + 52.2987 * math.exp(-0.6912 * z)

    def vind_z(F_pN):
        lo, hi = 0.0, 20.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if magneet_F(mid) > F_pN:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    forces_pN = [0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 40]

    print(f"{'F (pN)':<10} {'z (mm)'}")
    print("-" * 22)
    z_vals = []
    for F_pN in forces_pN:
        z = vind_z(F_pN)
        z_vals.append(z)
        print(f"{F_pN:<10} {z:.3f}")

    # Plot
    z_curve = [i * 0.05 for i in range(1, 260)]   # 0.05 tot 13 mm
    F_curve = [magneet_F(z) for z in z_curve]

    plt.figure(figsize=(7, 5))
    plt.plot(z_curve, F_curve, label="$F(z)$")
    plt.plot(z_vals, forces_pN, "ro", label="kalibratiepunten")
    plt.xlabel("$z$ (mm)")
    plt.ylabel("$F$ (pN)")
    plt.title("Magneetpositie vs. kracht")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def opdr5():
    kBT    = 4.1e-21
    kBT_Lp = 0.0911e-12
    Lc     = 7e-6
    R      = 1.4e-6

    def vind_Lext(F_N):
        lo, hi = 0.001 * Lc, 0.9999 * Lc
        for _ in range(80):
            mid = (lo + hi) / 2
            f_mid = kBT_Lp * (1 / (4*(1 - mid/Lc)**2) - 0.25 + mid/Lc)
            if f_mid < F_N: lo = mid
            else:           hi = mid
        return (lo + hi) / 2

    F        = 40e-12
    Lext     = vind_Lext(F)
    sigma_x  = (kBT * Lext / F) ** 0.5 * 1e9   # nm
    tracking = 2                                  # nm (gegeven)

    print(f"σ_x bij 40 pN  : {sigma_x:.1f} nm")
    print(f"Tracking noise : {tracking} nm")
    print(f"Verhouding     : {sigma_x/tracking:.1f}x")
    print()
    if sigma_x > 10 * tracking:
        print("→ Tracking noise is GEEN limiterende factor.")
    else:
        print("→ Tracking noise KAN een rol spelen.")
    print()
    print("Andere ruisbronnen:")
    print("  - Mechanische drift van de flowkamer (corrigeer met referentieballetje)")
    print("  - Statistische fout (neemt af met 1/√N datapunten)")
    print("  - Bead-to-bead variatie in magnetisch gehalte (~10%)")


# opdr1()
opdr2()
# opdr3()
# opdr4()
# opdr5()
