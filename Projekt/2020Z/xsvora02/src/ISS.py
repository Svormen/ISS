import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter
import winsound

# Open maskoff signal
ext_off_signal = "maskoff_tone.wav"
ext_on_signal = "maskon_tone.wav"

s, d_off = wavfile.read(ext_off_signal)
s1, d_on = wavfile.read(ext_on_signal)
""" ----------------------------------  3  ---------------------------------- """
# extract signal to 1 sec
d_off = d_off[2000:18000]
d_on = d_on[2020:18020]

# Ustrednenie
d_off = d_off - np.mean(d_off)
d_on = d_on - np.mean(d_on)

# normalization to (-1, 1)
d_off = d_off / np.abs(d_off).max()
d_on = d_on / np.abs(d_on).max()

""" check maximum and minimum
x, x1 = max(d), max(d_on)
y, y1 = min(d), min(d_on)
print("MAX d_off je:", x)
print("MIN d_off je:", y)
print("MAX d_on je:", x1)
print("MIN d_on je:", y1)
"""

overlap_off = int(s * 0.01)
overlap_on = int(s1 * 0.01)
fr_size = int(s * 0.02)
#fr_num = (16000 / overlap)

frames_off = np.ndarray((99, 320), float)
frames_on = np.ndarray((99, 320), float)
for i in range(0, 99):
    for j in range(0, 320):
        frames_off[i][j] = d_off[i * overlap_off + j]
        frames_on[i][j] = d_on[i * overlap_on + j]

plt.figure(figsize=(10, 3))
t_off = np.arange(frames_off[0].size) / s
plt.plot(t_off, frames_off[0])
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_ylabel('$y$')
plt.gca().set_title('Bez rúška')
plt.tight_layout()

plt.figure(figsize=(10, 3))
t_on = np.arange(frames_on[0].size) / s1
plt.plot(t_on, frames_on[0])
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_ylabel('$y$')
plt.gca().set_title('S rúškom')
plt.tight_layout()

plt.show()

""" ----------------------------------  4  ---------------------------------- """

clip_off = np.ndarray((99, 320), int)
clip_on = np.ndarray((99, 320), int)

# center clipping
for i in range(0, 99):
    max_off = max(abs(frames_off[i]))
    max_off = max_off * 0.7
    max_on = max(abs(frames_on[i]))
    max_on = max_on * 0.7
    for j in range(0, 320):
        if frames_off[i][j] > max_off:
            clip_off[i][j] = 1
        elif frames_off[i][j] < -max_off:
            clip_off[i][j] = -1
        else:
            clip_off[i][j] = 0

        if frames_on[i][j] > max_on:
            clip_on[i][j] = 1
        elif frames_on[i][j] < -max_on:
            clip_on[i][j] = -1
        else:
            clip_on[i][j] = 0

plt.figure(figsize=(10, 3))
t_off = np.arange(clip_off[0].size) / s
plt.plot(t_off, clip_off[0])
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_ylabel('$y$')
plt.gca().set_title('Central clipping bez rúška')
plt.tight_layout()

plt.figure(figsize=(10, 3))
t_on = np.arange(clip_on[0].size) / s1
plt.plot(t_on, clip_on[0])
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_ylabel('$y$')
plt.gca().set_title('Central clipping s rúškom')
plt.tight_layout()

plt.show()

# autokorelacia ramca

autocor_off = np.ndarray((99, 320), int)
autocor_on = np.ndarray((99, 320), int)

for i in range(0, 99):
    for j in range(0, 320):
        autocor_off[i][j] = 0
        autocor_on[i][j] = 0
        for k in range(0, 319 - j):
            result_off = clip_off[i][k] * clip_off[i][k + j]
            autocor_off[i][j] = autocor_off[i][j] + result_off
            result_on = clip_on[i][k] * clip_on[i][k + j]
            autocor_on[i][j] = autocor_on[i][j] + result_on

#plt.plot(np.correlate(clip_on[0], clip_on[0], mode='full')[319:])
#plt.plot(np.correlate(clip_off[0], clip_off[0], mode='full')[319:])

# obsahuje casti z prikladu 12 - same_lag_on/off
lag_on = np.array([0]*99)
lag_off = np.array([0]*99)
same_lag_on = np.array([0]*99)
same_lag_off = np.array([0]*99)

for i in range(0, 99):
    lag_on[i] = np.argmax(autocor_on[i][10:]) + 10
    lag_off[i] = np.argmax(autocor_off[i][10:]) + 10
    same_lag_on[i] = np.argmax(autocor_on[i][147:]) + 147
    same_lag_off[i] = np.argmax(autocor_off[i][149:]) + 149

plt.figure(figsize=(10, 3))
t_on = np.arange(autocor_on[0].size)
plt.plot(t_on, autocor_on[0])
plt.axvline(x=10, c='k', label="Práh")
plt.stem([lag_on[0]], [autocor_on[0][lag_on[0]]], label="Lag")
plt.legend()
plt.gca().set_xlabel('$vzorky$')
plt.gca().set_ylabel('$y$')
plt.gca().set_title('Autokolerácia s rúškom')
plt.tight_layout()

plt.figure(figsize=(10, 3))
t_off = np.arange(autocor_off[0].size)
plt.plot(t_off, autocor_off[0])
plt.gca().set_xlabel('$vzorky$')
plt.gca().set_ylabel('$y$')
plt.gca().set_title('Autokolerácia bez rúška')
plt.tight_layout()

plt.show()

# Zakladna frekvencia ramcov
# 1 / lag * 16000

# graf pre 12 priklad
plt.figure(figsize=(10, 3))
t_off = np.arange(autocor_on[6].size)
plt.plot(t_off, autocor_on[6])
plt.axvline(x=147, c='k', label="Práh")
plt.legend()
plt.gca().set_xlabel('$vzorky$')
plt.gca().set_ylabel('$y$')
plt.gca().set_title('Autokolerácia s rúškom + nájdený lag')
plt.tight_layout()

plt.show()

f0_off = np.array([0]*99)
f0_on = np.array([0]*99)

for i in range(0, 99):
    f0_off[i] = (1 / lag_off[i]) * s
    f0_on[i] = (1 / lag_on[i]) * s1

plt.figure(figsize=(10, 3))
t_off = np.arange(f0_off.size)
t_on = np.arange(f0_on.size)
plt.plot(t_off, f0_off, label="Bez rúška")
plt.plot(t_on, f0_on, label="S rúškom")
plt.legend()
plt.gca().set_xlabel('$rámce$')
plt.gca().set_ylabel('$f0$')
plt.gca().set_title('Základná frekvencia rámcov')
plt.tight_layout()
plt.show()


# rozptyl
rozptyl_off = np.var(f0_off)
rozptyl_on = np.var(f0_on)
print("R_OFF", rozptyl_off)
print("R_ON", rozptyl_on)

# stredna hodnota
str_hod_off = np.median(f0_off)
str_hod_on = np.median(f0_on)
print("S_off", str_hod_off)
print("S_ON", str_hod_on)

""" ----------------------------------  5  ---------------------------------- """
dft_off = np.zeros((99, 1024), dtype=complex)
dft_on = np.zeros((99, 1024), dtype=complex)
dft_f_off = np.ndarray((99, 1024), dtype=complex)
dft_f_on = np.ndarray((99, 1024), dtype=complex)
ddft_f_off = np.ndarray((99, 1024), dtype=complex)
ddft_f_on = np.ndarray((99, 1024), dtype=complex)

for i in range(0, 99):
    for j in range(0, 320):
        dft_off[i][j] = frames_off[i][j]
        dft_on[i][j] = frames_on[i][j]

#for i in range(0, 99):
#    for j in range(0, 1024):
#        for k in range(0, 1024):
#            dft_f_off[i][j] = dft_f_off[i][j] + dft_off[i][k] * (np.exp((-1j * 2 * np.pi * k * j) / 1024))
#            dft_f_on[i][j] = dft_f_on[i][j] + dft_on[i][k] * (np.exp((-1j * 2 * np.pi * k * j) / 1024))

for i in range(0, 99):
    ddft_f_off[i] = np.fft.fft(frames_off[i], 1024)
    ddft_f_on[i] = np.fft.fft(frames_on[i], 1024)

# P[k] = 10 log10 |X[k]|^2
log_off = np.ndarray((99, 1024), float)
log_on = np.ndarray((99, 1024), float)
for i in range(0, 99):
    log_off[i] = 10 * np.log10(np.abs(ddft_f_off[i])**2)
    log_on[i] = 10 * np.log10(np.abs(ddft_f_on[i])**2)

# k = [0..512]
log_off_k = np.ndarray((99, 512), float)
log_on_k = np.ndarray((99, 512), float)
for i in range(0, 99):
    for j in range(0, 512):
        log_off_k[i][j] = log_off[i][j]
        log_on_k[i][j] = log_on[i][j]

# transpose
log_off_trans = log_off_k.transpose()
log_on_trans = log_on_k.transpose()

# graph without mask
plt.figure(figsize=(8, 4))
leg_off = plt.imshow(log_off_trans, extent=[0, 1, 0, 8000], origin='lower', aspect='auto')
plt.colorbar(leg_off)
plt.gca().set_xlabel('$čas[s]$')
plt.gca().set_ylabel('$frekvencia$')
plt.gca().set_title('Spektogram bez rúška')
plt.tight_layout()

# graph with mask
plt.figure(figsize=(8, 4))
leg_on = plt.imshow(log_on_trans, extent=[0, 1, 0, 8000], origin='lower', aspect='auto')
plt.colorbar(leg_on)
plt.gca().set_xlabel('$čas[s]$')
plt.gca().set_ylabel('$frekvencia$')
plt.gca().set_title('Spektogram s rúškom')
plt.tight_layout()

plt.show()
""" ----------------------------------  6  ---------------------------------- """
# frekvenčná charakteristika
f_ch = np.ndarray((99, 1024), dtype=complex)

f_ch = ddft_f_on / ddft_f_off
f_ch_average = np.mean(np.abs(f_ch), axis=0)

f_ch_average_f = 10 * np.log10(np.abs(f_ch_average)**2)

# graph inspired by jupyter notes
plt.figure(figsize=(10, 3))
t = np.arange(f_ch_average_f.size) / 1024 * 16000
plt.plot(t[:t.size//2+1], f_ch_average_f[:f_ch_average_f.size//2+1])
plt.gca().set_xlabel('$Frekvencia [Hz]$')
plt.gca().set_ylabel('$Spektrálna hustota výkonu [dB]$')
plt.gca().set_title('Frekvenčná charakteristika')
plt.tight_layout()
plt.show()

""" ----------------------------------  7  ---------------------------------- """
# IDFT
i_dft = np.ndarray(1024, dtype=complex)
i_dft_my = np.zeros(1024, dtype=complex)

i_dft = np.fft.ifft(f_ch_average)

"""
for i in range(0, 1024):
    i_dft_my[i] = 0
    for j in range(0, 1024):
        i_dft_my[i] = i_dft_my[i] + f_ch_average[j] * (np.exp((1j * 2 * np.pi * j * i) / 1024))

for i in range(0, 1024):
    i_dft_my[i] = i_dft_my[i] / 1024
"""
i_dft = i_dft[:512]
#i_dft_log = 10 * np.log10(np.abs(i_dft_my)**2)

plt.figure(figsize=(8, 4))
t = np.arange(i_dft.size)
plt.plot(t, i_dft.real)
plt.gca().set_xlabel('$čas$')
plt.gca().set_ylabel('$y$')
plt.gca().set_title('Impúlzna odozva rúška')
plt.tight_layout()

plt.show()
""" ----------------------------------  8  ---------------------------------- """
ton_off = "maskoff_tone.wav"
ton_on = "maskon_tone.wav"

st, dt_off = wavfile.read(ton_off)
s1t, dt_on = wavfile.read(ton_on)


veta_off = "maskoff_sentence.wav"
veta_on = "maskon_sentence.wav"

sv, dv_off = wavfile.read(veta_off)
s1v, dv_on = wavfile.read(veta_on)

# Ustrednenie
dv_off = dv_off - np.mean(dv_off)
dt_off = dt_off - np.mean(dt_off)
# normalization
dv_off = dv_off / np.abs(dv_off).max()
dt_off = dt_off / np.abs(dt_off).max()

sim_v = lfilter(i_dft[:512].real, [1], dv_off)
sim_t = lfilter(i_dft[:512].real, [1], dt_off)

plt.figure(figsize=(10, 3))
v_off = np.arange(dv_off.size) / sv
plt.plot(v_off, dv_off)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_ylabel('$y$')
plt.gca().set_title('Veta bez rúška')
plt.tight_layout()

plt.figure(figsize=(10, 3))
v_on = np.arange(dv_on.size) / s1v
plt.plot(v_on, dv_on)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_ylabel('$y$')
plt.gca().set_title('Veta s rúškom')
plt.tight_layout()

plt.show()

plt.figure(figsize=(10, 3))
v_off = np.arange(sim_v.size) / sv
plt.plot(v_off, sim_v.real, label="Veta bez rúška s filtrom")
plt.plot(v_off, dv_off, label="Veta bez rúška")
plt.legend()
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_ylabel('$y$')
plt.gca().set_title('Simulovaná veta')
plt.tight_layout()

plt.show()

wavfile.write("sim_maskon_sentence.wav", 16000, sim_v.real)
wavfile.write("sim_maskon_tone.wav", 16000, sim_t.real)

""" ----------------------------------  13  ---------------------------------- """
same_frames_off = np.ndarray((16, 320), float)
same_frames_on = np.ndarray((16, 320), float)

same_dft_off = np.ndarray((16, 1024), dtype=complex)
same_dft_on = np.ndarray((16, 1024), dtype=complex)

count = 0
for i in range(0, 99):
    if f0_off[i] == f0_on[i]:
        same_frames_off[count] = frames_off[i]
        same_frames_on[count] = frames_on[i]
        count = count + 1

# DFT
for i in range(0, 16):
    same_dft_off[i] = np.fft.fft(same_frames_off[i], 1024)
    same_dft_on[i] = np.fft.fft(same_frames_on[i], 1024)

# frekvenčná charakteristika
same_f_ch = np.ndarray((16, 1024), dtype=complex)

same_f_ch = same_dft_on / same_dft_off
same_f_ch_average = np.mean(np.abs(same_f_ch), axis=0)
same_f_ch_average_f = 10 * np.log10(np.abs(same_f_ch_average)**2)

# graph inspired by jupyter notes
plt.figure(figsize=(10, 3))
t = np.arange(same_f_ch_average_f.size) / 1024 * 16000
plt.plot(t[:t.size//2+1], same_f_ch_average_f[:same_f_ch_average_f.size//2+1])
plt.gca().set_xlabel('$Frekvencia [Hz]$')
plt.gca().set_ylabel('$Spektrálna hustota výkonu [dB]$')
plt.gca().set_title('Frekvenčná charakteristika only match')
plt.tight_layout()
plt.show()

# IDFT
same_i_dft = np.ndarray(1024, dtype=complex)

same_i_dft = np.fft.ifft(same_f_ch_average)

same_ton_off = "maskoff_tone.wav"
same_ton_on = "maskon_tone.wav"

sst, sdt_off = wavfile.read(same_ton_off)
ss1t, sdt_on = wavfile.read(same_ton_on)

same_veta_off = "maskoff_sentence.wav"
same_veta_on = "maskon_sentence.wav"

ssv, sdv_off = wavfile.read(same_veta_off)
ss1v, sdv_on = wavfile.read(same_veta_on)

# Ustrednenie
sdt_off = sdt_off - np.mean(sdt_off)
sdv_off = sdv_off - np.mean(sdv_off)
# normalization
sdt_off = sdt_off / np.abs(sdt_off).max()
sdv_off = sdv_off / np.abs(sdv_off).max()

same_sim_t = lfilter(same_i_dft[:512].real, [1], sdt_off)
same_sim_v = lfilter(same_i_dft[:512].real, [1], sdv_off)

wavfile.write("sim_maskon_tone_only_match.wav", 16000, same_sim_t.real)
wavfile.write("sim_maskon_sentence_only_match.wav", 16000, same_sim_v.real)

""" ----------------------------------  12  ---------------------------------- """
# KOD JE DOPLNENY V PRIKLADE 4

""" ----------------------------------  11  ---------------------------------- """
b_m = np.ndarray((99, 320))
b_m_dft_on = np.ndarray((99, 1024), dtype=complex)
b_m_dft_off = np.ndarray((99, 1024), dtype=complex)

for i in range(0, 99):
    b_m[i] = np.blackman(320)

# DFT, Log
b_m_dft = np.fft.fft(b_m, 1024)
b_m_dft_a = np.mean(np.abs(b_m_dft), axis=0)
b_m_dft_f = 10 * np.log10(np.abs(b_m_dft_a)**2)

b_m_frames_on = frames_on * b_m
b_m_frames_off = frames_off * b_m

for i in range(0, 99):
    b_m_dft_on[i] = np.fft.fft(b_m_frames_on[i], 1024)
    b_m_dft_off[i] = np.fft.fft(b_m_frames_off[i], 1024)

b_m_result = b_m_dft_on/b_m_dft_off
b_m_result_a = np.mean(np.abs(b_m_result), axis=0)
b_m_result_f = 10 * np.log10(np.abs(b_m_result_a)**2)

# IDFT
b_m_idft = np.fft.ifft(b_m_result_a, 1024)

b_m_tone_off = "maskoff_tone.wav"
b_m_sentence_off = "maskoff_sentence.wav"

bmt, bmt_off = wavfile.read(b_m_tone_off)
bms, bms_off = wavfile.read(b_m_sentence_off)

bmt_off = bmt_off - np.mean(bmt_off)
bms_off = bms_off - np.mean(bms_off)

bmt_off = bmt_off / np.abs(bmt_off).max()
bms_off = bms_off / np.abs(bms_off).max()

sim_bm_t = lfilter(b_m_idft[:512].real, [1], bmt_off)
sim_bm_s = lfilter(b_m_idft[:512].real, [1], bms_off)

wavfile.write("sim_maskon_tone_window.wav", 16000, sim_bm_t.real)
wavfile.write("sim_maskon_sentence_window.wav", 16000, sim_bm_s.real)

# grafy
plt.figure(figsize=(10, 3))
bm = np.arange(b_m[0].size) / bmt
plt.plot(bm, b_m[0])
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_ylabel('$y$')
plt.gca().set_title('Časová oblasť Blackman')
plt.tight_layout()

plt.figure(figsize=(10, 3))
bm1 = np.arange(b_m_dft_f.size) / b_m_dft_a.size * 16000
plt.plot(bm1[:bm1.size//2], b_m_dft_f[:b_m_dft_f.size//2])
plt.gca().set_xlabel('$Frekvencia [Hz]$')
plt.gca().set_ylabel('$[dB]$')
plt.gca().set_title('Frekvenčná charakteristika s Blackman')
plt.tight_layout()

plt.figure(figsize=(10, 3))
bm2 = np.arange(b_m_result_f.size) / b_m_result_a.size * 16000
plt.plot(bm2[:bm2.size//2], b_m_result_f[:b_m_result_f.size//2], label="Blackman")
plt.plot(t[:t.size//2], f_ch_average_f[:f_ch_average_f.size//2], label="Pôvodná")
plt.legend()
plt.gca().set_xlabel('$Frekvencia [Hz]$')
plt.gca().set_ylabel('$[dB]$')
plt.gca().set_title('Frekvenčná charakteristika rúška Blackman')
plt.tight_layout()

plt.show()
