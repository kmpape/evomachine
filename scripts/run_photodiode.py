import numpy as np
import matplotlib.pyplot as plt
import sys
import time

sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/asitiger")
sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo")
sys.path.append("/home/hslab/workspace_python/conda_evomachine3.9/de-lta-rt")

from evomachine.evotypes import LEDType  # noqa
from syncboard.syncboardcontroller import SyncBoardController  # noqa
from syncboard.command import Command
from evomachine.dmd_socket import DMDControl  # noqa

led_channel_keys: dict[LEDType, int | None] = {
    LEDType.LED_385_NM: 7,
    LEDType.LED_450_NM: 1,
    LEDType.LED_515_NM: 2,
    LEDType.LED_560_NM: 3,
    LEDType.LED_625_NM: 4,
    LEDType.NO_LED: None,
}

ctr = SyncBoardController.from_serial_port(port='/dev/ttyACM0')
ctr.initialise()

dmd = DMDControl()
dmd.initialise()
dmd.display_full()


# TEST CODE for getting singular measurements
# def get_meas_avg(num_avg = 50) -> float | None:
#     i_valid = 0
#     tot = 0
#     for i in range(num_avg):
#         res = ctr.read_photodiode()
#         if res is not None:
#             tot += res
#             i_valid += 1
#     return tot / float(i_valid) if i_valid > 0 else None
#
# for intensity in [0, 0.01, 0.1, 0.2]:
#     if intensity > 0:
#         ctr.enable_led(led_id=led_channel_keys[LEDType.LED_450_NM], intensity=intensity)
#     else:
#         ctr.disable_led()
#     time.sleep(1)
#     meas = get_meas_avg(num_avg = 1)
#     print(f"intensity={intensity}, meas={meas}")


def get_meas_array(num_samples: int = 100, fs: float = 100) -> np.ndarray:
    if fs > 0:
        assert fs <= 400
        Ts = 1 / float(fs)
    else:
        Ts = 0
    res = np.zeros((num_samples, 2))
    for i in range(num_samples):
        start_time = time.perf_counter()
        res[i, 1] = ctr.read_photodiode()
        res[i, 0] = time.perf_counter()
        while fs > 0 and time.perf_counter() - start_time < Ts:
            pass
    res[:, 0] = res[:, 0] - res[0, 0]
    if fs > 0:
        bad_meas = res[:, 1] <= 0
        res = res[~bad_meas, :]
        t = np.arange(0, num_samples*Ts, Ts)
        res_interp = np.interp(t, res[:, 0], res[:, 1])
        res = np.zeros((num_samples, 2))
        res[:, 0] = t
        res[:, 1] = res_interp
    return res


intensities = [0, 0, 0.25, 0.25]
dmd_display = ['None', 'Full', 'None', 'Full']
num_fft_avg = 20
N_base = 100
N = N_base * num_fft_avg
fs = 100
frequencies = np.fft.rfftfreq(N_base, 1/fs)
fft_mags = np.zeros((len(frequencies), len(intensities)))
measurements = np.zeros((N_base, len(intensities)))
time_arrays = np.zeros((N_base, len(intensities)))
normalise_FFT = False
for i, (intensity, dd) in enumerate(zip(intensities, dmd_display, strict=True)):
    print(f"i={i+1} out of {len(intensities)}")
    if dd == 'None':
        dmd.display_none()
    else:
        dmd.display_full()
    if intensity > 0:
        ctr.enable_led(led_id=led_channel_keys[LEDType.LED_450_NM], intensity=intensity)
    else:
        ctr.disable_led()
    time.sleep(1)
    tmp = np.zeros((len(frequencies)), np.dtype('complex128'))
    for j in range(num_fft_avg):
        meas = get_meas_array(num_samples=N_base, fs=fs)
        tmp = tmp + np.fft.rfft(meas[:, 1])
        if j == 0:
            measurements[:, i] = meas[:, 1]
            time_arrays[:, i] = meas[:, 0]
    fft_mags[:, i] = np.abs(tmp) / num_fft_avg
    if normalise_FFT:
        fft_mags[:, i] = fft_mags[:, i] / fft_mags[1:, i].max()


fig, ax = plt.subplots(figsize=(10, 10))
for i, (intensity, dd) in enumerate(zip(intensities, dmd_display, strict=True)):
    label = f"LED={np.round(intensity*100,2)}%, DMD={dd}"
    ax.plot(frequencies[0:], fft_mags[0:, i], label=label)

ax.set_yscale('log')
ax.set_xlim(0, 50)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(f"Amplitude (normalised={normalise_FFT})")
# ax.set_title('With video mode (HDMI)')
# ax.set_title('With pattern-on-the-fly mode')
# ax.set_title('With video mode (solid field)')
ax.set_title('With video pattern mode')
ax.legend()
ax.grid(True)
plt.show()


fig, ax = plt.subplots(figsize=(10, 10))
for i, (intensity, dd) in enumerate(zip(intensities, dmd_display, strict=True)):
    label = f"LED={np.round(intensity*100,2)}%, DMD={dd}"
    ax.plot(time_arrays[0:, i], measurements[0:, i], label=label)

# ax.set_yscale('log')
ax.set_xlim(0, time_arrays.max())
ax.set_xlabel('Time (s)')
ax.set_ylabel(f"Amplitude (-)")
# ax.set_title('With video mode (HDMI)')
# ax.set_title('With pattern-on-the-fly mode')
# ax.set_title('With video mode (solid field)')
ax.set_title('With video pattern mode')
ax.legend()
ax.grid(True)
plt.show()
