import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载音频
audio_path = 'D:\PyCharm工作\AI music\CHEN\Chen.mp3'
y, sr = librosa.load(audio_path)

print("正在分析音频...")
print("=" * 50)

# 1. 波形图
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr, color='blue')
plt.title('古筝音频波形图')
plt.xlabel('时间 (秒)')
plt.ylabel('振幅')

# 2. 频谱图
plt.subplot(3, 1, 2)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
plt.colorbar(format='%+2.0f dB', label='强度 (dB)')
plt.title('频谱图 (对数频率)')
plt.ylabel('频率 (Hz)')

# 3. Mel频谱图
plt.subplot(3, 1, 3)
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel',
                         fmax=8000, cmap='magma')
plt.colorbar(format='%+2.0f dB', label='强度 (dB)')
plt.title('梅尔频谱图 (Mel Spectrogram)')
plt.ylabel('梅尔频率')

plt.tight_layout()
plt.show()

# 修正节奏分析部分
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
estimated_tempo = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)

# 使用正确的转换方式
hop_length = 512  # 默认值
beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

print(f"估计速度: {estimated_tempo:.2f} BPM")
print(f"理论节拍间隔: {60/estimated_tempo:.3f} 秒")
print(f"实际检测节拍数: {len(beat_times)}")
print(f"实际平均间隔: {np.mean(np.diff(beat_times)):.3f} 秒")

# 5. 音频特征分析
# 响度（RMS能量）
rms = librosa.feature.rms(y=y)[0]
times_rms = librosa.times_like(rms, sr=sr)

# 零交叉率（ZCR）
zcr = librosa.feature.zero_crossing_rate(y)[0]
times_zcr = librosa.times_like(zcr, sr=sr)

# 频谱质心
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
times_sc = librosa.times_like(spectral_centroids, sr=sr)

# 6. 可视化所有特征
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# 波形 + 节拍
ax1 = axes[0]
librosa.display.waveshow(y, sr=sr, alpha=0.7, ax=ax1)
ax1.vlines(beat_times, -1, 1, color='red', alpha=0.7, linestyle='--',
           linewidth=1.2, label=f'节拍 ({estimated_tempo:.1f} BPM)')
ax1.set_ylabel('振幅')
ax1.set_title(f'古筝音频波形与节拍检测 (共{len(beat_times)}个节拍)')
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)

# RMS能量
ax2 = axes[1]
ax2.plot(times_rms, rms, color='green', linewidth=2)
ax2.fill_between(times_rms, 0, rms, alpha=0.3, color='green')
ax2.set_ylabel('RMS能量')
ax2.set_title('音频能量变化')
ax2.grid(alpha=0.3)

# 频谱质心
ax3 = axes[2]
ax3.plot(times_sc, spectral_centroids, color='purple', linewidth=2)
ax3.fill_between(times_sc, 0, spectral_centroids, alpha=0.3, color='purple')
ax3.set_ylabel('频率 (Hz)')
ax3.set_title('频谱质心变化 (反映音色亮度)')
ax3.grid(alpha=0.3)

# 零交叉率
ax4 = axes[3]
ax4.plot(times_zcr, zcr, color='orange', linewidth=2)
ax4.fill_between(times_zcr, 0, zcr, alpha=0.3, color='orange')
ax4.set_xlabel('时间 (秒)')
ax4.set_ylabel('零交叉率')
ax4.set_title('零交叉率变化 (反映打击感)')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 7. 音乐结构分析
# 将音频分段分析
duration = len(y) / sr
segment_duration = 30  # 每30秒分析一段
segments = int(np.ceil(duration / segment_duration))

print("音乐结构分析:")
print("=" * 50)

for i in range(segments):
    start_time = i * segment_duration
    end_time = min((i + 1) * segment_duration, duration)

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    if end_sample > len(y):
        break

    segment = y[start_sample:end_sample]

    # 计算段落的平均能量
    segment_rms = librosa.feature.rms(y=segment)[0].mean()

    # 计算段落的平均频谱质心
    segment_sc = librosa.feature.spectral_centroid(y=segment, sr=sr)[0].mean()

    print(f"段落 {i + 1}: {start_time:.0f}-{end_time:.0f}秒")
    print(f"  平均能量: {segment_rms:.4f}")
    print(f"  平均频谱质心: {segment_sc:.1f} Hz")

    if segment_rms > 0.1:
        energy_level = "高"
    elif segment_rms > 0.05:
        energy_level = "中"
    else:
        energy_level = "低"

    if segment_sc > 2000:
        brightness = "明亮"
    elif segment_sc > 1000:
        brightness = "中等"
    else:
        brightness = "低沉"

    print(f"  能量水平: {energy_level}")
    print(f"  音色亮度: {brightness}")
    print("-" * 30)

# 调性分析
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
plt.figure(figsize=(12, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('色谱图 (Chroma) - 调性分析')
plt.show()

# 谐波与打击成分分离
y_harmonic, y_percussive = librosa.effects.hpss(y)
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
librosa.display.waveshow(y_harmonic, sr=sr, alpha=0.7)
plt.title('谐波成分')
plt.subplot(2, 1, 2)
librosa.display.waveshow(y_percussive, sr=sr, alpha=0.7, color='orange')
plt.title('打击成分')
plt.tight_layout()
plt.show()

print("=" * 50)
print("分析完成!")