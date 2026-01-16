import librosa
import numpy as np


def extract_features(audio_path):
    """
    从单个音频文件中提取特征。
    参数:
        audio_path: 音频文件的路径
    返回:
        features: 一个字典，包含所有提取的特征
    """
    # 1. 加载音频
    y, sr = librosa.load(audio_path)

    # 2. 初始化特征字典
    features = {}

    # 3. 提取你已有的特征（复制并调整你的代码）
    # 例如，提取节拍
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)

    # 提取RMS能量（取平均值作为该片段特征）
    rms = librosa.feature.rms(y=y).mean()
    features['rms_mean'] = float(rms)

    # 提取频谱质心（取平均值）
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    features['spectral_centroid_mean'] = float(spectral_centroids)

    # 你可以继续添加更多特征，如零交叉率、频谱带宽等

    # 4. 新增：提取梅尔频谱图并扁平化（作为深度学习备用特征）
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    # 这里可以计算一些统计量，如均值、方差，或直接返回扁平化的部分数据
    features['mel_spec_mean'] = float(mel_spec_db.mean())
    features['mel_spec_std'] = float(mel_spec_db.std())

    return features


# 测试这个函数
if __name__ == "__main__":
    # 用你已有的Chen.mp3测试
    test_features = extract_features("Chen.mp3")
    print("提取的特征示例：")
    for key, value in test_features.items():
        print(f"  {key}: {value:.4f}")