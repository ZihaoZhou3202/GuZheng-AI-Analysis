import os
import pandas as pd
from feature_extractor import extract_features  # 导入你刚写好的函数


def create_dataset(audio_folder, label):
    """
    处理一个文件夹下的所有音频，并赋予相同标签
    audio_folder: 存放音频的文件夹路径
    label: 这组音频的标签，如 'calm' 或 'energetic'
    """
    all_features = []

    # 遍历文件夹内所有支持的文件
    for filename in os.listdir(audio_folder):
        if filename.endswith(('.wav', '.mp3', '.flac')):  # 支持常见格式
            audio_path = os.path.join(audio_folder, filename)
            print(f"正在处理: {filename}")

            try:
                # 提取特征
                features = extract_features(audio_path)
                # 添加标签
                features['label'] = label
                all_features.append(features)
            except Exception as e:
                print(f"  处理 {filename} 时出错: {e}")

    # 转换为Pandas DataFrame（表格）
    df = pd.DataFrame(all_features)
    return df


# 示例：明天录完音后，你会这样用
if __name__ == "__main__":
    # 假设你明天录了两类曲子，放在两个文件夹里
    df_calm = create_dataset('./data/calm/', label='calm')
    df_energetic = create_dataset('./data/energetic/', label='energetic')

    # 合并两个表格
    final_df = pd.concat([df_calm, df_energetic], ignore_index=True)

    # 保存为CSV文件，用于后续机器学习
    final_df.to_csv('guzheng_emotion_dataset.csv', index=False)
    print("数据集已保存为 'guzheng_emotion_dataset.csv'")

    print("批量处理框架已就绪。请明天录制音频后，取消注释上面的代码并修改文件夹路径。")