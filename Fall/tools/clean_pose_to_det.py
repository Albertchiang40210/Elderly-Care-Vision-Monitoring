# clean_pose_to_det.py
import os
import shutil

src_dir = "./active_learning_dataset"
dst_dir = "./detection_dataset"

# 1. 複製整個結構（包含圖片）
if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
shutil.copytree(src_dir, dst_dir)

print("📁 資料夾複製完成，開始剝離 Pose 關鍵點...")

# 2. 清洗 labels 底下的所有 .txt
labels_path = os.path.join(dst_dir, "labels")
if os.path.exists(labels_path):
    for filename in os.listdir(labels_path):
        if filename.endswith(".txt"):
            file_key = os.path.join(labels_path, filename)
            
            with open(file_key, "r") as f:
                lines = f.readlines()
            
            clean_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # 關鍵動作：只切前 5 個欄位 (class, x, y, w, h)
                    clean_lines.append(" ".join(parts[:5]))
            
            with open(file_key, "w") as f:
                f.write("\n".join(clean_lines))

print("🎯 降維清洗完成！已生成純偵測資料集於: ./detection_dataset")


#「它是我們資料集從『骨架格式』瘦身成『純偵測方框』的資料降維魔法師。」
#這是一個標準的資料前處理（Preprocessing）腳本。
#因為你在前線採集的是 YOLOv11-pose 的資料（裡面除了有框，還包含了一大堆關節點的座標，像是 [class, x, y, w, h, px1, py1, pv1, ...]）。但當你要拿這些資料去重新訓練後台的 RT-DETR（DEIM） 時，RT-DETR 不需要、也看不懂那些關節點，它只需要最基本的「方框座標」（也就是前 5 個欄位：class, x, y, w, h）。
#這個檔案做的事就是：
#先把整個骨架資料集複製一份（避免弄壞原始檔案）。
#像外科手術一樣，去讀取裡面所有的標註文字檔（.txt），把後面的關節點座標通通切掉，只保留前 5 個方框座標。
#產生出一個乾淨、專門給 RT-DETR 訓練用的「純偵測資料集」！