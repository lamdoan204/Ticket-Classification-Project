import re
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ==============================================
# 2️⃣ Dữ liệu ví dụ
# ==============================================
comments = [
    "Sản phẩm rất tốt", 
    "Giao hàng nhanh và đóng gói cẩn thận", 
    "Chất lượng kém, không đáng tiền", 
    "Hàng quá tệ, tôi thất vọng", 
    "Shop phục vụ nhiệt tình", 
    "Giá rẻ mà dùng bền", 
    "Không như mô tả, rất thất vọng"
]
labels = [1, 1, 0, 0, 1, 1, 0]  # 1: tích cực, 0: tiêu cực

# ==============================================
# 3️⃣ Làm sạch & Tokenization
# ==============================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zà-ỹ0-9\s]', '', text)
    return text.split()

tokenized_comments = [clean_text(c) for c in comments]

print("👉 Sau khi tokenization:")
for i, c in enumerate(tokenized_comments):
    print(f"{i+1}. {c}")

# # ==============================================
# # 4️⃣ Huấn luyện Word2Vec
# # ==============================================
# w2v_model = Word2Vec(sentences=tokenized_comments, vector_size=100, window=5, min_count=1, sg=1)

# print("\n👉 Ví dụ vector Word2Vec của từ 'tốt':")
# print(w2v_model.wv['tốt'][:10])  # in 10 giá trị đầu

# # ==============================================
# # 5️⃣ Tạo tokenizer và embedding matrix
# # ==============================================
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts([' '.join(c) for c in tokenized_comments])
# vocab_size = len(tokenizer.word_index) + 1
# embedding_dim = 100

# embedding_matrix = np.zeros((vocab_size, embedding_dim))
# for word, i in tokenizer.word_index.items():
#     if word in w2v_model.wv:
#         embedding_matrix[i] = w2v_model.wv[word]

# print("\n👉 Số lượng từ trong vocabulary:", vocab_size)
# print("Một vài từ trong từ điển:", list(tokenizer.word_index.items())[:10])